from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812


def normalize_to_scale(latent, target_min, target_max, *, dim=(-3, -2, -1)):
    min_val, max_val = (
        latent.amin(dim=dim, keepdim=True),
        latent.amax(dim=dim, keepdim=True),
    )
    normalized = (latent - min_val).div_(max_val - min_val)
    return (
        normalized.mul_(target_max - target_min)
        .add_(target_min)
        .clamp_(target_min, target_max)
    )


# Referenced/modified from https://discuss.pytorch.org/t/filtered-mean-and-std-from-a-tensor/147258
def nanstd(
    o: torch.Tensor,
    dim: tuple[int, ...] | int,
    keepdim: bool = False,  # noqa: FBT001
) -> torch.Tensor:
    return torch.nanmean(
        (o - torch.nanmean(o, dim=dim, keepdim=True)).pow_(2.0),
        dim=dim,
        keepdim=keepdim,
    ).sqrt_()


def fixup_waveform(
    waveform: torch.Tensor,
    *,
    copy: bool = True,
    move_to_cpu: bool = True,
    ensure_stereo: bool = False,
) -> torch.Tensor:
    if move_to_cpu:
        waveform = waveform.to(device="cpu", copy=copy)
    if waveform.ndim == 2:
        waveform = waveform[None]
    elif waveform.ndim == 1:
        waveform = waveform[None, None]
    if ensure_stereo and waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    return waveform


class TieredBlendWrapper:
    def __init__(
        self,
        blend_function: Callable,
        *,
        tiers: int = 16,
        start_dim: int = 1,
        end_dim: int = -1,
        descending: bool = True,
        abs_mode: bool = False,
        # a, b, add, sub, lerp (50% LERP)
        sort_target: str = "a",
        pad_value: float = 0.0,
    ):
        """Wraps any blending function to operate on 'Probability Tiers'.

        :param blend_function: Callable with signature (a, b, blend_ratio)
        :param tiers: Number of fake 'channels' or tiers to divide the data into.
        :param start_dim: The first dimension to flatten.
        :param end_dim: The last dimension to flatten.
        :param descending: Sort highest-to-lowest (True) or lowest-to-highest (False).
        :param pad_value: Value to pad with if the flattened size isn't divisible by tiers.
                          (For logits, -math.inf might be better, but 0.0 is safe for latents).
        """
        self.blend_function = blend_function
        self.tiers = tiers
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.descending = descending
        self.abs_mode = abs_mode
        self.sort_target = sort_target
        self.pad_value = pad_value

    def get_target(self, a_flat: torch.Tensor, b_flat: torch.Tensor) -> torch.Tensor:
        starget = self.sort_target
        if starget == "a":
            return a_flat
        if starget == "b":
            return b_flat
        if starget == "add":
            return a_flat + b_flat
        if starget == "sub":
            return a_flat - b_flat
        if starget == "lerp":
            return a_flat.lerp(b_flat, 0.5)
        raise ValueError("Invalid sort target")

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor | float,
        t: torch.Tensor | float,
        **kwargs: dict,
    ) -> torch.Tensor:
        if self.tiers < 1:
            return self.blend_function(a, b, t, **kwargs)

        orig_shape = a.shape

        start_dim, end_dim = (
            d if d >= 0 else a.ndim + d for d in (self.start_dim, self.end_dim)
        )
        if any(d < 0 or d > a.ndim for d in (start_dim, end_dim)):
            raise ValueError("Dimension out of range")

        b = (
            b.broadcast_to(orig_shape)
            if isinstance(b, torch.Tensor)
            else torch.full_like(a, fill_value=b)
        )

        a_flat = a.flatten(start_dim=start_dim, end_dim=end_dim)
        b_flat = b.flatten(start_dim=start_dim, end_dim=end_dim)

        t_is_tensor = isinstance(t, torch.Tensor) and t.numel() > 1
        if t_is_tensor:
            t_flat = t.broadcast_to(orig_shape).flatten(
                start_dim=start_dim,
                end_dim=end_dim,
            )
        else:
            t_flat = t

        a_flat = a_flat.transpose(start_dim, -1)
        b_flat = b_flat.transpose(start_dim, -1)
        if t_is_tensor:
            t_flat = t_flat.transpose(start_dim, -1)

        length = a_flat.shape[-1]

        target = self.get_target(a_flat, b_flat)
        if self.abs_mode:
            target = target.abs()

        target_vals, indices = torch.sort(target, dim=-1, descending=self.descending)
        del target
        if not self.abs_mode and self.sort_target == "a":
            a_vals = target_vals
        else:
            del target_vals
            a_vals = torch.gather(a_flat, dim=-1, index=indices)
        b_vals = torch.gather(b_flat, dim=-1, index=indices)
        if t_is_tensor:
            t_vals = torch.gather(t_flat, dim=-1, index=indices)

        # Pad if length not divisible by tiers.
        pad_len = (self.tiers - (length % self.tiers)) % self.tiers
        if pad_len > 0:
            a_vals = F.pad(a_vals, (0, pad_len), value=self.pad_value)
            b_vals = F.pad(b_vals, (0, pad_len), value=self.pad_value)
            if t_is_tensor:
                t_vals = F.pad(t_vals, (0, pad_len), value=self.pad_value)

        # Reshape into tiers (e.g., [..., length] -> [..., tiers, features])
        new_shape = (*a_vals.shape[:-1], self.tiers, -1)
        a_tiered = a_vals.reshape(new_shape)
        b_tiered = b_vals.reshape(new_shape)
        effective_t = t_vals.reshape(new_shape) if t_is_tensor else t

        blended_tiered = self.blend_function(a_tiered, b_tiered, effective_t, **kwargs)

        blended_flat = blended_tiered.reshape(a_vals.shape)

        if pad_len > 0:
            blended_flat = blended_flat[..., :-pad_len]

        # Scatter back to original element positions
        result_flat = torch.empty_like(blended_flat)
        result_flat.scatter_(dim=-1, index=indices, src=blended_flat)

        return result_flat.transpose(start_dim, -1).reshape(orig_shape)
