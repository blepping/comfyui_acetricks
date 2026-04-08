from collections.abc import Callable
from typing import Any, NamedTuple

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


class GlobalProjection(NamedTuple):
    # Shape: [2048, out_channels] (assuming using the Ace15 quantizer projection)
    projection_matrix: torch.Tensor
    # Shape: [2048]
    global_mean: torch.Tensor

    def clone(self) -> "GlobalProjection":
        return self.__class__(
            *(v.clone() if isinstance(v, torch.Tensor) else v for v in self),
        )

    def to(self, *args: Any, **kwargs: Any) -> "GlobalProjection":
        return self.__class__(
            *(
                v.to(*args, **kwargs) if isinstance(v, torch.Tensor) else v
                for v in self
            ),
        )

    @classmethod
    def build(
        cls,
        *,
        universe: torch.Tensor | None = None,
        # PCA is random, so this matters even if you don't generate
        # a random universe.
        seed: int = 42,
        # In channels only used if generating a random "universe".
        in_channels: int = 2048,
        out_channels: int = 192,
        ica_iterations: int = 10,
        # device/dtype only used when generating a random universe.
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "GlobalProjection":

        # 1. RANDOM PROJECTION FALLBACK
        # Used if no universe is provided. Much like our real universe,
        # there's no meaning to be found so we never PCA or ICA here.
        if universe is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

            # Generate random projection and Orthogonalize it (QR)
            rand_mat = torch.randn(
                in_channels,
                out_channels,
                device=generator.device,
                dtype=dtype,
                generator=generator,
            )
            q_mat = torch.linalg.qr(rand_mat)[0]

            return cls(
                projection_matrix=q_mat,
                global_mean=torch.zeros(in_channels, device=generator.device),
            )
        if universe.ndim != 2:
            raise ValueError("Universe must be a 2D tensor.")

        # 2. DATA-DRIVEN PROJECTIONS (PCA or ICA)
        global_mean = universe.mean(dim=0, keepdim=True)
        centered_universe = universe - global_mean

        with torch.random.fork_rng(
            devices=[centered_universe.device]
            if centered_universe.device.type != "cpu"
            else [],
        ):
            torch.manual_seed(seed)
            if ica_iterations < 1:
                # Just do PCA, Whiten the variance, and return
                s, v = torch.pca_lowrank(centered_universe, q=out_channels)[1:]
                # Whitening matrix for PCA (diag 1/S)
                v_white = v / (s.unsqueeze(0) + 1e-5)
                return cls(
                    projection_matrix=v_white,
                    global_mean=global_mean.squeeze(0),
                )

            # Run the full PCA -> Whiten -> ICA pipeline
            proj_matrix = cls.ica(
                centered_universe,
                iterations=ica_iterations,
                pca_rank=out_channels,
            )
        return cls(
            projection_matrix=proj_matrix,
            global_mean=global_mean.squeeze(0),
        )

    @classmethod
    # FastICA with optional PCA input reduction.
    def ica(
        cls,
        x: torch.Tensor,  # [N, C] or [B, N, C]
        *,
        iterations: int = 10,
        pca_rank: int = 192,
        # Skip running iterations if the change is less than this.
        tolerance: float = 1e-05,
        # When using tolerance, the result has to be within the limit
        # this many times consecutively.
        tolerance_checks: int = 2,
        # 0 to disable, otherwise forces the min/max item in each row
        # to have this sign by flipping the sign of elements in that row.
        row_maxval_polarity: int = 1,
    ) -> torch.Tensor:
        tolerance_checks = max(1, tolerance_checks)
        tolerance_counter = 0
        row_maxval_polarity = min(1, max(-1, int(row_maxval_polarity)))
        orig_ndim = x.ndim
        if x.ndim == 2:
            x = x.unsqueeze(0)

        shape = x.shape

        # 1. PCA Reduction Step
        if 0 < pca_rank < shape[-1]:
            # pca_lowrank expects 2D data, so we reshape if batched
            x_flat = x.reshape(-1, shape[-1])
            v_pca = torch.pca_lowrank(x_flat, pca_rank)[-1]  # [C_in, pca_rank]
            x = x @ v_pca  # [B, N, pca_rank]
        else:
            v_pca = None

        shape = x.shape

        # 2. Whitening Step
        x_white, w_white = cls.whiten(x)[:2]  # w_white: [B, C_pca, C_pca]

        # 3. ICA Initialization
        w_ica = (
            torch.eye(shape[-1], device=x_white.device, dtype=x_white.dtype)
            .expand(shape[0], -1, -1)
            .clone()
        )

        # 4. FastICA Iterations
        for _ in range(iterations):
            w_ica_prev = w_ica.clone() if tolerance > 0 else None
            projected = x_white @ w_ica.mT
            g_x = projected.pow(3)
            g_prime_x = projected.pow_(2).mul_(3)

            w_new = (g_x.mT @ x_white).div_(x_white.shape[1]) - (
                g_prime_x.mean(dim=1, keepdim=True).mT * w_ica
            )

            # SVD for symmetric orthogonalization
            u, _s, v = torch.linalg.svd(w_new)
            w_ica = u @ v
            if w_ica_prev is None:
                continue
            max_deviation = (
                (1.0 - torch.diagonal(w_ica @ w_ica_prev.mT, dim1=-2, dim2=-1).abs())
                .clamp_min_(0)
                .max()
                .detach()
                .cpu()
                .item()
            )
            tolerance_counter = (
                tolerance_counter + 1 if max_deviation < tolerance else 0
            )
            if tolerance_counter >= tolerance_checks:
                break
        if row_maxval_polarity != 0:
            max_signs = w_ica.gather(
                -1,
                w_ica.abs().argmax(
                    dim=-1,
                    keepdim=True,
                ),
            ).sign()
            w_ica *= max_signs.neg_() if row_maxval_polarity < 0 else max_signs
        # 5. Assemble the final Projection Matrix
        # Match batch dimensions: v_pca is [C_in, C_pca], w_white is [B, C_pca, C_pca]
        if v_pca is not None:
            # Broadcast v_pca to match the batch size
            v_pca_b = v_pca.unsqueeze(0).expand(shape[0], -1, -1)
            result = v_pca_b @ w_white @ w_ica.mT
        else:
            result = w_white @ w_ica.mT

        return result.squeeze(0) if orig_ndim == 2 else result

    @staticmethod
    def whiten(
        x: torch.Tensor,
        *,
        dim: int = 1,
        eps: float = 1e-05,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=dim, keepdim=True)
        x_cent = x - mean
        cov = (x_cent.mT @ x_cent).div_(x_cent.shape[dim] - 1)
        u, s = torch.linalg.svd(cov)[:2]
        s += eps
        w = u @ torch.diag_embed(1.0 / s.sqrt_()) @ u.mT
        return x_cent @ w, w, mean
