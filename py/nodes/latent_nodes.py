import math

import torch

from ..ace_utils import (
    ACE10_SILENCE,
    HAVE_ACE15_SILENCE,
    LATENT_TIME_MULTIPLIER,
    LATENT_TIME_MULTIPLIER_15,
    get_ace15_silence_latent,
)
from ..utils import normalize_to_scale


class SilentLatentNode:
    DESCRIPTION = "Creates a latent full of (roughly) silence. This node can work for ACE-Steps 1.5 if you connect a reference latent."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("LATENT",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "seconds": (
                    "FLOAT",
                    {
                        "default": 120.0,
                        "min": 1.0,
                        "max": 1000.0,
                        "step": 0.1,
                        "tooltip": "Number of seconds to generate. Ignored if optional latent input is connected.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "Batch size to generate. Ignored if optional latent input is connected.",
                    },
                ),
            },
            "optional": {
                "ref_latent_opt": (
                    "LATENT",
                    {
                        "tooltip": "When connected the other parameters are ignored and the latent output will match the length/batch size of the reference. This needs to be connected to get a ACE-Steps 1.5 silent latent."
                    },
                ),
            },
        }

    @classmethod
    def go_ace15(cls, ref_shape: torch.Size) -> tuple[dict]:
        if not HAVE_ACE15_SILENCE:
            raise RuntimeError("ACE 1.5 silence unavailable. ComfyUI version too old?")
        ndim = len(ref_shape)
        if ndim == 4 and ref_shape[-2] != 1:
            raise ValueError(
                "Can't handle 4D ACE 1.5 latent with non-empty dimension -2"
            )
        latent = torch.zeros(
            ref_shape[0], 64, ref_shape[-1], device="cpu", dtype=torch.float32
        )
        latent += get_ace15_silence_latent(ref_shape[-1], device="cpu").to(latent)
        if ndim == 4:
            latent = latent.unsqueeze(-2)
        return ({"samples": latent, "type": "audio"},)

    @classmethod
    def go(cls, *, seconds: float, batch_size: int, ref_latent_opt=None) -> tuple[dict]:
        if ref_latent_opt is not None:
            ref_shape = ref_latent_opt["samples"].shape
            if len(ref_shape) in {3, 4} and ref_shape[1] == 64:
                return cls.go_ace15(ref_shape=ref_shape)
            latent = torch.zeros(ref_shape, device="cpu", dtype=torch.float32)
        else:
            length = int(seconds * LATENT_TIME_MULTIPLIER)
            latent = torch.zeros(
                batch_size, 8, 16, length, device="cpu", dtype=torch.float32
            )
        latent += ACE10_SILENCE
        return ({"samples": latent, "type": "audio"},)


class VisualizeLatentNode:
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "latent": ("LATENT",),
                "scale_secs": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "tooltip": "Horizontal scale. Number of pixels that corresponds to one second of audio. You can use 0 for no scaling which is roughly 11 pixels per second.",
                    },
                ),
                "scale_vertical": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1024,
                        "tooltip": "Pixel expansion factor for channels (or frequency bands if you have swap_channels_freqs mode enabled).",
                    },
                ),
                "swap_channels_freqs": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Swaps the order of channels and frequency in the vertical dimension. When enabled, scale_vertical applies to frequency bands.",
                    },
                ),
                "normalize_dims": (
                    "STRING",
                    {
                        "default": "-1",
                        "tooltip": "Dimensions the latent scale is normalized using. Must be a comma-separated list. The default setting normalizes the channels and frequency bands independently per batch, you can try -3, -2, -1 if you want to see the relative differences better.",
                    },
                ),
                "mode": (
                    (
                        "split",
                        "combined",
                        "brg",
                        "rgb",
                        "bgr",
                        "split_flip",
                        "combined_flip",
                        "brg_flip",
                        "rgb_flip",
                        "bgr_flip",
                    ),
                    {
                        "default": "split",
                        "tooltip": "Split shows a monochrome view of of each channel/freq, combined shows the average. Flip means invert the energy in the channel (i.e. white -> black). The other modes put the latent channels into the RGB channels of the preview image.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        latent,
        scale_secs,
        scale_vertical,
        swap_channels_freqs,
        normalize_dims,
        mode,
    ) -> tuple:
        normalize_dims = normalize_dims.strip()
        normalize_dims = (
            ()
            if not normalize_dims
            else tuple(int(dim) for dim in normalize_dims.split(","))
        )
        samples = latent["samples"].to(dtype=torch.float32, device="cpu")
        if samples.ndim == 3 and samples.shape[1] in {6, 64, 2048}:
            samples = samples.unsqueeze(-2)
            temporal_scale_factor = LATENT_TIME_MULTIPLIER_15
        elif samples.ndim == 4:
            temporal_scale_factor = LATENT_TIME_MULTIPLIER
        else:
            raise ValueError(
                "Expected an ACE-Steps 1.0 latent with 4 dimensions or an Ace-Step 1.5 latent with 3 dimensions and 64 channels.",
            )
        color_mode = mode not in {"split", "combined", "split_flip", "combined_flip"}
        batch, _channels, _freqs, temporal = samples.shape
        samples = normalize_to_scale(samples, 0.0, 1.0, dim=normalize_dims)
        if mode.endswith("_flip"):
            samples = 1.0 - samples
        if swap_channels_freqs:
            samples = samples.movedim(2, 1)
        if mode.startswith("combined"):
            samples = samples.mean(dim=1, keepdim=True)
        if scale_vertical != 1:
            samples = samples.repeat_interleave(scale_vertical, dim=2)
        if not color_mode:
            samples = samples.reshape(batch, -1, temporal)
        if scale_secs > 0:
            new_temporal = round((temporal / temporal_scale_factor) * scale_secs)
            samples = torch.nn.functional.interpolate(
                samples.unsqueeze(1) if not color_mode else samples,
                size=(samples.shape[-2], new_temporal),
                mode="nearest-exact",
            )
            if not color_mode:
                samples = samples.squeeze(1)
        if not color_mode:
            return (samples[..., None].expand(*samples.shape, 3),)
        rgb_count = math.ceil(samples.shape[1] / 3)
        channels_pad = rgb_count * 3 - samples.shape[1]
        samples = torch.cat(
            (
                samples,
                samples.new_zeros(samples.shape[0], channels_pad, *samples.shape[-2:]),
            ),
            dim=1,
        )
        samples = torch.cat(samples.chunk(rgb_count, dim=1), dim=2).movedim(1, -1)
        if mode.startswith("bgr"):
            samples = samples.flip(-1)
        elif mode.startswith("brg"):
            samples = samples.roll(-1, -1)
        return (samples,)


class SqueezeUnsqueezeLatentDimensionNode:
    DESCRIPTION = "This node can be used to add or remove an empty dimension from latents. Useful with ACE 1.5 which uses 3D latents while many ComfyUI latent processing nodes expect 4D+ latents. You will also likely need to use the ModelPatchAce15Use4dLatent node."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("LATENT",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "latent": ("LATENT",),
                "dimension": (
                    "INT",
                    {
                        "default": 2,
                        "min": -9999,
                        "max": 9999,
                        "tooltip": "Negative dimensions count from the end.",
                    },
                ),
                "unsqueeze_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, unsqueezes (adds) an empty dimension at the specified position. When disabled will remove an empty dimension instead.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        latent: dict,
        dimension: int,
        unsqueeze_mode: bool,
    ) -> tuple[dict]:
        samples = latent["samples"]
        pos_dim = dimension if dimension >= 0 else samples.ndim + dimension
        max_dim = samples.ndim if unsqueeze_mode else samples.ndim - 1
        if pos_dim < 0 or pos_dim > max_dim:
            errstr = f"Specified dimension {dimension} out of range for latent with shape {samples.shape} ({samples.ndim} dimension(s))"
            raise ValueError(errstr)
        if unsqueeze_mode:
            samples = samples.unsqueeze(dimension)
        else:
            if samples.shape[dimension] != 1:
                errstr = f"Dimension {dimension} in latent with shape {samples.shape} is not empty, has size {samples.shape[dimension]}. This node can only squeeze empty dimensions."
                raise ValueError(errstr)
            samples = samples.squeeze(dimension)
        return (latent | {"samples": samples.clone()},)
