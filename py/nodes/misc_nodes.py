import torch

from ..ace_utils import LATENT_TIME_MULTIPLIER, LATENT_TIME_MULTIPLIER_15


class TimeOffsetNode:
    DESCRIPTION = "Can be used to calculate an offset into an ACE-Steps 1.0 latent given a time in seconds."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("INT", "FLOAT")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "seconds": ("FLOAT", {"default": 0.0, "min": 0.0}),
            },
            "optional": {
                "model_type": (("ACE 1.0", "ACE 1.5"), {"default": "ACE 1.0"}),
            },
        }

    @classmethod
    def go(cls, *, seconds: float, model_type: str | None = None) -> tuple[int, float]:
        time_multiplier = (
            LATENT_TIME_MULTIPLIER_15
            if model_type != "ACE 1.0"
            else LATENT_TIME_MULTIPLIER
        )
        offset = seconds * time_multiplier
        return (int(offset), offset)


class MaskNode:
    DESCRIPTION = "Can be used to create a mask based on time and frequency bands"
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("MASK",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "seconds": ("FLOAT", {"default": 120.0, "min": 1.0}),
                "start_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -99999.0,
                        "tooltip": "Negative values count from the end.",
                    },
                ),
                "end_time": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -99999.0,
                        "tooltip": "Negative values count from the end.",
                    },
                ),
                "start_freq": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 15,
                        "tooltip": "Frequency bands, 0 is the lowest frequency. Inclusive.",
                    },
                ),
                "end_freq": (
                    "INT",
                    {
                        "default": 15,
                        "min": 0,
                        "max": 15,
                        "tooltip": "Frequency bands, 0 is the lowest frequency. Inclusive.",
                    },
                ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "base_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        seconds: float,
        start_time: float,
        end_time: float,
        start_freq: int,
        end_freq: int,
        strength: float,
        base_value: float,
    ) -> tuple[torch.Tensor]:
        time_len = int(seconds * LATENT_TIME_MULTIPLIER)
        offs_start = int(start_time * LATENT_TIME_MULTIPLIER)
        offs_end = int(end_time * LATENT_TIME_MULTIPLIER)
        if offs_start < 0:
            offs_start = max(0, time_len + offs_start)
        if offs_end < 0:
            offs_end = max(0, time_len + offs_end)
        offs_start = min(time_len - 1, offs_start)
        offs_end = min(time_len - 1, offs_end)
        mask = torch.full(
            (1, 16, time_len),
            value=base_value,
            dtype=torch.float32,
            device="cpu",
        )
        mask[:, start_freq : end_freq + 1, offs_start : offs_end + 1] = strength
        return (mask,)
