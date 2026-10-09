import contextlib

import torch

from ..ace_utils import LATENT_TIME_MULTIPLIER, LATENT_TIME_MULTIPLIER_15
from ..utils import parse_audio_codes


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


class CutAudioCodesNode:
    DESCRIPTION = (
        "Can be used to select a range of audio codes based on time or indexes."
    )
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "framerate_hz": (
                    "STRING",
                    {
                        "default": "ace15",
                        "tooltip": "Used when setting a time-based time_mode. ACE-Step 1.5 uses 5hz (5.0), YuE2 uses 25hz (25.0). You may also enter a preset name. Available presets: ace15, yue2",
                    },
                ),
                "audio_codes": ("STRING",),
                "time_mode": (
                    ("seconds", "index"),
                    {
                        "default": "seconds",
                    },
                ),
                "start": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "max": 99999.0,
                        "min": -99999.0,
                    },
                ),
                "end": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "max": 99999.0,
                        "min": -99999.0,
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        framerate_hz: str,
        audio_codes: str,
        time_mode: str,
        start: float,
        end: float,
    ) -> tuple[str]:
        framerate_hz = framerate_hz.strip().lower()
        time_mode = time_mode.strip().lower()
        if time_mode not in {"seconds", "index"}:
            raise ValueError("Bad time_mode")
        presets = {
            "ace15": 5.0,
            "yue2": 25.0,
        }
        fr = presets.get(framerate_hz)
        if fr is None and framerate_hz:
            with contextlib.suppress(ValueError):
                fr = float(framerate_hz)
        if fr is None or fr <= 0:
            raise ValueError(
                "Bad format for framerate_hz. Must either be a preset or positive non-zero floating point value",
            )
        codes = parse_audio_codes(audio_codes, n_codes=0)
        n_codes = len(codes)
        if time_mode == "seconds":
            start, end = (
                round(tval * fr) + n_codes * int(tval < 0) for tval in (start, end)
            )
        else:
            # index mode handling
            start, end = (
                int(tval if tval >= 0 else n_codes + tval) for tval in (start, end)
            )
        print(f"CODES: start={start}, end={end}, n_codes={n_codes}")
        if not (start < end and all(0 <= tval < n_codes for tval in (start, end))):
            raise ValueError("Time out of range or start is greater than end")
        codes = "".join(f"<|audio_code_{c}|>" for c in codes[start:end])
        return (codes,)
