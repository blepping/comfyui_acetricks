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
    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = (
        "parsed_codes",
        "input_length",
        "output_length",
        "start_index",
        "end_index",
        "time",
        "start_time",
        "end_time",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "framerate_hz": (
                    "STRING",
                    {
                        "default": "ace15",
                        "placeholder": "Enter hz or a preset: ace15, yue2",
                        "tooltip": "Used when setting a time-based time_mode. ACE-Step 1.5 uses 5hz (5.0), YuE2 uses 25hz (25.0). You may also enter a preset name. Available presets: ace15, yue2",
                    },
                ),
                "audio_codes": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Input audio codes. This can handle a comma-seperated list of integers or audio codes (with no delimiter) in the format <|audio_code_123|>. The node will look for the audio code prefix first (and other preceding text will be ignored). If it can't find the prefix, it will try to parse as a comma-separated list of integers. In other words, leading non-code text is fine if the input is audio codes, otherwise it must be a list of integers separated by commas.",
                    },
                ),
                "time_mode": (
                    ("seconds", "index", "percent"),
                    {
                        "default": "seconds",
                        "tooltip": "seconds - interpret the start/end parameters based on framerate_hz.\nindex - interpret the start/end parameters as absolute (integer) indexes.\npercent - interpret start/end as a percentage of the total codes length.",
                    },
                ),
                "start": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "max": 999999.0,
                        "min": -999999.0,
                        "tooltip": "Negative values count from the end. This is clamped to the valid range.",
                    },
                ),
                "end": (
                    "FLOAT",
                    {
                        "default": 999999.0,
                        "max": 999999.0,
                        "min": -999999.0,
                        "tooltip": "Negative values count from the end. This is clamped to the valid range.",
                    },
                ),
                "output_mode": (
                    ("tokens", "integers_commasep", "integers_spacesep", "json"),
                    {
                        "default": "tokens",
                        "tooltip": "tokens - Outputs a sequence of <|audio_code_123|> tokens.\nintegers_commasep - outputs a sequence of comma-separated integers.\nintegers_commasep - outputs a sequence of space-delimited integers.\njson - same as integers_commasep except it will add square brackets around the result to turn it into a valid JSON (or Python) list.",
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
        output_mode: str,
    ) -> tuple[str, int, int, int, int, float, float, float]:
        framerate_hz = framerate_hz.strip().lower()
        time_mode = time_mode.strip().lower()
        output_mode = output_mode.strip().lower()
        if time_mode not in {"seconds", "index", "percent"}:
            raise ValueError("Bad time_mode")
        if output_mode not in {
            "tokens",
            "integers_commasep",
            "integers_spacesep",
            "json",
        }:
            raise ValueError("Bad output mode")
        if time_mode == "seconds":
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
        else:
            fr = 0.0
        codes = parse_audio_codes(audio_codes, codebook_size=0)
        n_codes = len(codes)
        if time_mode == "seconds":
            start, end = start * fr, end * fr
        elif time_mode == "percent":
            start, end = n_codes * start, n_codes * end
        start_idx = min(
            n_codes - 1,
            max(0, int(n_codes + start if start < 0 else start)),
        )
        end_idx = min(
            n_codes,
            max(0, int(n_codes + end if end < 0 else end)),
        )
        if time_mode == "seconds":
            start, end = start_idx / fr, end_idx / fr
        elif time_mode == "percent":
            start, end = (
                (start_idx / n_codes, end_idx / n_codes) if n_codes > 0 else (0.0, 0.0)
            )
        else:
            start, end = float(start_idx), float(end_idx)
        time_range = end - start if start < end else 0.0
        codes_slice = codes[start_idx:end_idx]
        output_length = len(codes_slice)
        codes_gen = (
            (f"<|audio_code_{c}|>" for c in codes_slice)
            if output_mode == "tokens"
            else (str(c) for c in codes_slice)
        )
        delim = (
            ""
            if output_mode == "tokens"
            else (" " if output_mode == "integers_spacesep" else ", ")
        )
        result = delim.join(codes_gen)
        if output_mode == "json":
            result = f"[{result}]"
        return (
            result,
            n_codes,
            output_length,
            start_idx,
            end_idx,
            time_range,
            start,
            end,
        )
