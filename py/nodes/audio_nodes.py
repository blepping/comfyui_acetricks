import math

import torch

from .. import external
from ..utils import fixup_waveform


class AudioFromBatchNode:
    DESCRIPTION = "Can be used to extract batch items from AUDIO."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("AUDIO",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "audio": ("AUDIO",),
                "start": (
                    "INT",
                    {
                        "default": 0,
                        "tooltip": "Start index (zero-based). Negative indexes count from the end.",
                    },
                ),
                "length": ("INT", {"default": 1, "min": 0}),
            },
        }

    @classmethod
    def go(cls, *, audio: dict, start: int, length: int) -> tuple:
        waveform = audio["waveform"]
        if not waveform.ndim == 3:
            raise ValueError("Expected 3D waveform")
        batch = waveform.shape[0]
        if start < 0:
            start = batch + start
        if start < 0:
            raise ValueError("Start index is out of range")
        new_waveform = waveform[start : start + length].clone()
        return (audio | {"waveform": new_waveform},)


class WaveformNode:
    DESCRIPTION = "Creates a waveform image from audio."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict]:
        return {
            "required": {
                "audio": ("AUDIO",),
                "width": ("INT", {"default": 800, "min": 1}),
                "height": ("INT", {"default": 200, "min": 1}),
                "background_rgb": ("STRING", {"default": "000020"}),
                "left_rgb": (
                    "STRING",
                    {
                        "default": "e0a080",
                        "tooltip": "Used for both channels in the case of mono audio.",
                    },
                ),
                "right_rgb": ("STRING", {"default": "80e0a0"}),
                "mode": (
                    ("normal", "rescaled", "log", "log_rescaled"),
                    {"default": "rescaled"},
                ),
                "log_factor": ("FLOAT", {"default": 10.0, "min": 0.0, "step": 0.01}),
                "oversampling": ("INT", {"default": 4, "min": 1}),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        audio: dict,
        width: int,
        height: int,
        background_rgb: str,
        left_rgb: str,
        right_rgb: str,
        mode: str,
        log_factor: float,
        oversampling: int,
    ) -> tuple:
        height = max(1, height // 2)
        n = int(background_rgb, 16)
        brgb = tuple(((n >> (i * 8)) & 255) / 255 for i in range(2, -1, -1))
        n = int(left_rgb, 16)
        lrgb = tuple(((n >> (i * 8)) & 255) / 255 for i in range(2, -1, -1))
        n = int(right_rgb, 16)
        rrgb = tuple(((n >> (i * 8)) & 255) / 255 for i in range(2, -1, -1))
        waveform = audio["waveform"]
        if waveform.ndim == 1:
            waveform = waveform[None, None]
        elif waveform.ndim == 2:
            waveform = waveform[None]
        elif waveform.ndim != 3:
            errstr = f"Unexpected number of dimensions in waveform, expected 1-3, got {waveform.ndim}"
            raise ValueError(errstr)
        waveform = waveform[:, :2, ...].to(dtype=torch.float64)
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(2),
            size=(height, min(waveform.shape[-1], width * oversampling)),
            mode="nearest-exact",
        ).movedim(1, -1)
        waveform = waveform.abs().clamp(0, 1)
        if mode in {"log", "log_rescaled"}:
            waveform = (
                (waveform * log_factor).log1p() / math.log1p(log_factor)
            ).clamp_(0, 1)
        if mode in {"rescaled", "log_rescaled"}:
            waveform = (waveform / waveform.max()).nan_to_num_().clamp_(0, 1)
        channels = waveform.shape[-1]
        hmask = torch.linspace(
            1.0, 0.0, height + 1, dtype=waveform.dtype, device=waveform.device
        )[1:].view(1, height, 1, 1)
        left_channel = waveform[..., 0:1]
        limg = torch.cat(
            tuple(
                torch.where(left_channel > hmask, fpixval, bpixval)
                for fpixval, bpixval in zip(lrgb, brgb, strict=False)
            ),
            dim=-1,
        )
        if channels >= 2:
            right_channel = waveform[..., 1:]
            rimg = torch.cat(
                tuple(
                    torch.where(right_channel > hmask, fpixval, bpixval)
                    for fpixval, bpixval in zip(rrgb, brgb, strict=False)
                ),
                dim=-1,
            )
        else:
            rimg = limg
        result = torch.cat((limg, rimg.flip(dims=(1,))), dim=1)
        if result.shape[2] != width:
            result = torch.nn.functional.interpolate(
                result.movedim(-1, 1), size=(height * 2, width), mode="bicubic"
            ).movedim(1, -1)
        return (result.to(device="cpu", dtype=torch.float32),)


class AudioLevelsNode:
    DESCRIPTION = "The values in the waveform range for -1 to 1. This node allows you to scale audio to a percentage of that range."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("AUDIO",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "audio": ("AUDIO",),
                "scale": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "tooltip": "Percentage where 1.0 indicates 100% of the maximum allowed value in an audio tensor. You can use 1.0 to make it as loud as possible without actually clipping.",
                    },
                ),
                "per_channel": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When enabled, the levels for each channel will be scaled independently. For multi-channel audio (like stereo) enabling this will not preserve the relative levels between the channels so probably should be left disabled most of the time.",
                    },
                ),
            },
        }

    @classmethod
    def go(cls, *, audio: dict, scale: float, per_channel: bool) -> tuple[dict]:
        waveform = audio["waveform"].to(device="cpu", copy=True)
        if waveform.ndim == 1:
            waveform = waveform[None, None, ...]
        elif waveform.ndim == 2:
            waveform = waveform[None, ...]
        elif waveform.ndim != 3:
            raise ValueError("Unexpected number of dimensions in waveform!")
        max_val = (
            waveform.abs().flatten(start_dim=2 if per_channel else 1).max(dim=-1).values
        )
        max_val = max_val[..., None] if per_channel else max_val[..., None, None]
        # Max could be 0, multiplying by 0 is fine in that case.
        waveform *= (scale / max_val).nan_to_num()
        return (audio | {"waveform": waveform.clamp(-1.0, 1.0)},)


class AudioAsLatentNode:
    DESCRIPTION = "This node allows you to rearrange AUDIO to look like a LATENT. Can be useful if you want to apply some latent operations to AUDIO. Can be reversed with the ACETricks LatentAsAudio node."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("LATENT",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "audio": ("AUDIO",),
                "use_width": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, you'll get a 4 channel with height 1 and the audio audio data in the width dimension, otherwise the opposite.",
                    },
                ),
            },
        }

    @classmethod
    def go(cls, *, audio: dict, use_width: bool) -> tuple:
        waveform = audio["waveform"].to(device="cpu", copy=True)
        if waveform.ndim == 1:
            waveform = waveform[None, None, ...]
        elif waveform.ndim == 2:
            waveform = waveform[None, ...]
        elif waveform.ndim != 3:
            raise ValueError("Unexpected number of dimensions in waveform!")
        waveform = waveform.unsqueeze(2) if use_width else waveform[..., None]
        return ({"samples": waveform},)


class LatentAsAudioNode:
    DESCRIPTION = "This node lets you rearrange a LATENT to look like AUDIO. Mainly useful for getting back after using the ACETricks AudioAsLatent node and performing some operations. If you connect the optional audio input it will use whatever non-waveform parameters exist in it (can be stuff like the sample rate), otherwise it will just add sample_rate: 41000 and the waveform."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("AUDIO",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "latent": ("LATENT",),
                "values_mode": (
                    ("rescale", "clamp"),
                    {"default": "rescale"},
                ),
                "use_width": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, takes the audio data from the first item in the width dimension, otherwise height.",
                    },
                ),
            },
            "optional": {
                "audio_opt": (
                    "AUDIO",
                    {
                        "tooltip": "Optional audio to use as a reference for sample rate and possibly other values."
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        latent: dict,
        values_mode: str,
        use_width: bool,
        audio_opt: dict | None = None,
    ) -> tuple:
        samples = latent["samples"]
        if samples.ndim != 4:
            raise ValueError("Expected a 4D latent but didn't get one")
        samples = (samples[..., 0, :] if use_width else samples[..., 0]).to(
            device="cpu", copy=True
        )
        if audio_opt is None:
            audio_opt = {"sample_rate": 44100}
        result = audio_opt | {"waveform": samples}
        if values_mode == "clamp":
            result["waveform"] = samples.clamp(-1.0, 1.0)
        elif torch.any(samples.abs() > 1.0):
            return AudioLevelsNode.go(audio=result, per_channel=False, scale=1.0)
        return (result,)


class MonoToStereoNode:
    DESCRIPTION = "Can convert mono AUDIO to stereo. It will leave AUDIO that's already stereo alone. Note: Always adds a batch dimension if it doesn't exist and moves to the CPU device."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("AUDIO",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {"audio": ("AUDIO",)}}

    @classmethod
    def go(cls, *, audio: dict) -> tuple:
        waveform = audio["waveform"].to(device="cpu")
        if waveform.ndim == 2:
            waveform = waveform[None]
        elif waveform.ndim == 1:
            waveform = waveform[None, None]
        channels = waveform.shape[1]
        audio = audio.copy()
        if channels == 1:
            waveform = waveform.repeat(1, 2, 1)
        audio["waveform"] = waveform
        return (audio,)


class SetAudioDtypeNode:
    DESCRIPTION = "Advanced node that allows the datatype of the audio waveform. The 16 and 8 bit types are not recommended."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("AUDIO",)

    _ALLOWED_DTYPES = (
        "float64",
        "float32",
        "float16",
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "audio": ("AUDIO",),
                "dtype": (
                    cls._ALLOWED_DTYPES,
                    {"default": "float64", "tooltip": "TBD"},
                ),
            },
        }

    @classmethod
    def go(cls, *, audio: dict, dtype: str) -> tuple[dict]:
        if dtype not in cls._ALLOWED_DTYPES:
            raise ValueError("Bad dtype")
        waveform = audio["waveform"]
        dt = getattr(torch, dtype)
        if waveform.dtype == dt:
            return (audio,)
        return (audio | {"waveform": waveform.to(dtype=dt)},)


class AudioBlendNode:
    DESCRIPTION = "Blends two AUDIO inputs together. If you have ComfyUI-bleh installed you will have access to many additional blend modes."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("AUDIO",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        external.ensure_blend_modes()
        if not external.BLEND_MODES:
            raise RuntimeError  # Impossible
        return {
            "required": {
                "audio_a": ("AUDIO",),
                "audio_b": ("AUDIO",),
                "audio_b_strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": -1000.0,
                        "max": 1000.0,
                    },
                ),
                "blend_mode": (
                    tuple(external.BLEND_MODES.keys()),
                    {
                        "default": "lerp",
                    },
                ),
                "length_mismatch_mode": (
                    ("shrink", "blend"),
                    {
                        "default": "shrink",
                        "tooltip": "Shrink mode will return audio matching whatever the shortest input was. Blend will blend up to the shortest input's size and use unblended longer input to fill the rest. Note that this adjustment occurs before blending.",
                    },
                ),
                "normalization_mode": (
                    ("clamp", "levels", "levels_per_channel", "none"),
                    {
                        "default": "levels",
                        "tooltip": "Clamp will just clip the result to ensure it is within the permitted range. Levels will rebalance it so the maximum value is the maximum value for the permitted range. Levels per channel is the same, except the maximum value is determined separately per channel. Setting this to none is not recommended unless you are planning to do your own normalization as it may leave invalid values in the audio latent.",
                    },
                ),
                "result_template": (
                    ("a", "b"),
                    {
                        "default": "a",
                        "tooltip": "AUDIOs contain metadata like sampling rate. The result will be based on the metadata from the audio input you select here, with the blended result as the waveform in it.",
                    },
                ),
            }
        }

    @classmethod
    def go(
        cls,
        *,
        audio_a: dict,
        audio_b: dict,
        audio_b_strength: float,
        blend_mode: str,
        length_mismatch_mode: str,
        normalization_mode: str,
        result_template: str,
    ) -> tuple:
        wa = fixup_waveform(audio_a["waveform"])
        wb = fixup_waveform(audio_b["waveform"])
        if wa.dtype != wb.dtype:
            wa = wa.to(dtype=torch.float32)
            wb = wb.to(dtype=torch.float32)
        if wa.shape[:-1] != wb.shape[:-1]:
            errstr = f"Unexpected batch or channels shape mismatch in audio. audio_a has shape {wa.shape}, audio_b has shape {wb.shape}"
            raise ValueError(errstr)
        assert BLEND_MODES is not None  # Make static analysis happy.
        blend_function = BLEND_MODES[blend_mode]
        walen, wblen = wa.shape[-1], wb.shape[-1]
        if walen != wblen:
            if length_mismatch_mode == "shrink":
                minlen = min(walen, wblen)
                wa = wa[..., :minlen]
                wb = wb[..., :minlen]
            elif walen > wblen:
                wb_temp = wa.clone()
                wb_temp[..., :wblen] = wb
                wb = wb_temp
            else:
                wa_temp = wb.clone()
                wa_temp[..., :walen] = wa
                wa = wa_temp
            walen = wblen = wa.shape[-1]
        result = blend_function(wa, wb, audio_b_strength)
        result_audio = audio_a.copy() if result_template == "a" else audio_b.copy()
        if normalization_mode == "clamp":
            result = result.clamp_(min=-1.0, max=1.0)
        elif normalization_mode in {"levels", "levels_per_channel"}:
            result = AudioLevelsNode.go(
                audio={"waveform": result},
                scale=1.0,
                per_channel=normalization_mode == "levels_per_channel",
            )[0]["waveform"]
        result_audio["waveform"] = result
        return (result_audio,)
