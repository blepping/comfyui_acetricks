from collections.abc import Sequence
from typing import NamedTuple

import torch
from comfy import model_management

try:
    from comfy.ldm.ace.ace_step15 import get_silence_latent as get_ace15_silence_latent

    HAVE_ACE15_SILENCE = True
except Exception:  # noqa: BLE001
    HAVE_ACE15_SILENCE = False


LATENT_TIME_MULTIPLIER = 44100 / 512 / 8
LATENT_TIME_MULTIPLIER_15 = 25.0  # 48000 / 1920

ACE10_SILENCE = torch.tensor(
    (
        (
            -0.6462,
            -1.2132,
            -1.3026,
            -1.2432,
            -1.2455,
            -1.2162,
            -1.2184,
            -1.2114,
            -1.2153,
            -1.2144,
            -1.2130,
            -1.2115,
            -1.2063,
            -1.1918,
            -1.1154,
            -0.7924,
        ),
        (
            0.0473,
            -0.3690,
            -0.6507,
            -0.5677,
            -0.6139,
            -0.5863,
            -0.5783,
            -0.5746,
            -0.5748,
            -0.5763,
            -0.5774,
            -0.5760,
            -0.5714,
            -0.5560,
            -0.5393,
            -0.3263,
        ),
        (
            -1.3019,
            -1.9225,
            -2.0812,
            -2.1188,
            -2.1298,
            -2.1227,
            -2.1080,
            -2.1133,
            -2.1096,
            -2.1077,
            -2.1118,
            -2.1141,
            -2.1168,
            -2.1134,
            -2.0720,
            -1.7442,
        ),
        (
            -4.4184,
            -5.5253,
            -5.7387,
            -5.7961,
            -5.7819,
            -5.7850,
            -5.7980,
            -5.8083,
            -5.8197,
            -5.8202,
            -5.8231,
            -5.8305,
            -5.8313,
            -5.8153,
            -5.6875,
            -4.7317,
        ),
        (
            1.5986,
            2.0669,
            2.0660,
            2.0476,
            2.0330,
            2.0271,
            2.0252,
            2.0268,
            2.0289,
            2.0260,
            2.0261,
            2.0252,
            2.0240,
            2.0220,
            1.9828,
            1.6429,
        ),
        (
            -0.4177,
            -0.9632,
            -1.0095,
            -1.0597,
            -1.0462,
            -1.0640,
            -1.0607,
            -1.0604,
            -1.0641,
            -1.0636,
            -1.0631,
            -1.0594,
            -1.0555,
            -1.0466,
            -1.0139,
            -0.8284,
        ),
        (
            -0.7686,
            -1.0507,
            -1.3932,
            -1.4880,
            -1.5199,
            -1.5377,
            -1.5333,
            -1.5320,
            -1.5307,
            -1.5319,
            -1.5360,
            -1.5383,
            -1.5398,
            -1.5381,
            -1.4961,
            -1.1732,
        ),
        (
            0.0199,
            -0.0880,
            -0.4010,
            -0.3936,
            -0.4219,
            -0.4026,
            -0.3907,
            -0.3940,
            -0.3961,
            -0.3947,
            -0.3941,
            -0.3929,
            -0.3889,
            -0.3741,
            -0.3432,
            -0.169,
        ),
    ),
    dtype=torch.float32,
    device="cpu",
)[None, ..., None]


def parse_audio_codes(audio_codes: str | Sequence[int] | None) -> tuple[int, ...]:
    if audio_codes is None:
        return ()
    if isinstance(audio_codes, str):
        audio_codes = audio_codes.strip()
        if audio_codes.startswith("<|audio_code_"):
            cs = tuple(
                ac.rsplit("_", 1)[-1].strip()
                for ac in audio_codes.split("|")
                if ac.startswith("audio_code_")
            )
        else:
            cs = tuple(ac.strip() for ac in audio_codes.split(","))
        if not all(ac.isdigit() for ac in cs if ac):
            raise ValueError(
                "When specified as a string, codes must be comma separated integer values or a sequence of <|audio_code_123|> tokens",
            )
        audio_codes = tuple(int(ac) for ac in cs if ac)
    else:
        audio_codes = tuple(audio_codes)
    if not all(isinstance(ac, int) and (0 <= ac < 64000) for ac in audio_codes):
        raise TypeError("Audio codes must parse to integer values >= 0 and < 64000.")
    return audio_codes


class DeconstructedHints(NamedTuple):
    indices: torch.Tensor
    hints_2048d: torch.Tensor
    hints_6d: torch.Tensor

    @staticmethod
    def get_codebook_parts(
        model: object,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = torch.float32,
        # Returns projection, implicit codebook, scale
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not (hasattr(model, "tokenizer") and hasattr(model.tokenizer, "quantizer")):
            raise ValueError(
                "Can't find tokenizer or quantizer in model. Supplied model input must be ACE-Step 1.5.",
            )
        quantizer: torch.nn.Module = model.tokenizer.quantizer

        if len(quantizer.layers) != 1:
            raise ValueError("Unexpected number of quantizer layers (>1)")
        fsq_layer = quantizer.layers[0]
        scale = quantizer.scales[0].to(dtype=dtype, device=device)

        # 1. Grab the pure [-1, 1] continuous coordinates for all 64,000 possible codes
        implicit_cb = model_management.cast_to(
            fsq_layer.implicit_codebook,
            device=device,
            dtype=dtype,
        )

        # 2. Project all 64,000 codes into the 2048D semantic space
        valid_6d = implicit_cb * scale
        # Shape: [64000, 2048] (~500MB VRAM at float32)
        return quantizer.project_out(valid_6d), implicit_cb, scale

    @classmethod
    def deconstruct(
        cls,
        model: object,
        hints_5hz: torch.Tensor,
        *,
        dtype: torch.dtype | None = torch.float32,
        chunk_size: int = 512,
    ) -> "DeconstructedHints":
        quantizer: torch.nn.Module = model.tokenizer.quantizer
        device = hints_5hz.device
        orig_dtype = hints_5hz.dtype

        if dtype is not None and dtype != orig_dtype:
            hints_5hz = hints_5hz.to(dtype=dtype)

        valid_2048d, implicit_cb, scale = cls.get_codebook_parts(
            model,
            device=device,
            dtype=dtype,
        )

        batch, sequence, dims = hints_5hz.shape
        hints_flat = hints_5hz.reshape(batch * sequence, dims)

        new_indices_flat = torch.empty(
            batch * sequence,
            dtype=torch.int32,
            device=device,
        )

        # 3. Find the closest valid codebook item for every frame
        # We process in chunks to prevent VRAM spikes (512 * 64k is tiny).
        # Searching 64K codes might sound scary but 10min at 5hz is only
        # a sequence size of 3,000.
        for i in range(0, batch * sequence, chunk_size):
            chunk = hints_flat[i : i + chunk_size]

            # Compute L2 distance between our hints and the 64k valid 2048D vectors
            dists = torch.cdist(chunk, valid_2048d)  # Shape: [chunk_size, 64000]

            # Because our codes are exactly indexed 0-63999, the argmin gives us the integer code.
            new_indices_flat[i : i + chunk_size] = torch.argmin(dists, dim=-1).to(
                torch.int32,
            )

        new_indices = new_indices_flat.view(batch, sequence)

        # 4. Extract the snapped 6D continuous vectors
        quantized_6d = torch.nn.functional.embedding(new_indices, implicit_cb)
        quantized_6d *= scale

        # 5. Project back to 2048D
        new_hints_5hz = quantizer.project_out(quantized_6d)

        new_hints_5hz, quantized_6d = (
            t.to(dtype=orig_dtype) for t in (new_hints_5hz, quantized_6d)
        )
        return cls(new_indices, new_hints_5hz, quantized_6d)


__all__ = (
    "ACE10_SILENCE",
    "LATENT_TIME_MULTIPLIER",
    "LATENT_TIME_MULTIPLIER_15",
    "get_ace15_silence_latent",
)
