from typing import Self

import torch
from comfy import model_management, model_patcher, ops, storage
from comfy import utils as comfy_utils
from comfy.audio import resample as resample_audio
from comfy.audio_encoders import audio_encoders, mert2
from comfy.audio_encoders.sheetsage2 import sliding_window_plan
from comfy.ldm.modules.attention import optimized_attention_for_device
from torch import nn


class MERT2Encoder(mert2.MERT2):
    def forward(
        self,
        mel: torch.Tensor,
        *,
        n_layers: int | None = None,
    ) -> torch.Tensor:
        layers = self.layers
        n_layers = len(layers) if n_layers is None else min(len(layers), n_layers)
        x = self.subsampling_module(mel)
        positions = self.position_embeddings(x)
        attention = optimized_attention_for_device(x.device)
        for layer in layers[:n_layers]:
            x = layer(x, positions, attention)
        return x


class TokenizerHead(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int = 1024,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        vocab_size: int = 32768,
        max_len: int = 512,
        device=None,
        dtype=None,
        operations=ops.manual_cast,
    ):
        super().__init__()
        self.inp = operations.Linear(in_dim, d_model, device=device, dtype=dtype)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        # TODO: Split up to use operations/ComfyUI's attention handling.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = operations.LayerNorm(d_model, device=device, dtype=dtype)
        self.head = operations.Linear(
            d_model,
            vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.pos[:, : x.shape[1], :].to(x)
        return self.head(self.norm(self.enc(self.inp(x).add_(pos))))


class MERT2Tokenizer(nn.Module):
    def __init__(self, device=None, dtype=None, operations=ops.manual_cast):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.mert2_encoder = MERT2Encoder(
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.tokenizer_head = TokenizerHead(
            device=device,
            dtype=dtype,
            operations=operations,
        )


class MERT2TokenizerAudioEncoderModel(audio_encoders.AudioEncoderModel):
    def __init__(self, *, fast_disk: bool = False):
        self.load_device = model_management.text_encoder_device()
        offload_device = model_management.text_encoder_offload_device()
        self.dtype = torch.float32
        self.model_sample_rate = 24000.0
        self.model = MERT2Tokenizer(
            dtype=self.dtype,
            device=offload_device,
            operations=ops.manual_cast,
        )
        self.model.eval()
        self.patcher = model_patcher.CoreModelPatcher(
            self.model,
            load_device=self.load_device,
            offload_device=offload_device,
            fast_disk=fast_disk,
        )
        model_management.archive_model_dtypes(self.model)

    @classmethod
    def load_audioencoder(cls, *, mert2_path: str, tokenizer_head_path: str) -> Self:
        sd = {
            f"mert2_encoder.{k}": v
            for k, v in comfy_utils.load_torch_file(mert2_path).items()
        }
        sd |= {
            f"tokenizer_head.{k}": v
            for k, v in comfy_utils.load_torch_file(tokenizer_head_path).items()
        }
        fast_disk = storage.model_fast_disk((mert2_path, tokenizer_head_path))
        audio_encoder = cls(fast_disk=fast_disk)
        missing, unknown = audio_encoder.load_sd(sd)
        if missing:
            raise ValueError(f"Missing state dict keys: {missing}")  # noqa: EM102
        if unknown:
            raise ValueError(f"Unknown state dict keys: {unknown}")  # noqa: EM102
        return audio_encoder

    def get_mert2_feats(self, *, waveform: torch.Tensor) -> torch.Tensor:
        encoder = self.model.mert2_encoder
        sr, fps = 25.0, self.model_sample_rate
        duration = waveform.shape[-1] / sr
        if duration <= 300.0:
            return encoder(encoder.feature_extractor(waveform), n_layers=20)
        plan = sliding_window_plan(
            duration,
            window_seconds=300.0,
            overlap_seconds=200.0,
            lookahead_seconds=100.0,
        )
        chunks = []
        for w in plan:
            chunk = waveform[:, round(w["start"] * sr) : round(w["end"] * sr)]
            # [B, T_chunk, 1024]
            chunk_feat = encoder(encoder.feature_extractor(chunk), n_layers=20)

            rel_start_sec = w["accept_start"] - w["start"]
            rel_end_sec = w["accept_end"] - w["start"]
            f_start = round(rel_start_sec * fps)
            f_end = round(rel_end_sec * fps)

            chunks.append(chunk_feat[:, f_start:f_end, :])

        return torch.cat(chunks, dim=1)

    def tokenize_mert2_feats(
        self,
        *,
        feat: torch.Tensor,
        normalize: bool = True,
        in_place: bool = True,
        chunk_size: int = 512,
        eps: float | None = None,
    ) -> torch.Tensor:
        tokenizer_head = self.model.tokenizer_head
        if normalize:
            if eps is None:
                eps = torch.finfo(feat.dtype).eps * 1.25
            if not in_place:
                feat = feat.clone()
            std, mean = torch.std_mean(feat, dim=1, keepdim=True)
            feat = feat.sub_(mean).div_(std.clamp_min_(eps))

        frames = feat.shape[1]
        tokens_list = []

        for start_idx in range(0, frames, chunk_size):
            end_idx = min(start_idx + chunk_size, frames)
            feat_chunk = feat[:, start_idx:end_idx, :]
            # [B, chunk_len, 32768]
            logits = tokenizer_head(feat_chunk)
            codes = logits.argmax(dim=-1)
            tokens_list.append(codes)
        return torch.cat(tokens_list, dim=1)

    def audio_to_yue2_codes(
        self,
        *,
        waveform: torch.Tensor,
        sample_rate: float,
    ) -> torch.Tensor:
        sr = 24000.0
        model_management.load_model_gpu(self.patcher)
        device = self.load_device
        if waveform.ndim not in {1, 2, 3}:
            raise ValueError("Can only handle 1-3D inputs")
        if waveform.ndim == 1:
            waveform = waveform[None, None]
        elif waveform.ndim == 2:
            waveform = waveform[None]
        waveform = waveform.float().mean(dim=1).to(device=device)
        if sample_rate != sr:
            waveform = resample_audio(waveform, sample_rate, sr)
        feat = self.get_mert2_feats(waveform=waveform)
        return self.tokenize_mert2_feats(feat=feat)
