import math
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, Self

import torch
import transformers
from comfy import model_management, model_prefetch
from comfy.text_encoders import yue2
from comfy.text_encoders.llama import FixedKV, rope_matrix

from . import external, llm
from .yue2_support import TokenizerTopK


class TokenizerTopKGuideConfig(NamedTuple):
    start_frame: int = 0
    end_frame: int = -1
    frame_offset: int = 0
    allow_end_token: bool = False
    k_start: int | None = None
    k_end: int | None = None
    k_min: int = 0
    k_max: int = -1
    k_schedule: str | None = None
    k_schedule_power: float = 1.0
    blend_function: Callable | None = torch.lerp
    scale: float = 0.5
    scale_min: float | None = None
    scale_max: float | None = None
    schedule: str | None = None
    schedule_power: float = 1.0
    unguided_scale: float = 1.0
    align_topk_scale: float = 1.0
    align_topk_blend: float = 0.0
    align_topk_allow_up: bool = True
    align_topk_allow_down: bool = True
    align_topk_full_batch: bool = False
    convert_scores: bool = True
    always_convert_scores: bool = False
    guide_cond: bool = True
    processor_index: int = 0
    topk_chunks: tuple[TokenizerTopK, ...] = ()

    @classmethod
    def build(cls, **kwargs: Any) -> Self:
        fs = frozenset(cls._fields)
        blend_mode = kwargs.pop("blend_mode") if "blend_mode" in kwargs else None
        kwargs = {k: v for k, v in kwargs.items() if k in fs}
        if "blend_function" not in kwargs:
            kwargs["blend_function"] = (
                None if blend_mode is None else external.BLEND_MODES[blend_mode]
            )
        return cls(**kwargs)

    @property
    def last_guide_frame(self) -> int:
        return -1 if not self.topk_chunks else self.topk_chunks[-1].last_frame

    @property
    def eff_start_frame(self):
        return max(0, self.start_frame)

    @property
    def eff_end_frame(self):
        lgf = self.last_guide_frame
        return (
            -1 if lgf < 0 else (lgf if self.end_frame < 0 else min(lgf, self.end_frame))
        )

    def check_frame_match(self, *, offset: int) -> bool:
        offset = offset + self.frame_offset
        sf, ef = self.eff_start_frame, self.eff_end_frame
        return offset >= 0 and sf <= offset <= ef

    def slice_k(
        self,
        tk_chunk: TokenizerTopK,
        *,
        progress: float | None = None,
    ) -> TokenizerTopK | None:
        if self.k_max == 0:
            return None
        chunk_k = tk_chunk.values.shape[-1]
        k_start = self.k_start if self.k_start is not None else 0
        k_end = self.k_end if self.k_end is not None else chunk_k
        k_start, k_end = (max(0, min(v, chunk_k)) for v in (k_start, k_end))
        k_spread = k_end - k_start
        if progress is not None:
            k_spread = round(k_spread * progress)
            if self.k_min > 0:
                k_spread = max(self.k_min, k_spread)
            if self.k_max > 0:
                k_spread = min(self.k_max, k_spread)
            k_end = max(k_start, min(chunk_k, k_start + k_spread))
        # print(
        #     f"\nK schedule: progress={progress}, start={k_start}, end={k_end}, spread={k_spread}",
        # )
        if k_start > k_end or k_spread < 1:
            return None
        return tk_chunk._replace(
            values=tk_chunk.values[..., k_start:k_end],
            indices=tk_chunk.indices[..., k_start:k_end],
        )

    def find_chunk(self, offset: int) -> tuple[TokenizerTopK, int] | None:
        if not self.check_frame_match(offset=offset):
            return None
        offset = offset + self.frame_offset
        for tk_chunk in self.topk_chunks:
            chunk_start = tk_chunk.frame_offset + tk_chunk.f_start
            chunk_end = tk_chunk.frame_offset + tk_chunk.f_end
            if chunk_start <= offset < chunk_end:
                return tk_chunk, offset - tk_chunk.frame_offset
        return None


class TokenizerGuideLogitsProcessor(transformers.LogitsProcessor):
    def __init__(self, *, guide: TokenizerTopKGuideConfig):
        super().__init__()
        self.guide = guide

    @staticmethod
    def _log_softmax_scores(
        scores: torch.Tensor,
        *,
        shift_min: bool = True,
    ) -> torch.Tensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        if shift_min:
            scores -= scores.min(dim=-1, keepdim=True).values
        return scores

    def _get_progress(
        self,
        *,
        offset: int,
        schedule: str | None = None,
        schedule_power: float = 1.0,
        progress_min: float = 0.0,
        progress_max: float = 1.0,
    ) -> float:
        if schedule is None:
            return 1.0
        g = self.guide
        sf, ef = g.eff_start_frame, g.eff_end_frame
        if sf < 0 or ef < 0:
            return 1.0
        pct_range = max(0.0, ef - sf)
        progress = (
            max(0.0, min((offset - sf) / pct_range, 1.0)) if pct_range != 0 else 1.0
        )
        if schedule in {"sin", "inv_sin"}:
            progress = math.sin(progress * math.pi)
            if schedule == "inv_sin":
                progress = 1.0 - progress
        elif schedule == "scale_down":
            progress = 1.0 - progress
        elif schedule != "scale_up":
            raise ValueError("Unknown schedule")
        progress **= schedule_power
        return max(progress_min, min(progress_max, progress))

    def _get_scheduled_scale(self, *, offset: int) -> float:
        g = self.guide
        scale = g.scale * self._get_progress(
            offset=offset,
            schedule=g.schedule,
            schedule_power=g.schedule_power,
        )
        if g.scale_min is not None:
            scale = max(g.scale_min, scale)
        if g.scale_max is not None:
            scale = min(g.scale_max, scale)
        # print(
        #     f"\nSCHEDULE: progress={progress:.6f}, orig scale={g.scale:.6f}, adjusted scale={scale:.6f}",
        # )
        return scale

    def _align_topk(
        self,
        *,
        b_offset: int,
        codec_scores: torch.Tensor,
        guide_logits: torch.Tensor,
    ) -> torch.Tensor:
        g = self.guide
        if g.align_topk_blend == 0 or not (
            g.align_topk_allow_up or g.align_topk_allow_down
        ):
            return guide_logits
        golden_logit = (
            codec_scores.max()
            if g.align_topk_full_batch
            else codec_scores[b_offset].max()
        ).mul_(g.align_topk_scale)
        max_guide_logit = guide_logits.max()
        align_diff = (golden_logit - max_guide_logit).mul_(g.align_topk_blend)
        if not (g.align_topk_allow_up and g.align_topk_allow_down):
            align_diff = align_diff.clamp_(
                None if g.align_topk_allow_down else 0.0,
                None if g.align_topk_allow_up else 0.0,
            )
        return guide_logits + align_diff

    # Uses in-place ops.
    def _guide(
        self,
        *,
        guide_frame_idx: int,
        scale: float,
        tk_chunk: TokenizerTopK,
        codec_scores: torch.Tensor,
    ) -> torch.Tensor:
        g = self.guide
        if g.unguided_scale == 1 and g.blend_function is None:
            return codec_scores
        bf = g.blend_function
        bidx = int(not g.guide_cond)
        if bidx >= codec_scores.shape[0]:
            raise ValueError("Batch offset out of range")
        batch_slice = slice(0 + bidx, 1 + bidx)
        chunk_indexing = (slice(None, 1), guide_frame_idx, ...)
        guide_indices = tk_chunk.indices[chunk_indexing].to(device=codec_scores.device)
        model_logits = codec_scores[batch_slice].gather(-1, guide_indices)
        if bf is None:
            guided_logits = model_logits
        else:
            guide_logits = tk_chunk.values[chunk_indexing].to(codec_scores)
            guide_logits = self._align_topk(
                b_offset=bidx,
                codec_scores=codec_scores,
                guide_logits=guide_logits,
            )
            guided_logits = bf(model_logits, guide_logits, scale)
        if g.unguided_scale != 1.0:
            codec_scores[bidx] *= g.unguided_scale
        codec_scores[bidx].scatter_(-1, guide_indices[0], guided_logits[0])
        return codec_scores

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        g = self.guide
        if not g.guide_cond and scores.shape[0] != 2:
            raise ValueError(
                "guide_cond disabled but there is no uncond to guide (CFG disabled)",
            )
        offset = input_ids.shape[-1]
        maybe_chunk = g.find_chunk(offset=offset)
        offset += g.frame_offset
        if maybe_chunk:
            tk_chunk, guide_frame_idx = maybe_chunk
            k_progress = (
                None
                if g.k_schedule is None
                else self._get_progress(
                    offset=offset,
                    schedule=g.k_schedule,
                    schedule_power=g.k_schedule_power,
                )
            )
            tk_chunk = g.slice_k(tk_chunk=tk_chunk, progress=k_progress)
            maybe_chunk = bool(tk_chunk)
        matched = bool(maybe_chunk)
        codec_slice = slice(yue2.CODEC_OFFSET, yue2.CODEC_OFFSET + yue2.CODEC_SIZE)
        scores = scores.clone()
        if g.convert_scores and g.always_convert_scores:
            codec_scores = self._log_softmax_scores(scores[..., codec_slice])
        scale = g.scale
        if matched and g.blend_function is not None:
            # We won't end up here if offset is negative.
            scale = self._get_scheduled_scale(offset=offset)
            if scale == 0:
                matched = False
        if not matched:
            if g.convert_scores and g.always_convert_scores:
                scores[..., codec_slice] = codec_scores
            return scores
        if g.convert_scores and not g.always_convert_scores:
            codec_scores = self._log_softmax_scores(scores[..., codec_slice])
        elif not g.convert_scores:
            codec_scores = scores[..., codec_slice].clone()
        scores[..., codec_slice] = self._guide(
            scale=scale,
            tk_chunk=tk_chunk,
            guide_frame_idx=guide_frame_idx,
            codec_scores=codec_scores,
        )
        if not g.allow_end_token:
            scores[..., yue2.MUSIC_END] = -math.inf
        return scores


class ModelYuE2LLM:
    def __init__(
        self,
        *,
        model: object,
        execution_dtype: torch.dtype | None = None,
        pad_token: int | None = None,
        eos_token: int | None = None,
        device: torch.device | str | None = None,
        hidden_cfg_blend_function: Callable = torch.lerp,
        hidden_cfg_scale: float = 1.0,
        hidden_cfg_cond: bool = True,
        logits_cfg_scale: float = 1.0,
    ):
        self.model = model
        self.logits_cfg_scale = logits_cfg_scale
        self.hidden_cfg_cond = hidden_cfg_cond
        self.hidden_cfg_scale = hidden_cfg_scale
        self.hidden_cfg_blend_function = hidden_cfg_blend_function
        special_tokens = getattr(model, "special_tokens", {})
        self.pad_token = 0
        self.eos_token = (
            eos_token if eos_token is not None else special_tokens.get("eos")
        )
        self.device = model.execution_device
        if execution_dtype is None:
            self.dtype = (
                torch.bfloat16
                if model_management.should_use_bf16(self.device)
                else torch.float32
            )
        else:
            self.dtype = execution_dtype
        self.reset()

    def reset(self) -> None:
        self.kv_cache = None
        self.attention_mask = None
        self.logits = None

    def prepare(
        self,
        ids: Sequence[Sequence[int]],
        *,
        min_tokens: int = 1,
        max_tokens: int = 32767,
        reset_state: bool = True,
    ) -> None:
        if reset_state:
            self.reset()
        prefixes = [list(pfx) for pfx in ids]
        if len(prefixes) not in {1, 2}:
            raise ValueError("We only support 1 or 2 (CFG) prefixes here")
        self.prefixes = prefixes
        max_pfx = max(len(pfx) for pfx in prefixes)
        self.max_prefix = max_pfx
        logits, kv_cache, mask = self.model._prefill(
            prefixes,
            max_pfx + max_tokens,
            self.dtype,
        )
        self.attention_mask = mask
        self.kv_cache = kv_cache
        self.logits = logits

    def __call__(
        self,
        ids: Sequence[Sequence[int]],
        *,
        min_tokens: int = 1,
        max_tokens: int = 32776,
        reset_state: bool = True,
    ):
        device = self.device
        dtype = self.dtype
        self.prepare(
            ids,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            reset_state=reset_state,
        )
        if self.logits is None:
            raise RuntimeError
        fixed_kv = isinstance(self.kv_cache[0], FixedKV)
        step = 0
        decode_tokens = torch.empty(
            (len(self.prefixes), 1),
            device=device,
            dtype=torch.long,
        )
        positions = torch.tensor(
            [[len(p)] for p in self.prefixes],
            device=device,
            dtype=torch.long,
        )
        model = self.model.model
        # Decoder inputs and rotary tensors must keep their addresses across graph replays.
        decode_buffers = None
        if fixed_kv:
            decode_buffers = (
                torch.empty(
                    (len(self.prefixes), 1, model.config.hidden_size),
                    device=device,
                    dtype=dtype,
                ),
                rope_matrix(model.compute_freqs_cis(positions, device)),
            )
        else:
            decode_buffers = None
        use_attn_mask = self.attention_mask is not None and not fixed_kv
        mask = self.attention_mask
        hidden_cfg = self.hidden_cfg_scale
        cfg_cond = not self.hidden_cfg_cond

        try:
            while True:
                next_tokens = yield self.logits
                if not next_tokens:
                    break

                decode_tokens.fill_(next_tokens[0][0])
                if use_attn_mask:
                    mask = self.attention_mask[:, : self.max_prefix + step + 1]
                if fixed_kv:
                    model_prefetch.malloc_graph_begin(device)
                output = model(
                    decode_tokens,
                    past_key_values=self.kv_cache,
                    dtype=dtype,
                    position_ids=positions,
                    attention_mask=mask,
                    decode_buffers=decode_buffers,
                )
                self.kv_cache = output[2]
                hidden = output[0][:, -1]
                if hidden_cfg != 1:
                    hidden_guided = self.hidden_cfg_blend_function(
                        hidden[1 if cfg_cond else 0][None],
                        hidden[0 if cfg_cond else 1][None],
                        hidden_cfg,
                    )
                    if self.logits_cfg_scale != 1:
                        hidden = torch.cat(
                            (hidden_guided, hidden[1][None])
                            if cfg_cond
                            else (hidden[0][None], hidden_guided),
                            dim=0,
                        )
                    else:
                        hidden = hidden_guided
                    del hidden_guided
                logits = model.lm_head(hidden)
                self.logits.copy_(logits)
                # if logits.shape[0] < self.logits.shape[0]:
                #     self.logits[:1].copy_(logits[0])
                #     if logits.shape[0] == 1 and self.logits.shape[0] == 2:
                #         self.logits[1:2].copy_(logits[0])
                # else:
                #     self.logits.copy_(logits)
                # self.logits.copy_(model.lm_head(hidden))
                del output, hidden, logits
                if fixed_kv:
                    model_prefetch.malloc_graph_end()
                positions += 1
                step += 1
        finally:
            model_prefetch.cleanup_prefetch_queues()


class YuE2LLMSamplingState(llm.CustomNoiseLLMSamplingState):
    def __init__(self, *args: Any, **kwargs: Any):
        audio_only = kwargs.pop("audio_only", None)
        tokenizer_guide = kwargs.pop("tokenizer_guide", None)
        super().__init__(*args, **kwargs)
        if audio_only is not None:
            if audio_only:
                start_id, end_id = (
                    yue2.CODEC_OFFSET,
                    yue2.CODEC_OFFSET + yue2.CODEC_SIZE - 1,
                )
            else:
                start_id, end_id = 0, 151642
            self.logits_processors.insert(
                1,
                llm.BiasTokenIdsLogitsProcessor(
                    ranges=(
                        (self.eos_token_id, self.eos_token_id),
                        (start_id, end_id),
                    ),
                ),
            )
        if tokenizer_guide is not None:
            self.logits_processors.insert(
                tokenizer_guide.processor_index,
                TokenizerGuideLogitsProcessor(guide=tokenizer_guide),
            )
