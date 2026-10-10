from collections.abc import Callable, Sequence
from typing import Any

import torch
from comfy import model_management, model_prefetch
from comfy.text_encoders import yue2
from comfy.text_encoders.llama import FixedKV, rope_matrix

from . import llm


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
