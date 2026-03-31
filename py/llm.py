import contextlib
import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import torch
import transformers
from comfy import model_management
from comfy import utils as comfy_utils
from tqdm import tqdm

from . import external
from .utils import TieredBlendWrapper, nanstd


class WindowedLogitsProcessor(transformers.LogitsProcessor):
    def __init__(
        self,
        *args: Any,
        # None - no window. positive value - from start of IDs. negative - from end.
        window: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.window = window

    # Returns the original or a view.
    def _get_windowed_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        n_ids = input_ids.shape[-1]
        if not n_ids or self.window is None:
            return input_ids
        wneg = self.window < 0
        window = n_ids + self.window if wneg else self.window
        window = max(0, min(window, n_ids - 1))
        if window < 1:
            return input_ids
        ids_slice = slice(None, window + 1) if not wneg else slice(window, None)
        return input_ids[..., ids_slice]


class BiasTokenIdsLogitsProcessor(transformers.LogitsProcessor):
    def __init__(
        self,
        ranges,
        *,
        invert_ranges: bool = True,
        bias_value: float = -math.inf,
        # Multiply if false.
        add_bias: bool = True,
    ):
        super().__init__()
        self.ranges = tuple(tuple(r) for r in ranges)
        self.invert_ranges = invert_ranges
        self.mask: torch.Tensor | None = None
        self.bias_value = bias_value
        self.add_bias = add_bias

    def _get_mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        if self.mask is not None and self.mask.shape[-1] == vocab_size:
            if self.mask.device != device:
                self.mask = self.mask.to(device=device)
            return self.mask
        if not self.ranges:
            self.mask = torch.full(
                vocab_size,
                not self.invert_ranges,
                dtype=torch.bool,
                device=device,
            )
            return self.mask
        ids = torch.arange(vocab_size, dtype=torch.int64, device=device)
        mask = None
        for rstart, rend in self.ranges:
            curr_mask = (ids >= rstart) & (ids <= rend)
            mask = mask | curr_mask if mask is not None else curr_mask
        if mask is not None and self.invert_ranges:
            mask = ~mask
        self.mask = mask
        return mask

    def __call__(
        self,
        input_ids: torch.LongTensor,  # noqa: ARG002
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if not self.ranges:
            return scores.clone()
        mask = self._get_mask(scores.shape[-1], device=scores.device)
        mask = mask.reshape(*((1,) * (scores.ndim - 1)), -1)
        adjusted_scores = (
            scores + self.bias_value if self.add_bias else scores * self.bias_value
        )
        return torch.where(mask, adjusted_scores, scores)


class NoRepeatNGramExtLogitsProcessor(WindowedLogitsProcessor):
    def __init__(
        self,
        ngram_size: int,
        *args: Any,
        penalty: float = -math.inf,
        add_penalty: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.processor = transformers.NoRepeatNGramLogitsProcessor(ngram_size)
        self.penalty = penalty
        self.add_penalty = add_penalty

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        input_ids = self._get_windowed_ids(input_ids)

        # Transformers logits processors don't do in-place operations, so this is safe.
        processed_scores = self.processor(input_ids, scores)
        # The Transformers logits processor either penalizes by -inf or does nothing.
        mask = scores != processed_scores
        processed_scores = (
            scores + self.penalty if self.add_penalty else scores * self.penalty
        )
        return torch.where(mask, processed_scores, scores)


class RepetitionPenaltyExtLogitsProcessor(WindowedLogitsProcessor):
    def __init__(self, *args: Any, **kwargs: Any):
        window = kwargs.pop("window", None)
        super().__init__(window=window)
        self.processor = transformers.RepetitionPenaltyLogitsProcessor(*args, **kwargs)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        input_ids = self._get_windowed_ids(input_ids)
        return self.processor(input_ids, scores)


class ForbidPrefixLogitsProcessor(transformers.LogitsProcessor):
    def __init__(self, prefix: list[int] | tuple[int, ...]):
        super().__init__()
        self.forbidden_prefix = tuple(prefix)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        pfx = self.forbidden_prefix
        pfx_len = len(pfx)
        n_ids = input_ids.shape[-1]
        token_id = pfx[n_ids - 1] if n_ids < pfx_len else -1
        scores = scores.clone()
        if (
            n_ids == 0
            or n_ids > pfx_len
            or token_id < 0
            or token_id >= scores.shape[-1]
        ):
            return scores
        scores[..., token_id] = -math.inf
        return scores


class CFGExtLogitsProcessor(transformers.LogitsProcessor):
    @staticmethod
    def simple_cfg(
        cond: torch.Tensor,
        uncond: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        if scale == 1.0:
            return cond.clone()
        if scale == 0.0:
            return uncond.clone()
        return (cond - uncond).mul_(scale).add_(uncond)

    @staticmethod
    @lru_cache(maxsize=16)
    def get_fake_channels(size: int, *, target: int = 256) -> int:
        for divisor in range(target, 0, -1):
            if float(size / divisor).is_integer():
                return divisor
        return 1

    def __init__(
        self,
        guidance_scale: float = 1.0,
        *,
        start_pos: int = 0,
        end_pos: int = -1,
        blend_mode: str | dict | None = None,
        # "cfg" blend cond with CFG result, "diff": blend CFG diff with cond, None or other: pass cond and uncond.
        blend_strategy: str | None = None,
        blend_tiers: int = 0,
        # None/other = normal logits, probs, logprobs
        guidance_space: str | None = None,
        plausibility_mask_top_k: int = 0,
        plausibility_mask_min_p: float = -1.0,
        variance_rescaling_strength: float = 0.0,
        variance_rescaling_target: str = "cond",
        mean_rescaling_strength: float = 0.0,
        mean_rescaling_target: str = "cond",
        temperature: float = 1.0,
        final_remask: bool = True,
    ):
        self.guidance_scale = guidance_scale
        self.guidance_space = guidance_space
        self.temperature = temperature if temperature != 0.0 else 1.0
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.plausibility_mask_top_k = plausibility_mask_top_k
        self.plausibility_mask_min_p = plausibility_mask_min_p
        self.variance_rescaling_strength = variance_rescaling_strength
        self.variance_rescaling_target = variance_rescaling_target
        self.mean_rescaling_strength = mean_rescaling_strength
        self.mean_rescaling_target = mean_rescaling_target
        self.final_remask = final_remask

        self.blend_strategy = blend_strategy
        if blend_mode is not None:
            external.ensure_blend_modes()
            self.blend_function = external.BLEND_MODES[blend_mode]
            if blend_tiers > 0:
                self.blend_function = TieredBlendWrapper(
                    self.blend_function,
                    tiers=blend_tiers,
                    start_dim=1,
                    end_dim=-1,
                    descending=True,
                    pad_value=0.0,
                )
        else:
            self.blend_function = self.simple_cfg

    def _to_guidance_space(
        self, cond: torch.Tensor, uncond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.guidance_space not in {"probs", "logprobs"}:
            return cond, uncond
        cond = cond / self.temperature if self.temperature != 1.0 else cond
        uncond = uncond / self.temperature if self.temperature != 1.0 else uncond
        fun = (
            torch.nn.functional.log_softmax
            if self.guidance_space == "logprobs"
            else torch.softmax
        )
        return fun(cond, dim=-1), fun(uncond, dim=-1)

    # Uses in-place operations.
    def _from_guidance_space(
        self,
        result: torch.Tensor,
        *,
        min_logit: float = 1e-08,
    ) -> torch.Tensor:
        if self.guidance_space == "probs":
            result = result.clamp_(min=min_logit)
            result /= result.sum(dim=-1, keepdim=True)
            result = result.log_()
        if self.guidance_space in {"probs", "logprobs"} and self.temperature != 1.0:
            result *= self.temperature
        return result

    def _get_rescaling_target(
        self,
        mode: str,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        result: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor | None:
        if mode == "cond":
            return cond
        if mode == "uncond":
            return uncond
        if mode == "diff":
            return cond - uncond
        if mode == "neg_diff":
            return uncond - cond
        return None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.Tensor:
        tbatch, sbatch = input_ids.shape[0], scores.shape[0]
        if self.guidance_scale == 1.0 and tbatch == sbatch:
            return scores.clone()
        if tbatch * 2 != sbatch:
            errstr = f"Unexpected batch sizes. Tokens shape: {input_ids.shape}, logits share: {scores.shape}"
            raise ValueError(errstr)
        cond, uncond = scores.split(tbatch, dim=0)
        cond_orig, uncond_orig = cond, uncond
        ntoks = input_ids.shape[-1]
        stok, etok = (
            pos if pos >= 0 else ntoks + pos for pos in (self.start_pos, self.end_pos)
        )
        if not stok <= (ntoks - 1) <= etok:
            return cond.clone()
        dtype = cond.dtype
        finfo = torch.finfo(dtype)
        ninf_proxy = finfo.min * 0.85
        pinf_proxy = finfo.max * 0.85
        ntn_args = (ninf_proxy, pinf_proxy, ninf_proxy)
        uncond = torch.where(uncond.isfinite(), uncond, cond)
        mask = cond.isfinite()
        if not torch.any(mask):
            return torch.full_like(cond, -torch.inf)
        result = cond.clone()
        fake_shape = (
            (tbatch, self.get_fake_channels(cond.shape[-1]), 1, -1)
            if cond.ndim < 3 or cond.shape[1] < 2
            else cond.shape
        )
        blend = self.guidance_scale
        cond = cond.nan_to_num(*ntn_args)
        uncond = uncond.nan_to_num(*ntn_args)
        pm_topk = self.plausibility_mask_top_k
        pm_minp = self.plausibility_mask_min_p
        if pm_topk > 0 or pm_minp >= 0.0:
            plausibility_mask = torch.ones_like(cond, dtype=torch.bool)
            if pm_topk > 0:
                topk_indices = torch.topk(cond, k=pm_topk, dim=-1)[1]
                tk_mask = torch.zeros_like(cond, dtype=torch.bool)
                tk_mask.scatter_(index=topk_indices, dim=-1, value=True)
                plausibility_mask &= tk_mask
            if pm_minp >= 0.0:
                p_cond = torch.softmax(cond / self.temperature, dim=-1)
                max_p = p_cond.max(dim=-1, keepdim=True)[0]
                mp_mask = p_cond >= (max_p * pm_minp)
                plausibility_mask &= mp_mask
            mask &= plausibility_mask
        else:
            plausibility_mask = None
        cond, uncond = self._to_guidance_space(cond, uncond)
        if self.blend_strategy == "diff":
            blend_b = cond - uncond
        elif self.blend_strategy == "cfg":
            blend_b = self.simple_cfg(cond, uncond, self.guidance_scale)
            blend = 1.0
        else:
            blend_b = uncond
        result[mask] = self.blend_function(
            cond.reshape(fake_shape),
            blend_b.reshape(fake_shape),
            blend,
        ).reshape(cond_orig.shape)[mask]
        result = self._from_guidance_space(result)
        if self.variance_rescaling_strength != 0.0:
            vr_target = self.variance_rescaling_target
            target = self._get_rescaling_target(
                vr_target,
                cond_orig,
                uncond_orig,
                result,
            )
            if target is not None:
                std_target = nanstd(target, dim=-1, keepdim=True)
            elif vr_target in {"min", "max"}:
                fun = torch.minimum if vr_target == "min" else torch.maximum
                std_target = fun(
                    *(
                        nanstd(t, dim=-1, keepdim=True)
                        for t in (cond_orig, uncond_orig)
                    ),
                )
            else:
                raise ValueError("Unsupported variance_rescaling_target value")
            std_result = nanstd(result, dim=-1, keepdim=True).add_(1e-08)
            result_rescaled = result * (std_target / std_result)
            result = result.lerp(result_rescaled, self.variance_rescaling_strength)
        if self.mean_rescaling_strength != 0.0:
            mr_target = self.mean_rescaling_target
            target = self._get_rescaling_target(
                mr_target,
                cond_orig,
                uncond_orig,
                result,
            )
            if target is not None:
                mean_target = torch.nanmean(target, dim=-1, keepdim=True)
            elif mr_target in {"min", "max"}:
                fun = torch.minimum if vr_target == "min" else torch.maximum
                mean_target = fun(
                    *(
                        torch.nanmean(t, dim=-1, keepdim=True)
                        for t in (cond_orig, uncond_orig)
                    ),
                )
            else:
                raise ValueError("Unsupported variance_rescaling_target value")
            mean_result = torch.nanmean(result, dim=-1, keepdim=True).neg_()
            result += mean_result.add_(mean_target).mul_(self.mean_rescaling_strength)
        if self.final_remask and plausibility_mask is not None:
            inv_mask = ~plausibility_mask
            result[inv_mask] = cond_orig[inv_mask]
        return result


class LLMSamplingState:
    def __init__(
        self,
        *,
        device: str | torch.device,
        min_tokens: int,
        max_tokens: int,
        sampling_dtype: str | None = None,
        **kwargs: Any,
    ):
        self.device = device
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        if sampling_dtype is not None:
            valid_dtypes = {
                "float64": torch.float64,
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            self.sampling_dtype = valid_dtypes.get(sampling_dtype)
            if self.sampling_dtype is None:
                raise ValueError("Unsupported sampling_dtype value")
        else:
            self.sampling_dtype = torch.float64
        self.max_tokens_expdecay_factor = kwargs.pop("max_tokens_expdecay_factor", 0.85)
        self.max_tokens_expdecay_penalty = kwargs.pop(
            "max_tokens_expdecay_penalty",
            1.01,
        )
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.temperature = kwargs.pop("temperature", 0.85)
        cfg_kwargs_map = {
            "cfg_scale": "guidance_scale",
            "cfg_start_pos": "start_pos",
            "cfg_end_pos": "end_pos",
            "cfg_blend_mode": "blend_mode",
            "cfg_blend_strategy": "blend_strategy",
            "cfg_blend_tiers": "blend_tiers",
            "cfg_guidance_space": "guidance_space",
            "cfg_plausibility_mask_top_k": "plausibility_mask_top_k",
            "cfg_plausibility_mask_min_p": "plausibility_mask_min_p",
            "cfg_variance_rescaling_strength": "variance_rescaling_strength",
            "cfg_variance_rescaling_target": "variance_rescaling_target",
            "cfg_mean_rescaling_strength": "mean_rescaling_strength",
            "cfg_mean_rescaling_target": "mean_rescaling_target",
            "cfg_final_remask": "final_remask",
        }
        self.cfg_kwargs = {
            cfg_k: kwargs.pop(k) for k, cfg_k in cfg_kwargs_map.items() if k in kwargs
        }
        self.cfg_kwargs["temperature"] = self.temperature
        self.cfg_scale = self.cfg_kwargs.get("guidance_scale", 1.0)
        if self.temperature > 0:
            self.seed = kwargs.pop("seed", 0)
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(self.seed)
        self.top_p = kwargs.pop("top_p", 0.9)
        self.top_k = kwargs.pop("top_k", 0)
        self.min_p = kwargs.pop("min_p", -1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 0.0)
        self.repetition_penalty_window = kwargs.pop("repetition_penalty_window", None)
        self.no_repeat_ngram_penalty = kwargs.pop("no_repeat_ngram_penalty", -math.inf)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.no_repeat_ngram_add_penalty_mode = bool(
            kwargs.pop("no_repeat_ngram_add_penalty_mode", True),
        )
        self.no_repeat_ngram_window = kwargs.pop("no_repeat_ngram_window", None)
        self.logits_processors = self.get_logits_processors()

    def get_logits_processors(self) -> transformers.LogitsProcessorList:
        device = self.device
        lp_list = transformers.LogitsProcessorList()
        if self.cfg_scale != 1.0:
            lp_list.append(CFGExtLogitsProcessor(**self.cfg_kwargs))
        if self.repetition_penalty > 0.0:
            lp_list.append(
                RepetitionPenaltyExtLogitsProcessor(
                    penalty=self.repetition_penalty,
                    window=self.repetition_penalty_window,
                ),
            )
        if self.no_repeat_ngram_size > 0 and self.no_repeat_ngram_penalty != 0.0:
            lp_list.append(
                NoRepeatNGramExtLogitsProcessor(
                    ngram_size=self.no_repeat_ngram_size,
                    penalty=self.no_repeat_ngram_penalty,
                    add_penalty=bool(self.no_repeat_ngram_add_penalty_mode),
                    window=self.no_repeat_ngram_window,
                ),
            )
        if self.eos_token_id is not None and self.min_tokens > 0:
            lp_list.append(
                transformers.MinLengthLogitsProcessor(
                    min_length=self.min_tokens,
                    eos_token_id=self.eos_token_id,
                    device=device,
                ),
            )
        if (
            self.eos_token_id is not None
            and self.max_tokens_expdecay_factor != 1.0
            and self.max_tokens_expdecay_penalty != 1.0
        ):
            expdecay_start = int(self.max_tokens * self.max_tokens_expdecay_factor)
            lp_list.append(
                transformers.ExponentialDecayLengthPenalty(
                    (expdecay_start, self.max_tokens_expdecay_penalty),
                    self.eos_token_id,
                    # Start token index - we only pass new tokens to these processors and don't consider the prompts.
                    0,
                ),
            )
        if self.temperature != 1.0:
            lp_list.append(
                transformers.TemperatureLogitsWarper(temperature=self.temperature)
            )
        if self.top_k > 0:
            lp_list.append(transformers.TopKLogitsWarper(top_k=self.top_k))
        if 0.0 < self.top_p < 1.0:
            lp_list.append(transformers.TopPLogitsWarper(top_p=self.top_p))
        if 0 <= self.min_p <= 1.0:
            lp_list.append(transformers.MinPLogitsWarper(min_p=self.min_p))
        return lp_list

    def sample(self, logits: torch.Tensor) -> list[int]:
        logits = logits.to(dtype=self.sampling_dtype)
        logits = logits.nan_to_num(nan=-math.inf)
        if self.temperature <= 0:
            tids = logits.argmax(dim=-1)
        else:
            tids = torch.multinomial(
                torch.softmax(logits, dim=-1),
                num_samples=1,
                generator=self.generator,
            ).squeeze(1)
        return tids.flatten().detach().cpu().tolist()

    def filter_and_sample(
        self,
        past_ids: torch.LongTensor | tuple | list,
        logits: torch.Tensor,
    ) -> list[int]:
        logits = logits.to(dtype=self.sampling_dtype)
        if not isinstance(past_ids, torch.Tensor):
            past_ids = torch.tensor(
                [list(ptids) for ptids in past_ids],
                device=logits.device,
                dtype=torch.int64,
            )
        # tqdm.write(f"** SHAPES: {past_ids.shape}, {logits.shape}")
        logits = self.logits_processors(past_ids, logits)
        return self.sample(logits)


class ACE15LLMSamplingState(LLMSamplingState):
    def __init__(self, *args: Any, **kwargs: Any):
        ace15_audio_only = kwargs.pop("ace15_audio_only", None)
        forbid_prefix = kwargs.pop("tokens_forbid_prefix", None)
        custom_noise = kwargs.pop("custom_noise", None)
        custom_noise_topk = kwargs.pop("custom_noise_topk", 250)
        custom_noise_topk_temperature_rescaling = kwargs.pop(
            "custom_noise_topk_temperature_rescaling",
            False,
        )
        dynamic_noise_power = kwargs.pop("custom_noise_dynamic_power", 0.0)
        custom_noise_normalized = kwargs.pop("custom_noise_normalized", True)
        custom_noise_blend_mode = kwargs.pop("custom_noise_blend_mode", "inject")
        custom_noise_blend = kwargs.pop("custom_noise_blend", 1.0)
        custom_noise_blend_tiers = kwargs.pop("custom_noise_blend_tiers", 0)
        super().__init__(*args, **kwargs)
        if ace15_audio_only is not None:
            if bool(ace15_audio_only):
                start_id, end_id = 151669, 215669
            else:
                start_id, end_id = 0, 151668
            self.logits_processors.insert(
                1,
                BiasTokenIdsLogitsProcessor(
                    ranges=(
                        (self.eos_token_id, self.eos_token_id),
                        (start_id, end_id),
                    ),
                ),
            )
        if forbid_prefix:
            # Shouldn't matter if this is before or after token biasing.
            self.logits_processors.insert(1, ForbidPrefixLogitsProcessor(forbid_prefix))
        self.custom_noise = custom_noise
        if custom_noise is None:
            return
        if not hasattr(custom_noise, "make_noise_sampler"):
            raise ValueError(
                "Custom noise sampler has no make_noise_sampler attribute!",
            )
        self.custom_noise_normalized = custom_noise_normalized
        self.custom_noise_blend_function = external.BLEND_MODES[custom_noise_blend_mode]
        self.custom_noise_blend = custom_noise_blend
        self.custom_noise_topk = custom_noise_topk
        self.custom_noise_topk_temperature_rescaling = (
            custom_noise_topk_temperature_rescaling
        )
        self.dynamic_noise_power = dynamic_noise_power
        if custom_noise_blend_mode != "inject" and custom_noise_blend_tiers > 0:
            self.custom_noise_blend_function = TieredBlendWrapper(
                self.custom_noise_blend_function,
                tiers=custom_noise_blend_tiers,
                start_dim=1,
                end_dim=-1,
                descending=True,
                pad_value=0.0,
            )
        self.noise_sampler = None

    def _get_custom_noise_topk_std(
        self,
        logits: torch.Tensor,
        *,
        score_cutoff: float = 1000,
    ) -> torch.Tensor | None:
        if self.custom_noise_topk < 1:
            return None
        k = min(self.custom_noise_topk, logits.shape[-1])
        tk_vals, _tk_idxs = torch.topk(
            logits * self.temperature
            if self.custom_noise_topk_temperature_rescaling
            and self.temperature not in {0.0, 1.0}
            else logits,
            k=k,
        )

        # Best scores (per batch)
        best_scores = tk_vals[:, :1]
        cutoff = best_scores - score_cutoff

        # Replace garbage with NAN so we can use nanstd
        # This handles -inf, -3.4e38, and just generally "bad" tokens
        tk_vals[(tk_vals < cutoff) | tk_vals.isinf()] = torch.nan

        topk_std = nanstd(tk_vals.to(dtype=torch.float64), dim=-1, keepdim=True)

        # Edge Case: If only 1 token was valid (rest were NaNs), std is NaN.
        # Or if std is 0 (all tokens identical).
        # We fill nans with 0.0 or 1.0 depending on how you want to handle "single choice" scenarios.
        # Usually if std is 0/NaN, we shouldn't inject noise that scales with it, so 0 is safe.
        return torch.nan_to_num(topk_std, nan=0.0).to(dtype=logits.dtype)

    def _do_blend(
        self,
        *,
        logits: torch.Tensor,
        noise: torch.Tensor,
        noise_shape: torch.Size | tuple[int, ...],
        topk_std: torch.Tensor | None,  # noqa: ARG002
        # Proxy values for -inf and +inf, this percentage of the dtype min/max values.
        ninf_proxy_multiplier: float = 0.85,
        pinf_proxy_multiplier: float = 0.85,
    ) -> torch.Tensor:
        invalid_mask = ~(logits.isfinite() & noise.isfinite())
        finfo = torch.finfo(noise.dtype)
        ninf_proxy = finfo.min * ninf_proxy_multiplier
        pinf_proxy = finfo.max * pinf_proxy_multiplier
        ntn_args = (ninf_proxy, pinf_proxy, ninf_proxy)
        result = self.custom_noise_blend_function(
            logits.nan_to_num(*ntn_args).reshape(noise_shape),
            noise.nan_to_num_(*ntn_args).reshape(noise_shape),
            self.custom_noise_blend,
        ).reshape(logits.shape)
        invalid_mask = ~result.isfinite()
        # tqdm.write(
        #     f"SAMPLING NOISE: shape={noise.shape} ({noise_shape}), topk std={topk_std}, noise std={noise.std(dim=-1, keepdim=True)}, noise min/max={(*noise.aminmax(dim=-1, keepdim=True),)}, valid={noise.numel() - int(invalid_mask.float().sum().item())}",
        # )
        result[invalid_mask] = logits[invalid_mask]
        return result

    def sample(self, logits: torch.Tensor) -> list[int]:
        if self.temperature <= 0 or self.custom_noise is None:
            return super().sample(logits)
        if logits.dtype not in {torch.float32, torch.float64}:
            logits = logits.to(dtype=torch.float32)
        logits = logits.nan_to_num(nan=-math.inf)
        if self.noise_sampler is None:
            seed = getattr(self, "seed", None)
            if seed is not None:
                torch.manual_seed(seed)
            noise_ref = (
                logits.reshape(logits.shape[0], 1, 1, -1) if logits.ndim < 4 else logits
            )
            noise_sampler = self.custom_noise.make_noise_sampler(
                noise_ref,
                cpu=logits.device == torch.device("cpu"),
                seed=getattr(self, "seed", None),
                normalized=self.custom_noise_normalized,
            )
            self.noise_sampler = noise_sampler
        topk_std = self._get_custom_noise_topk_std(logits)
        fake_sigma = logits.new_tensor(1.0)
        noise = self.noise_sampler(fake_sigma, fake_sigma * 0.5)
        noise_shape = noise.shape
        noise = noise.reshape(logits.shape)
        if topk_std is not None:
            if self.dynamic_noise_power != 0.0:
                topk_std = (
                    topk_std.abs().pow_(self.dynamic_noise_power).copysign_(topk_std)
                )
            noise *= topk_std
        result = self._do_blend(
            logits=logits,
            noise=noise,
            noise_shape=noise_shape,
            topk_std=topk_std,
        )
        result = result.reshape(logits.shape[0], -1)
        return result.argmax(dim=-1).detach().cpu().tolist()


class ModelLLM:
    def __init__(
        self,
        *,
        model: object,
        execution_dtype: torch.dtype | None = None,
        pad_token: int | None = None,
        eos_token: int | None = None,
        device: torch.device | str | None = None,
    ):
        self.model = model
        special_tokens = getattr(model, "special_tokens", {})
        self.pad_token = (
            pad_token if pad_token is not None else special_tokens.get("pad")
        )
        self.eos_token = (
            eos_token if eos_token is not None else special_tokens.get("eos")
        )
        self.device = (
            torch.device(device) if device is not None else model.execution_device
        )
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
        self.model_state: tuple | None = None

    def build_kv_cache(self, embeds: torch.Tensor, *, min_tokens: int = 1) -> list:
        model_config = self.model.transformer.model.config
        embeds_batch = embeds.shape[0]

        kv_cache = []
        shape = [
            embeds_batch,
            model_config.num_key_value_heads,
            embeds.shape[1] + min_tokens,
            model_config.head_dim,
        ]
        device = embeds.device

        for _ in range(model_config.num_hidden_layers):
            kv_cache.append(
                (
                    torch.empty(shape, device=device, dtype=self.dtype),
                    torch.empty(shape, device=device, dtype=self.dtype),
                    0,
                )
            )

        return kv_cache

    def get_pad_tokens(
        self,
        ids: Sequence[Sequence[int]],
    ) -> tuple[tuple[int, ...], ...]:
        max_len = max(len(tids) for tids in ids)
        pad_token = self.pad_token
        return tuple((pad_token,) * (max_len - len(tids)) for tids in ids)

    def prepare(
        self,
        ids: Sequence[Sequence[int]],
        *,
        min_tokens: int = 1,
        reset_state: bool = True,
    ) -> None:
        if reset_state:
            self.reset()
        pad_tokens = self.get_pad_tokens(ids)
        padded_ids = [[*ptoks, *toks] for toks, ptoks in zip(ids, pad_tokens)]
        self.model_state = self.model.process_tokens(
            padded_ids,
            self.device,
        )
        embeds, attention_mask, _num_tokens, _embeds_info = self.model_state
        for i, pt in enumerate(pad_tokens):
            pad_len = len(pt)
            attention_mask[i, :pad_len] = 0
            attention_mask[i, pad_len:] = 1
        self.attention_mask = attention_mask
        self.kv_cache = self.build_kv_cache(embeds, min_tokens=min_tokens)

    def __call__(
        self,
        ids: Sequence[Sequence[int]],
        *,
        min_tokens: int = 1,
        reset_state: bool = True,
    ):
        device = self.device
        self.prepare(ids, min_tokens=min_tokens, reset_state=reset_state)
        embeds, attention_mask, num_tokens, embeds_info = self.model_state
        embeds_batch = embeds.shape[0]
        while True:
            outputs = self.model.transformer(
                None,
                self.attention_mask,
                embeds=embeds.to(dtype=self.dtype),
                num_tokens=num_tokens,
                intermediate_output=None,
                dtype=self.dtype,
                embeds_info=embeds_info,
                past_key_values=self.kv_cache,
            )
            logits = self.model.transformer.logits(outputs[0])[:, -1]
            self.kv_cache = outputs[2]
            next_tokens = yield logits
            if not next_tokens:
                break
            new_state = self.model.process_tokens(next_tokens, device)
            embeds = new_state[0]
            emb_repeat_factor = embeds_batch // embeds.shape[0]
            if embeds.shape[0] * emb_repeat_factor != embeds_batch:
                errstr = f"Unexpected embeds batch size. Originally had batch size {embeds_batch}, shape now {embeds.shape}."
                raise RuntimeError(errstr)
            elif emb_repeat_factor > 1:
                embeds = embeds.repeat(embeds_batch, *((1,) * (embeds.ndim - 1)))
            self.model_state = self.model_state.__class__((embeds, *new_state[1:]))
            self.attention_mask = torch.cat(
                [
                    self.attention_mask,
                    torch.ones(
                        (embeds_batch, 1),
                        device=device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=1,
            )


class LLMSampling:
    def __init__(self, *, model: object, state_class: type):
        self.model = model
        self.state_class = state_class

    def token_output(
        self,
        *,
        step: int,  # noqa: ARG002
        tokens: list[int],
        current_tokens: list[list[int]] | None = None,  # noqa: ARG002
    ) -> list[int]:
        return tokens

    def __call__(
        self,
        ids: list[list[int]],
        *,
        execution_dtype: torch.dtype | None = None,
        min_tokens: int = 0,
        max_tokens: int = 2048,
        progress: bool = True,
        split_at_eos: bool = True,
    ):
        if not ids:
            return []

        llm = ModelLLM(
            model=self.model,
            execution_dtype=execution_dtype,
        )
        state = self.state_class(
            device=llm.device,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )

        llm_gen = llm(ids, min_tokens=min_tokens)
        progress_bar = comfy_utils.ProgressBar(max_tokens) if progress else None
        step_iter = (
            comfy_utils.model_trange(max_tokens, desc="LM sampling")
            if progress
            else range(max_tokens)
        )

        eos_tid = state.eos_token_id
        output_tokens: list[list[int]] | None = None
        next_tokens = None

        with contextlib.suppress(StopIteration):
            for step in step_iter:
                logits = llm_gen.send(next_tokens)
                out_batch_size = (
                    logits.shape[0] // 2 if state.cfg_scale != 1 else logits.shape[0]
                )
                if output_tokens is None:
                    output_tokens = [[] for _ in range(out_batch_size)]
                tokens = state.filter_and_sample(output_tokens, logits)
                tokens = self.token_output(
                    step=step,
                    tokens=tokens,
                    current_tokens=output_tokens,
                )
                if not any(tid != eos_tid for tid in tokens):
                    break
                if len(output_tokens) == len(tokens):
                    for new_token, tokens_batch in zip(tokens, output_tokens):
                        tokens_batch.append(new_token)
                else:
                    errstr = f"Output token batch size changed on step {step}, was {len(output_tokens)} but sampled {len(tokens)} id(s)!"
                    raise RuntimeError(errstr)
                next_tokens = [[tids] for tids in tokens]
                if progress_bar:
                    progress_bar.update_absolute(step)
        if output_tokens is None:
            return []
        if not split_at_eos or eos_tid is None:
            return output_tokens
        for bidx, btids in enumerate(output_tokens):
            if not btids:
                continue
            with contextlib.suppress(ValueError):
                eos_idx = btids.index(eos_tid)
                output_tokens[bidx] = btids[:eos_idx]
        return output_tokens


class Ace15LLMSampling(LLMSampling):
    def __init__(
        self,
        *,
        tokenizer: object | None = None,
        verbose_interval: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.verbose_interval = verbose_interval
        self.tokenizer = tokenizer

    def token_output(
        self,
        *,
        step: int,
        tokens: list[int],
        current_tokens: list[list[int]] | None = None,
    ) -> list[int]:
        if (
            self.verbose_interval > 0
            and self.tokenizer is not None
            and step > 0
            and current_tokens is not None
            and (step % self.verbose_interval) == 0
        ):
            tqdm.write(f"* LLM sampling at step {step}:")
            for bidx, (ctoks, ntok) in enumerate(zip(current_tokens, tokens)):
                tqdm.write(f"  - Batch {bidx}:")
                decoded = self.tokenizer.decode(
                    [*ctoks[-(self.verbose_interval - 1) :], ntok]
                )
                tqdm.write(decoded)
            tqdm.write("###### END LLM sampling output ######\n")

        return super().token_output(
            step=step,
            tokens=tokens,
            current_tokens=current_tokens,
        )
