from functools import partial

import torch
import yaml
from comfy import model_management
from tokenizers import AddedToken

from ..external import ensure_blend_modes
from ..llm import LLMSampling, get_token_idxs
from ..llm_yue2 import ModelYuE2LLM, YuE2LLMSamplingState, yue2


def get_yue2_tokenizer(clip: object) -> object:
    clip_tokenizer = clip.tokenizer
    if not isinstance(clip_tokenizer, yue2.YuE2Tokenizer):
        raise ValueError("CLIP doesn't have a YuE2 tokenizer")  # noqa: TRY004

    clip_tokenizer = clip_tokenizer.__class__(
        tokenizer_data={"yue2_tokenizer_json": clip_tokenizer.tokenizer_json},
    )
    tokenizer = clip_tokenizer.tokenizer
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != 151643:
        raise ValueError("Bad tokenizer vocabular size")
    tokenizer.add_tokens(["<|yue2_eod|>"])
    pre_abc_pad_size = yue2.ABC_START - yue2.EOD - 1
    pre_music_pad_size = yue2.MUSIC_START - yue2.ABC_END - 1
    pre_abc_pad = [
        AddedToken(content=f"<|yue2_preabc_pad_{idx}|>", special=True)
        for idx in range(pre_abc_pad_size)
    ]
    pre_music_pad = [
        AddedToken(content=f"<|yue2_premusic_pad_{idx}|>", special=True)
        for idx in range(pre_music_pad_size)
    ]
    if pre_abc_pad:
        tokenizer.add_tokens(pre_abc_pad)
    tokenizer.add_tokens(["<|abc_start|>", "<|abc_end|>"])
    if pre_music_pad:
        tokenizer.add_tokens(pre_music_pad)
    tokenizer.add_tokens(["<|music_start|>", "<|music_end|>"])
    tokenizer.add_tokens([f"<|audio_code_{i}|>" for i in range(yue2.CODEC_SIZE)])
    if tokenizer.get_vocab_size() != yue2.CODEC_OFFSET + yue2.CODEC_SIZE:
        raise RuntimeError("Bad tokenizer state")
    return tokenizer


class RawTextEncodeYuE2Node:
    DESCRIPTION = "TBD"
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("CONDITIONING", "FLOAT")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": (
                    "STRING",
                    {"defaultInput": True, "dynamicPrompts": False, "multiline": True},
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        clip: object,
        prompt: str,
    ) -> tuple:
        tokenizer = get_yue2_tokenizer(clip)
        tokens = tokenizer.encode(prompt).ids
        abc_start_idx, abc_end_idx, music_start_idx, music_end_idx = get_token_idxs(
            tokens,
            yue2.ABC_START,
            yue2.ABC_END,
            yue2.MUSIC_START,
            yue2.MUSIC_END,
        )
        if music_start_idx is None:
            raise ValueError("Missing <|music_start|> token") from None
        clip_metadata = {
            "cot": "off",
            "cfg_scale": 1.0,
            "max_tokens": 10,
            "prefix": [(t, 1.0) for t in tokens],
        }
        prefix_ids = tokens[: music_start_idx + 1]
        codec_ids = tokens[
            music_start_idx + 1 : None if music_end_idx is None else music_end_idx + 1
        ]
        clip.load_model(clip_metadata)
        csm = clip.cond_stage_model
        if not hasattr(csm, "_acoustic_conditioning"):
            raise RuntimeError("Missing _acoustic_conditioning method")
        csm.set_clip_options({"execution_device": clip.patcher.load_device})
        dtype = (
            torch.bfloat16
            if model_management.should_use_bf16(csm.execution_device)
            else torch.float32
        )
        conditioning_tensor, chunks = csm._acoustic_conditioning(
            prefix_ids,
            codec_ids,
            dtype,
        )
        if (
            abc_start_idx is not None
            and abc_end_idx is not None
            and abc_end_idx - abc_start_idx > 0
        ):
            abc_ids = list(tokens[abc_start_idx + 1 : abc_end_idx])
        else:
            abc_ids = []
        frames = len(codec_ids) / yue2.FRAMES_PER_SECOND
        metadata = {
            "yue2_chunks": chunks,
            "yue2_frames": frames,
            "yue2_abc_ids": abc_ids,
        }
        conditioning = [[conditioning_tensor, metadata]]
        return (conditioning, frames)


class YuE2LLMInferenceNode:
    DESCRIPTION = "Node for LLM inference YuE2's LLM model. Note: The default parameters do not work well for ABC generation. I recommend using the official node parameters as a reference and sampling in stages.\nWARNING: In development, will likely be changed."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("STRING",)

    _DEFAULT_YAML = """
# For reference, YuE2 prompt prefixes for CoT modes:
#   off   : <|yue2_eod|>Generate music with codec tokens from the given conditions.
#   melody: <|yue2_eod|>Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.
#   full  : <|yue2_eod|>Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.

# Does logit sampling/manipulation in float64. Not much of a performance cost generally.
sampling_dtype: float64

# One of: new, full, from_abc, from_music
output_mode: new

sampling_parameters:
    seed: 123
    # Does logit sampling/manipulation in float64 by default. I don't recommend
    # going below float32 here and float64 doesn't seem to affect performance noticeably.
    sampling_dtype: float64
    temperature: 1.0
    cfg_scale: 1.0
    # If set, can use blend modes from ComfyUI-bleh
    cfg_blend_mode: null
    # One of diff, cfg or null. diff blends the CFG difference into cond,
    # cfg does the normal CFG blend and then blends the result with cond.
    cfg_blend_strategy: null
    # Can be null, probs or logprobs.
    cfg_guidance_space: null
    # When >0, will only apply CFG to those top K items.
    cfg_plausibility_mask_top_k: 0
    # When >= 0, will only that are at least >= that percentage of
    # the highest logit. Be more generous with this than min_p for
    # sampling, something like 0.125 is probably a good place to start.
    # Probably better than the top K approach.
    cfg_plausibility_mask_min_p: -1.0
    # When non-zero will rescale the CFG result variance to match cond by default.
    cfg_variance_rescaling_strength: 0.0
    # When non-zero will rescale the CFG result mean to match cond by default.
    cfg_mean_rescaling_strength: 0.0
    # Rescaling to cond is the safe option. Other options:
    #   uncond, diff, neg_diff, min, max
    cfg_variance_rescaling_target: cond
    cfg_mean_rescaling_target: cond
    # Only applies when using plausibility masking. When set, will reapply the mask
    # again at the very end, causing stuff like variance rescaling to only apply
    # to the masked items.
    cfg_final_remask: true

    top_k: 100
    top_p: 0.95
    min_p: 0.0
    # 0 is disabled. Values above 1.0 penality, non-zero values below it
    # encourage repetition which is probably not what you want.
    repetition_penalty: 1.1
    # Windows count from the beginning if positive, or the end if negative.
    # I.E. -128 means consider at most the last 128 tokens.
    repetition_penalty_window: -500
    # N-grams - sequences of however many tokens.
    no_repeat_ngram_size: 0
    no_repeat_ngram_penalty: -.inf
    no_repeat_ngram_window: -500
    # Starts penalizing when 85% of max tokens is reached
    # not counting the prompt.
    max_tokens_expdecay_factor: 0.85
    # Values above 1.0 penalize.
    max_tokens_expdecay_penalty: 1.01
"""
    _DEFAULT_PROMPT = """<|yue2_eod|>Generate music with codec tokens from the given conditions.
[Tags]
Your prompt
[Lyrics]
Your lyrics
<|abc_start|><|abc_end|><|music_start|>
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "clip": ("CLIP",),
                "minimum_tokens": ("INT", {"default": 128, "min": 1, "max": 32767}),
                "maximum_tokens": ("INT", {"default": 8192, "min": 1, "max": 32767}),
                "verbose_interval": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "tooltip": "When set to a value over 0, will dump the current LLM output to the console at that token interval.",
                    },
                ),
                "llm_prompt": (
                    "STRING",
                    {
                        "default": cls._DEFAULT_PROMPT,
                        "dynamicPrompts": False,
                        "multiline": True,
                    },
                ),
                "yaml_params": (
                    "STRING",
                    {
                        "dynamicPrompts": False,
                        "multiline": True,
                        "default": cls._DEFAULT_YAML,
                        "tooltip": "Parameters that affect sampling, in YAML format.",
                    },
                ),
            },
            "optional": {
                "llm_prompt_negative": (
                    "STRING",
                    {
                        "dynamicPrompts": False,
                        "multiline": True,
                        "tooltip": "Only has an effect when cfg_scale is set to a value other than 1.0.",
                    },
                ),
                "custom_noise": (
                    ("SONAR_CUSTOM_NOISE,OCS_NOISE"),
                    {
                        "tooltip": "Optional custom noise input for temperature sampling. Can take custom noise inputs from my ComfyUI-Sonar and comfyui_overly_complicated_sampling node packs. Note: You will probably want to use a noise factor around 0.4 to 0.6 and also make sure that the noise is normalized. This will not work well with noise samplers that care about the sigma.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        clip: object,
        minimum_tokens: int,
        maximum_tokens: int,
        verbose_interval: int,
        llm_prompt: str,
        yaml_params: str,
        llm_prompt_negative: str | None = None,
        custom_noise: object | None = None,
    ) -> tuple:
        ensure_blend_modes()
        params = yaml.safe_load(yaml_params)
        if not isinstance(params, dict):
            raise TypeError("yaml_params must be a YAML object")
        output_mode = params.pop("output_mode", "new").strip().lower()
        if output_mode not in {"new", "full", "from_abc", "from_music"}:
            raise ValueError("Bad output mode")
        sampling_params = params.get("sampling_parameters", {})
        minimum_tokens, maximum_tokens = (
            min(minimum_tokens, maximum_tokens),
            max(minimum_tokens, maximum_tokens),
        )
        tokenizer = get_yue2_tokenizer(clip)
        tokens = tokenizer.encode(llm_prompt).ids

        abc_start_idx, abc_end_idx, music_start_idx, music_end_idx = get_token_idxs(
            tokens,
            yue2.ABC_START,
            yue2.ABC_END,
            yue2.MUSIC_START,
            yue2.MUSIC_END,
        )
        if music_end_idx is not None:
            raise ValueError(
                "Prompt already contains a <|music_end|> token - nothing to generate!",
            )
        if music_start_idx is not None:
            if (abc_start_idx is not None and abc_start_idx > music_start_idx) or (
                abc_end_idx is not None and abc_end_idx > music_start_idx
            ):
                raise ValueError("ABC start/end tokens must be before music start!")
            eos_token_id = yue2.MUSIC_END
        else:
            eos_token_id = yue2.ABC_END

        cfg_scale = sampling_params.get("cfg_scale", 1.0)
        if cfg_scale != 1.0:
            if llm_prompt_negative is None:
                raise ValueError(
                    "Must provide negative prompt when cfg_scale is not 1.0",
                )
            tokens_neg = tokenizer.encode(llm_prompt_negative).ids
        else:
            tokens_neg = None
        clip_metadata = {
            "cot": "off",
            "cfg_scale": cfg_scale,
            "max_tokens": maximum_tokens,
            "prefix": [(t, 1.0) for t in tokens],
        }
        clip.load_model(clip_metadata)
        csm = clip.cond_stage_model
        csm.set_clip_options({"execution_device": clip.patcher.load_device})
        llm_sampler = LLMSampling(
            verbose_interval=verbose_interval,
            tokenizer=tokenizer,
            model=csm,
            llm_class=ModelYuE2LLM,
            state_class=partial(
                YuE2LLMSamplingState,
                audio_only=eos_token_id == yue2.MUSIC_END,
                eos_token_id=eos_token_id,
                custom_noise=None if custom_noise is None else custom_noise.clone(),
                **sampling_params,
            ),
        )
        output_tokens = llm_sampler(
            ids=[tokens] if tokens_neg is None else [tokens, tokens_neg],
            min_tokens=minimum_tokens,
            max_tokens=maximum_tokens,
        )[0]
        if output_mode != "new":
            output_tokens = [*tokens, *output_tokens]
            if output_mode != "full":
                start_idx = get_token_idxs(
                    output_tokens,
                    yue2.ABC_START if output_mode == "from_abc" else yue2.MUSIC_START,
                )[0]
                if start_idx is None:
                    errstr = f"Output mode is {output_mode} but start token of that type missing"
                    raise RuntimeError(errstr)
                output_tokens = output_tokens[start_idx + 1 :]

        decoded_outputs = tokenizer.decode(output_tokens)
        return (decoded_outputs,)
