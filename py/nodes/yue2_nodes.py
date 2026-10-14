from functools import partial

import folder_paths
import torch
import yaml
from comfy import model_management
from comfy_execution.graph import ExecutionBlocker
from tokenizers import AddedToken

from .. import external
from ..llm import LLMSampling, get_token_idxs
from ..llm_yue2 import (
    ModelYuE2LLM,
    TokenizerTopKGuideConfig,
    YuE2LLMSamplingState,
    yue2,
)
from ..yue2_support import MERT2TokenizerAudioEncoderModel, TokenizerTopK


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

# Only used when the tokenizer_topk input is connected.
tokenizer_guide_parameters:
    # 25 frames per second. Negative values for start/end mean no restriction.
    start_frame: 0
    end_frame: -1
    frame_offset: 0
    # When disabled, bans the MUSIC_END token there's a guide match.
    allow_end_token: false
    # Linear interpolation by default. You can also try setting this to inject to
    # simply add the guide logits.
    blend_mode: lerp
    # Blend scale for the guide logits.
    scale: 0.5
    # Either null or one of the following:
    #    scale_up: The scale gets multiplied by the progress between start_frame and end_frame.
    #  scale_down: Same as scale_up, but with the scale inverted.
    #         sin: Ramps up such that scale reaches the target at 50% of progress (then it descends).
    #     inv_sin: Like sin, except the scale multiplier starts at 1 and reaches 0 at 50% of progress.
    schedule: null
    # Only has an effect when schedule is set. Exponent for progress, applied after the schedule type.
    schedule_power: 1.0
    # When set, clamps the minimum scale to this value. Only useful with scheduling.
    scale_min: null
    # When set, clamps the maximum scale to this value. Only useful with scheduling.
    scale_max: null
    # Allows setting a start/end slice for the guide top-K. I.E. you can save the top 200 but
    # only use the top 20 items by leaving start at null (or 0) and setting k_end to 20.
    k_start: null
    k_end: null
    # If k drops below 1 (based on schedule, etc), the guide will be disabled automatically.
    k_min: 0
    # Negative values mean no restriction.
    k_max: -1
    # Same options as the schedule parameter below. Schedule scaling applies to the spread
    # between k_start and k_end.
    k_schedule: null
    k_schedule_power: 1.0
    # When non-zero, will align the top guide logit with the YuE2 LLM's logits.
    # At ratio 1.0, the top guide logit will be equal to the top LLM logit.
    align_topk_blend: 0.0
    # Scale factor on the top LLM logit. Only has an effect when align_topk_blend isn't 0.
    align_topk_scale: 1.0
    # When false, only allows scaling down the guide logits.
    align_topk_allow_up: true
    # When false, only allows scaling up the guide logits.
    align_topk_allow_down: true
    # When true, calculates the top logit across the whole batch (cond and uncond).
    align_topk_full_batch: false
    # When enabled will run the logits through log softmax.
    convert_scores: true
    # When enabled, will always run the scores through log softmax.
    always_convert_scores: false
    # If false, will try to guide uncond instead (requires CFG).
    guide_cond: true
    # Position to insert the logits processor. The CFG processor will be at position 0, so
    # you may want to set this to 1 to insert it after.
    processor_index: 0

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
                "tokenizer_topk": (
                    "YUE2_TOKENIZER_TOPK",
                    {
                        "tooltip": "If supplied, will guide sampling using tokenizer logits.",
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
        tokenizer_topk: tuple[TokenizerTopK, ...] | None = None,
    ) -> tuple:
        external.ensure_blend_modes()
        params = yaml.safe_load(yaml_params)
        if not isinstance(params, dict):
            raise TypeError("yaml_params must be a YAML object")
        output_mode = params.pop("output_mode", "new").strip().lower()
        if output_mode not in {"new", "full", "from_abc", "from_music"}:
            raise ValueError("Bad output mode")
        sampling_params = params.get("sampling_parameters", {})
        guide_config = (
            None
            if not tokenizer_topk
            else params.pop("tokenizer_guide_parameters", None)
        )
        if guide_config is not None and not isinstance(guide_config, dict):
            raise TypeError(
                "tokenizer_guide_parameters must be a YAML object or null if supplied",
            )
        tokenizer_guide = (
            None
            if guide_config is None
            else TokenizerTopKGuideConfig.build(**guide_config)._replace(
                topk_chunks=tokenizer_topk,
            )
        )
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

        hidden_cfg_scale = sampling_params.pop("hidden_cfg_scale", 1.0)
        hidden_cfg_blend_mode = (
            sampling_params.pop("hidden_cfg_blend_mode", None)
            if hidden_cfg_scale != 1
            else None
        )
        hidden_cfg_blend_function = (
            torch.lerp
            if hidden_cfg_blend_mode is None
            else external.BLEND_MODES[hidden_cfg_blend_mode]
        )
        hidden_cfg_cond = bool(sampling_params.pop("hidden_cfg_cond", True))
        cfg_scale = sampling_params.get("cfg_scale", 1.0)
        if not hidden_cfg_cond and cfg_scale == 1:
            raise ValueError(
                "Disabling hidden_cfg_cond with cfg_scale==1 doesn't make sense",
            )
        using_cfg = cfg_scale != 1 or hidden_cfg_scale != 1
        if using_cfg:
            if llm_prompt_negative is None:
                raise ValueError(
                    "Must provide negative prompt when cfg_scale is not 1.0",
                )
            tokens_neg = tokenizer.encode(llm_prompt_negative).ids
        else:
            tokens_neg = None
        clip_metadata = {
            "cot": "off",
            "cfg_scale": 2.0 if using_cfg else 1.0,
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
            llm_class=partial(
                ModelYuE2LLM,
                logits_cfg_scale=cfg_scale,
                hidden_cfg_scale=hidden_cfg_scale,
                hidden_cfg_blend_function=hidden_cfg_blend_function,
                hidden_cfg_cond=hidden_cfg_cond,
            ),
            state_class=partial(
                YuE2LLMSamplingState,
                audio_only=eos_token_id == yue2.MUSIC_END,
                tokenizer_guide=tokenizer_guide,
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


class LoadYuE2TokenizerAudioEncoderNode:
    DESCRIPTION = "Loader for MERT2 FullSong and MotherSuperior's YuE2 audio tokenizer."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("AUDIO_ENCODER",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        encoders_list = folder_paths.get_filename_list("audio_encoders")
        return {
            "required": {
                "mert2_fullsong": (
                    encoders_list,
                    {
                        "default": "mert2-fullsong.safetensors",
                    },
                ),
                "tokenizer_head": (
                    encoders_list,
                    {
                        "default": "tokenizer_head_joint_v9.safetensors",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        mert2_fullsong: str,
        tokenizer_head: str,
    ) -> tuple[MERT2TokenizerAudioEncoderModel]:
        mert2_fullsong, tokenizer_head = (
            folder_paths.get_full_path_or_raise("audio_encoders", fn)
            for fn in (mert2_fullsong, tokenizer_head)
        )
        audio_encoder = MERT2TokenizerAudioEncoderModel.load_audioencoder(
            mert2_path=mert2_fullsong,
            tokenizer_head_path=tokenizer_head,
        )
        return (audio_encoder,)


class YuE2AudioToCodesNode:
    DESCRIPTION = "Converts audio to YuE2 codes. Load the audio encoder with the LoadYuE2TokenizerAudioEncoder node."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("STRING", "YUE2_TOKENIZER_TOPK")
    OUTPUT_IS_LIST = (True, True)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "audio_encoder": ("AUDIO_ENCODER",),
                "audio": ("AUDIO",),
                "tokenizer_topk": (
                    "INT",
                    {
                        "min": 0,
                        "max": 32768,
                        "tooltip": "When non-zero, the node will output the top-K logits for each frame.",
                    },
                ),
                "pad_seconds": (
                    "FLOAT",
                    {
                        "min": 0.0,
                        "max": 9999.0,
                        "default": 1.0,
                        "tooltip": "When enabled, left-pads the audio with silence. The padded section is stripped from the output.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        audio_encoder: object,
        audio: dict,
        tokenizer_topk: int,
        pad_seconds: float,
    ) -> tuple[
        list[str],
        list[ExecutionBlocker] | list[tuple[TokenizerTopK, ...]],
    ]:
        if not isinstance(audio_encoder, MERT2TokenizerAudioEncoderModel):
            raise ValueError(  # noqa: TRY004
                "Unexpected audio encoder. Use the LoadYuE2TokenizerAudioEncoder node to load the audio encoder.",
            )
        codes, topk_list = audio_encoder.audio_to_yue2_codes(
            waveform=audio["waveform"],
            sample_rate=audio["sample_rate"],
            save_topk=tokenizer_topk,
        )
        codes = codes.to(device="cpu", dtype=torch.int32)
        codes_result = [
            "".join(f"<|audio_code_{code}|>" for code in batch_item)
            for batch_item in codes.tolist()
        ]
        batch_size = len(codes_result)
        if not topk_list:
            no_topk_eb = ExecutionBlocker("Returning top-K items was disabled")
            return (codes_result, [no_topk_eb] * batch_size)
        cpu_device = torch.device("cpu")
        topk_result = [
            tuple(
                tki.get_batch(bidx).to(device=cpu_device, dtype=torch.float32)
                for tki in topk_list
            )
            for bidx in range(batch_size)
        ]
        return (codes_result, topk_result)
