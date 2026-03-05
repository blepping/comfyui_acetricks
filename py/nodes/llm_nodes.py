from functools import partial

import yaml

from ..llm import Ace15LLMSampling, ACE15LLMSamplingState


class Ace15LLMInferenceNode:
    DESCRIPTION = "Node for LLM inference with ACE 1.5's Qwen models. WARNING: In development, will likely be changed."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)

    _DEFAULT_YAML = """
# Does logit sampling/manipulation in float64. Not much of a performance cost generally.
sampling_dtype: float64
# null - no constraint, true - only audio codes tokens, false - only non-audio codes tokens
ace15_audio_only: null

sampling_parameters:
    seed: 123
    # Does logit sampling/manipulation in float64 by default. I don't recommend
    # going below float32 here and float64 doesn't seem to affect performance noticeably.
    sampling_dtype: float64
    temperature: 0.85
    cfg_scale: 2.0
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

    top_k: 0
    top_p: 0.0
    min_p: 0.075
    # 0 is disabled. Values above 1.0 penality, non-zero values below it
    # encourage repetition which is probably not what you want.
    repetition_penalty: 0.0
    # Windows count from the beginning if positive, or the end if negative.
    # I.E. -128 means consider at most the last 128 tokens.
    repetetion_penalty_window: null
    # N-grams - sequences of however many tokens.
    no_repeat_ngram_size: 0
    no_repeat_ngram_penalty: -.inf
    no_repeat_ngram_window: null
    # Starts penalizing when 85% of max tokens is reached
    # not counting the prompt.
    max_tokens_expdecay_factor: 0.85
    # Values above 1.0 penalize.
    max_tokens_expdecay_penalty: 1.01
"""
    _DEFAULT_PROMPT = """<|im_start|>system
    # Instruction
    Generate audio semantic tokens based on the given conditions:

    <|im_end|>
    <|im_start|>user
    # Caption
    A beautiful instrumental euphoric trance piece featuring realistic piano arpeggios.
    Varied and creative. High-fidelity.

    # Lyric
    [Instrumental]

    <|im_end|>
    <|im_start|>assistant
    <think>
    bpm: 80
    duration: 160
    keyscale: B minor
    timesignature: 4
    language: unknown
    </think>

    <|im_end|>
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "clip": ("CLIP",),
                "minimum_tokens": ("INT", {"default": 1, "min": 1, "max": 32767}),
                "maximum_tokens": ("INT", {"default": 2048, "min": 1, "max": 32767}),
                "add_ace15_tokens": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When enabled, will add ACE 1.5's audio code tokens to the vocabulary.",
                    },
                ),
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
                    ("SONAR_CUSTOM_NOISE,OCS_CUSTOM_NOISE"),
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
        add_ace15_tokens: bool,
        verbose_interval: int,
        llm_prompt: str,
        yaml_params: str,
        llm_prompt_negative: str | None = None,
        custom_noise: object | None = None,
    ) -> tuple:
        params = yaml.safe_load(yaml_params)
        if not isinstance(params, dict):
            raise TypeError("yaml_params must be a YAML object")
        sampling_params = params.get("sampling_parameters", {})
        minimum_tokens, maximum_tokens = (
            min(minimum_tokens, maximum_tokens),
            max(minimum_tokens, maximum_tokens),
        )
        clip = clip.clone()
        clip_tokenizer = getattr(clip.tokenizer, clip.tokenizer.clip).tokenizer
        tokenizer_path = getattr(clip_tokenizer, "init_kwargs", {}).get("name_or_path")
        if tokenizer_path is None:
            raise ValueError("Missing tokenizer path")
        tokenizer = clip_tokenizer.__class__.from_pretrained(
            tokenizer_path,
            **clip_tokenizer.init_kwargs,
        )
        if add_ace15_tokens:
            tokenizer.add_tokens([f"<|audio_code_{i}|>" for i in range(64000)])
        tokens = tokenizer.encode(llm_prompt)
        cfg_scale = sampling_params.get("cfg_scale", 1.0)
        if cfg_scale != 1.0:
            if llm_prompt_negative is None:
                raise ValueError(
                    "Must provide negative prompt when cfg_scale is not 1.0"
                )
            tokens_neg = tokenizer.encode(llm_prompt_negative)
        else:
            tokens_neg = None
        clip_metadata = {
            "lm_prompt": list((t, 1.0) for t in tokens),
            "lm_metadata": {"min_tokens": minimum_tokens},
        }
        csm = clip.cond_stage_model
        csm.reset_clip_options()
        clip.load_model(clip_metadata)
        csm.set_clip_options({"execution_device": clip.patcher.load_device})
        model_key = params.pop("model_key", None)
        if model_key is None:
            model_key = getattr(
                clip.cond_stage_model,
                "lm_model",
                getattr(clip.cond_stage_model, "qwen3_06b", None),
            )
        if model_key is None:
            raise ValueError("Missing model key")
        model = getattr(clip.cond_stage_model, model_key)
        eos_token_id = params.pop("eos_token_id", None)
        if not isinstance(eos_token_id, int):
            eos_token_id = getattr(model, "special_tokens", {}).get("eos")
        if not isinstance(eos_token_id, int):
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError(
                "Couldn't determine EOS token id. You may be using an unsupported model.",
            )
        llm_sampler = Ace15LLMSampling(
            verbose_interval=verbose_interval,
            tokenizer=tokenizer,
            model=model,
            state_class=partial(
                ACE15LLMSamplingState,
                ace15_audio_only=params.get("ace15_audio_only"),
                eos_token_id=eos_token_id,
                custom_noise=custom_noise.clone() if custom_noise is not None else None,
                **sampling_params,
            ),
        )
        output_tokens = llm_sampler(
            ids=[tokens] if tokens_neg is None else [tokens, tokens_neg],
            min_tokens=minimum_tokens,
            max_tokens=maximum_tokens,
        )
        decoded_outputs = [tokenizer.decode(out_tids) for out_tids in output_tokens]
        return (decoded_outputs,)
