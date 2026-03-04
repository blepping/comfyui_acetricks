import itertools

import torch
import yaml
from comfy import model_management, model_patcher, patcher_extension, samplers

from ..ace_utils import LATENT_TIME_MULTIPLIER_15


class ModelPatchAce15Use4dLatentNode:
    DESCRIPTION = "Patches an ACE 1.5 model (or theoretically, any model that uses 3D latents) to handle 4D latent inputs/outputs. The point of doing this is because many existing nodes are not able to handle 3D latents. You will also likely need to use the SqueezeUnsqueezeLatentDimension node on both sides of sampling (unsqueeze before, squeeze after). NOTE: This node is a hacky abomination. There are probably many cases where it where it won't function correctly, it's also quite likely to break if ComfyUI changes stuff. There really just isn't a reliable way to do what this is trying to do."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("MODEL",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "preserve_patch": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Preserves/delegates to an existing patch of the same type (if it exists) instead of replacing it.",
                    },
                ),
                "dimension": (
                    "INT",
                    {
                        "default": 2,
                        "min": -9999,
                        "max": 9999,
                        "tooltip": "Negative dimensions count from the end.",
                    },
                ),
                "expected_dimensions": (
                    "INT",
                    {
                        "default": 4,
                        "min": 2,
                        "max": 9999,
                        "tooltip": "Number of dimensions to expect, generally this would be original model dimensions + 1.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        model: model_patcher.ModelPatcher,
        preserve_patch: bool = True,
        dimension: int = 2,
        expected_dimensions: int = 4,
    ) -> tuple:
        model = model.clone()
        old_wrapper = (
            model.model_options.get("sampler_calc_cond_batch_function")
            if preserve_patch
            else None
        )

        def calc_cond_batch_patch(args: dict) -> list[torch.Tensor | None]:
            model = args["model"]
            x: torch.Tensor = args["input"]
            skip = (
                not isinstance(x, torch.Tensor)
                or x.ndim != expected_dimensions
                or x.shape[dimension] != 1
            )
            if not skip:
                x = x.squeeze(dimension)
            if old_wrapper is None:
                result = samplers.calc_cond_batch(
                    model,
                    args["conds"],
                    x,
                    args["sigma"],
                    args["model_options"],
                )
            else:
                result = old_wrapper(args | {"input": x})
            if skip:
                return result
            return [
                t if not isinstance(t, torch.Tensor) else t.unsqueeze(dimension)
                for t in result
            ]

        def outer_sample_wrapper(executor, *args, **kwargs):
            noise = args[0]
            co = getattr(executor, "class_obj", None)
            inner_sample = getattr(co, "inner_sample", None)
            if (
                inner_sample is None
                or noise.ndim != expected_dimensions
                or noise.shape[dimension] != 1
            ):
                return executor(*args, **kwargs)

            def the_horror_pt1(*args, **kwargs):
                # AKA inner sample wrapper, which can't currently be hooked.
                inner_noise = args[0]
                inner_model = getattr(co, "inner_model", None)
                extra_conds = getattr(inner_model, "extra_conds", None)
                if (
                    extra_conds is None
                    or inner_noise.ndim != expected_dimensions
                    or inner_noise.shape[dimension] != 1
                ):
                    return inner_sample(*args, **kwargs)

                def the_horror_pt2(*args, **kwargs):
                    ec_noise = kwargs.get("noise", None)
                    if (
                        isinstance(ec_noise, torch.Tensor)
                        and ec_noise.ndim == expected_dimensions
                        and ec_noise.shape[dimension] == 1
                    ):
                        kwargs["noise"] = ec_noise.squeeze(dimension)
                    return extra_conds(*args, **kwargs)

                inner_model.extra_conds = the_horror_pt2
                try:
                    return inner_sample(*args, **kwargs)
                finally:
                    if getattr(inner_model, "extra_conds", None) is the_horror_pt2:
                        inner_model.extra_conds = extra_conds
                    else:
                        raise RuntimeError(
                            "ACETricks: Use4dLatent hack: Can't replace original extra_conds wrapper. You may need to restart ComfyUI.",
                        )

            # Nice clean implementation, definitely isn't ever going to cause issues.
            try:
                co.inner_sample = the_horror_pt1
                return executor(*args, **kwargs)
            finally:
                if co.inner_sample is the_horror_pt1:
                    co.inner_sample = inner_sample
                else:
                    raise RuntimeError(
                        "ACETricks: Use4dLatent hack: Can't replace original inner_sample wrapper. You may need to restart ComfyUI.",
                    )

        model.set_model_sampler_calc_cond_batch_function(calc_cond_batch_patch)
        model.add_wrapper_with_key(
            patcher_extension.WrappersMP.OUTER_SAMPLE,
            "acetricks_use4dlatent_hack",
            outer_sample_wrapper,
        )
        return (model,)


class EmptyAce15LatentFromConditioningNode:
    DESCRIPTION = "Creates an empty latent from ACE-Steps 1.5 conditioning. Calculates the size from the number of LM audio codes in the conditioning, so it's only useful if you are using the LM feature."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("LATENT",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "minimum_duration": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 9999.0,
                        "tooltip": "Will trigger an error if there aren't enough audio codes to reach the specified duration.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        conditioning: list,
        batch_size: int = 1,
        minimum_duration: float = 0.0,
    ) -> tuple:
        n_codes = 0
        for _, d in conditioning:
            audio_codes = d.get("audio_codes")
            if (
                not isinstance(audio_codes, list)
                or not audio_codes
                or not len(audio_codes[0])
            ):
                continue
            n_codes = max(n_codes, *(len(codes) for codes in audio_codes))
        duration = 0.2 * n_codes  # 5hz
        if duration < minimum_duration:
            errstr = f"{n_codes} 5hz code(s) ({duration:.3f} second(s)) doesn't match the minimum of {minimum_duration:.3f}."
            raise ValueError(errstr)
        device = model_management.intermediate_device()
        temporal_length = int(duration * LATENT_TIME_MULTIPLIER_15)
        latent = torch.zeros(batch_size, 64, temporal_length, device=device)
        return ({"samples": latent, "type": "audio"},)


class Ace15CompressDuplicateAudioCodesNode:
    DESCRIPTION = "Creates an empty latent from ACE-Steps 1.5 conditioning. Calculates the size from the number of LM audio codes in the conditioning, so it's only useful if you are using the LM feature."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("CONDITIONING",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "mode": (
                    ("start", "end", "start_end", "anywhere"),
                    {
                        "default": "end",
                        "tooltip": "Controls whether the node prunes duplicate audio codes from the beginning, end of the list, both head and tail or has no location restriction.",
                    },
                ),
                "repeat_limit": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 9999,
                    },
                ),
                "repeat_replace": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 9999,
                        "tooltip": "When set to -1, will use the repeat_limit value. Otherwise, you can can prune repeated codes by setting this to 0 or possibly even expand them further (sounds like a bad idea).",
                    },
                ),
            },
        }

    @staticmethod
    def simple_rle(items: list | tuple) -> tuple:
        n_items = len(items)
        if not n_items:
            return ()
        result = []
        idx = 0
        while idx < n_items:
            item = items[idx]
            run_counter = 0
            for subitem in items[idx + 1 :]:
                if subitem != item:
                    break
                run_counter += 1
            run_counter += 1
            result.append((item, run_counter))
            idx += run_counter
        return tuple(result)

    @staticmethod
    def simple_unrle(rle_items: list | tuple) -> tuple:
        return tuple(itertools.chain(*((i,) * n for i, n in rle_items)))

    @classmethod
    def apply_limits(
        cls,
        audio_codes: list | tuple,
        *,
        apply_any: bool,
        apply_head: bool,
        apply_tail: bool,
        limit: int = 1,
        replace: int = -1,
    ) -> list | tuple:
        if not (apply_any or apply_head or apply_tail):
            return audio_codes[:]
        ac_rle = cls.simple_rle(audio_codes)
        if not ac_rle:
            return audio_codes[:]
        # print(f"\nAC RLE: {ac_rle}")

        def limit_item(item: int, n_reps: int) -> tuple[int, int]:
            n_reps = (
                min(limit, n_reps)
                if replace < 0
                else (n_reps if n_reps <= limit else replace)
            )
            return (item, n_reps)

        if apply_any:
            ac_rle = tuple(itertools.starmap(limit_item, ac_rle))
        if apply_head:
            ac_rle = (
                limit_item(*ac_rle[0]),
                *ac_rle[1:],
            )
        if apply_tail:
            ac_rle = (
                *ac_rle[:-1],
                limit_item(*ac_rle[-1]),
            )
        return audio_codes.__class__(cls.simple_unrle(ac_rle))

    @classmethod
    def go(
        cls,
        *,
        conditioning: list,
        mode: str,
        repeat_limit: int = 3,
        repeat_replace: int = -1,
    ) -> tuple:
        mode = mode.strip().lower()
        repeat_limit = max(1, repeat_limit)
        apply_any = mode == "anywhere"
        apply_head = mode in {"start", "start_end"}
        apply_tail = mode in {"end", "start_end"}

        result = []

        for t, d in conditioning:
            d = d.copy()
            audio_codes = d.get("audio_codes")
            if (
                isinstance(audio_codes, (list, tuple))
                and audio_codes
                and len(audio_codes[0])
            ):
                d["audio_codes"] = audio_codes.__class__(
                    cls.apply_limits(
                        codes_item,
                        apply_any=apply_any,
                        apply_head=apply_head,
                        apply_tail=apply_tail,
                    )
                    for codes_item in audio_codes
                )
            result.append([t.clone(), d])
        return (result,)


class TextEncodeAce15Node:
    DESCRIPTION = "Text encoder for ACE-Steps 1.5 with extended features."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("CONDITIONING",)

    _DEFAULT_YAML = """# YAML metadata here. Must be a YAML object.
# Most keys support a _negative suffix, i.e. bpm_negative.
# There is no error checking so if you do "bpm: urmom", all I can say is good luck.
bpm: 120
timesignature: 4/4
language: en
keyscale: D major
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "clip": ("CLIP",),
                "duration": ("FLOAT", {"default": 120.0, "min": 0.2, "step": 0.2}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "tags": (
                    "STRING",
                    {"defaultInput": True, "dynamicPrompts": False, "multiline": True},
                ),
                "lyrics": (
                    "STRING",
                    {"defaultInput": True, "dynamicPrompts": False, "multiline": True},
                ),
                "generate_audio_codes": ("BOOLEAN", {"default": True}),
                "cfg_scale": ("FLOAT", {"default": 2.0}),
                "temperature": ("FLOAT", {"default": 0.85}),
                "top_p": ("FLOAT", {"default": 0.9}),
                "top_k": ("INT", {"default": 0, "min": 0}),
                "yaml_metadata": (
                    "STRING",
                    {
                        "dynamicPrompts": False,
                        "multiline": True,
                        "default": cls._DEFAULT_YAML,
                    },
                ),
            },
            "optional": {
                "tags_negative": (
                    "STRING",
                    {"defaultInput": True, "dynamicPrompts": False, "multiline": True},
                ),
                "lyrics_negative": (
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
        duration: float,
        seed: int,
        tags: str,
        lyrics: str,
        generate_audio_codes: bool,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        top_k: int,
        yaml_metadata: str,
        tags_negative: str | None = None,
        lyrics_negative: str | None = None,
    ) -> tuple:
        parsed_metadata = yaml.safe_load(yaml_metadata)
        if parsed_metadata is None:
            parsed_metadata = {}
        elif not isinstance(parsed_metadata, dict):
            raise TypeError("yaml_metadata must be a YAML object")
        tkwargs = {
            "lyrics": lyrics,
            "duration": duration,
            "seed": seed,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "generate_audio_codes": generate_audio_codes,
        }
        if tags_negative is not None:
            tkwargs["caption_negative"] = tags_negative
        if lyrics_negative is not None:
            tkwargs["lyrics_negative"] = lyrics_negative
        tkwargs |= parsed_metadata
        tokens = clip.tokenize(tags, **tkwargs)
        return (clip.encode_from_tokens_scheduled(tokens),)


class RawTextEncodeAce15Node:
    DESCRIPTION = "Advanced text encoder for ACE-Steps 1.5."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("CONDITIONING",)

    _DEFAULT_YAML = """# YAML metadata here. Must be a YAML object.
# There is absolutely no error checking or type checking for these fields.

lm_metadata:
  # 5hz tokens, use desired seconds * 5.

  # The end of text token is banned until min_tokens is reached.
  min_tokens: 600

  # The model is allowed to generate at most max_tokens.
  max_tokens: 1536

  seed: 0
  generate_audio_codes: true
  cfg_scale: 2.5
  temperature: 1.25
  top_p: 0.9
  top_k: 0

# List of strings such as [lyrics_prompt, dit_prompt].
# Will allow encoding tokens with the normal weight parsing, I.E. (blah:1.2).
# This probably won't work well if at all.
allow_token_weights: []

# All prompt keys use the same format.
# This is the format for multiline YAML strings. Line breaks will
# be preserved but leading ident (to the initial level) will be stripped.
# Use key: |- instead of just | to avoid a trailing line break.

# NOTE: If you attach the optional prompt string inputs, the
# respective keys in the YAML here will be overwritten.

# Used to encode the lyrics (for DIT sampling)
lyrics_prompt: |
  # Languages
  en

  # Lyric
  [Intro]

  [Verse]

# Used for sampling. Note that duration is specified with a seconds unit.
# Time signatures ending with /4 should just use the initial value. I.E. 4/4 -> 4
# Results may vary for other time signatures.
dit_prompt: |
  # Instruction
  Generate audio semantic tokens based on the given conditions:

  # Caption
  User caption/tags here.

  # Metas
  - bpm: 60
  - timesignature: 4
  - keyscale: F minor
  - duration: 195 seconds
  <|endoftext|>

# Positive prompt used for generating audio codes.
# Note that duration is a raw integer value.
lm_prompt: |
  <|im_start|>system
  # Instruction
  Generate audio semantic tokens based on the given conditions:

  <|im_end|>
  <|im_start|>user
  # Caption
  User caption/tags here.

  # Lyric
  [Intro]

  [Verse]
  <|im_end|>
  <|im_start|>assistant
  <think>
  bpm: 60
  duration: 195
  keyscale: F minor
  language: en
  timesignature: 4
  </think>

  <|im_end|>

# Note the blank CoT (<think></think> section).
# Caption is not a recommendation, just an example.
lm_prompt_negative: |
  <|im_start|>system
  # Instruction
  Generate audio semantic tokens based on the given conditions:

  <|im_end|>
  <|im_start|>user
  # Caption
  AI slop. Low quality, boring, repetitive.

  # Lyric
  [Intro]

  [Verse]
  <|im_end|>
  <|im_start|>assistant
  <think>
  </think>

  <|im_end|>

# When set, disables generate_audio codes. Can be specified in two formats:
#   A list of integers, I.E. [1, 2, 3]
#   A string with audio code tokens, I.E: <|audio_code_36126|><|audio_code_36123|>
audio_codes: null
"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "clip": ("CLIP",),
                "yaml_text": (
                    "STRING",
                    {
                        "dynamicPrompts": False,
                        "multiline": True,
                        "default": cls._DEFAULT_YAML,
                    },
                ),
            },
            "optional": {
                "lyrics_prompt": (
                    "STRING",
                    {
                        "defaultInput": True,
                        "dynamicPrompts": False,
                        "multiline": True,
                        "tooltip": "Optional input. See example YAML for the correct format. Will overwrite the YAML key when populated.",
                    },
                ),
                "dit_prompt": (
                    "STRING",
                    {
                        "defaultInput": True,
                        "dynamicPrompts": False,
                        "multiline": True,
                        "tooltip": "Optional input. See example YAML for the correct format. Will overwrite the YAML key when populated.",
                    },
                ),
                "lm_prompt": (
                    "STRING",
                    {
                        "defaultInput": True,
                        "dynamicPrompts": False,
                        "multiline": True,
                        "tooltip": "Optional input. See example YAML for the correct format. Will overwrite the YAML key when populated.",
                    },
                ),
                "lm_prompt_negative": (
                    "STRING",
                    {
                        "defaultInput": True,
                        "dynamicPrompts": False,
                        "multiline": True,
                        "tooltip": "Optional input. See example YAML for the correct format. Will overwrite the YAML key when populated.",
                    },
                ),
                "audio_codes": (
                    "STRING",
                    {
                        "defaultInput": True,
                        "dynamicPrompts": False,
                        "multiline": True,
                        "tooltip": "Optional input. See example YAML for the correct format. Will overwrite the YAML key when populated. When attached, must either be a string of comma-separated integer audio code values or tokens as show in the example.",
                    },
                ),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        clip: object,
        yaml_text: str,
        lyrics_prompt: str | None = None,
        dit_prompt: str | None = None,
        lm_prompt: str | None = None,
        lm_prompt_negative: str | None = None,
        audio_codes: str | None = None,
    ) -> tuple:
        parsed_metadata = yaml.safe_load(yaml_text)
        if not isinstance(parsed_metadata, dict):
            raise TypeError("yaml_metadata must be a YAML object")

        lm_metadata = parsed_metadata.get("lm_metadata", {})
        if not isinstance(lm_metadata, dict):
            raise TypeError("lm_metadata key must be a YAML object")

        verbose = parsed_metadata.pop("verbose", False)

        lyrics_prompt, dit_prompt, lm_prompt, lm_prompt_negative = (
            v if v is not None else parsed_metadata.pop(k, None)
            for k, v in (
                ("lyrics_prompt", lyrics_prompt),
                ("dit_prompt", dit_prompt),
                ("lm_prompt", lm_prompt),
                ("lm_prompt_negative", lm_prompt_negative),
            )
        )
        if not all(isinstance(p, str) for p in (dit_prompt, lyrics_prompt)):
            raise ValueError(
                "At the least, dit_prompt and lyrics_prompt keys must be specified and be type string."
            )
        audio_codes = (
            audio_codes
            if audio_codes is not None
            else lm_metadata.pop("audio_codes", None)
        )
        if isinstance(audio_codes, str):
            audio_codes = audio_codes.strip()
        if isinstance(audio_codes, (tuple, list)):
            if not all(isinstance(ac, int) for ac in audio_codes):
                raise TypeError(
                    "When present, audio codes must be a list of integer values."
                )
        elif isinstance(audio_codes, str):
            if audio_codes.startswith("<|audio_code_"):
                audio_codes = [
                    int(ac.rsplit("_", 1)[-1])
                    for ac in audio_codes.split("|")
                    if ac.startswith("audio_code_")
                ]
            else:
                audio_codes = [int(ac) for ac in audio_codes.split(",")]
        elif audio_codes is not None:
            raise TypeError("Bad format for specified audio codes")

        generate_audio_codes = (
            lm_metadata.get("generate_audio_codes", False)
            and audio_codes is None
            and lm_prompt is not None
            and (lm_prompt_negative is not None or lm_metadata.get("cfg", None) == 1.0)
        )
        lm_metadata["generate_audio_codes"] = generate_audio_codes
        allow_token_weights = frozenset(parsed_metadata.pop("allow_token_weights", ()))
        if not generate_audio_codes:
            lm_prompt = lm_prompt_negative = None

        tokenizer_kwargs = {}
        tokenizer_options = getattr(clip, "tokenizer_options", {})
        if tokenizer_options:
            tokenizer_kwargs["tokenizer_options"] = tokenizer_options.copy()

        key_fixup = {"dit_prompt": "qwen3_06b", "lyrics_prompt": "lyrics"}
        prompts = (
            ("lyrics_prompt", lyrics_prompt),
            ("dit_prompt", dit_prompt),
            ("lm_prompt", lm_prompt),
            ("lm_prompt_negative", lm_prompt_negative),
        )
        encode_input = {
            key_fixup.get(k, k): clip.tokenizer.qwen3_06b.tokenize_with_weights(
                v,
                False,
                disable_weights=k not in allow_token_weights,
                **tokenizer_kwargs,
            )
            for k, v in prompts
            if v is not None
        }
        if generate_audio_codes and "lm_prompt_negative" not in encode_input:
            encode_input["lm_prompt_negative"] = encode_input["lm_prompt"]
        encode_input["lm_metadata"] = lm_metadata
        if verbose:
            print(
                f"\n*** ACETricks RawTextEncodeAce15 debug:\n* PROMPTS: {dict(prompts)}\n* ENCODE INPUT: {encode_input}\n* AUDIO_CODES: {audio_codes}\n"
            )
        conditioning = clip.encode_from_tokens_scheduled(encode_input)
        if audio_codes is not None:
            for _, d in conditioning:
                d["audio_codes"] = [list(audio_codes)]
        return (conditioning,)


class Ace15LatentToAudioCodesNode:
    DESCRIPTION = "Extracts audio codes from an ACE-Steps 1.5 latent. This is probably not working correctly at the moment since the extracted codes don't work very well for sampling."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("CONDITIONING_ACE_LYRICS", "LATENT", "STRING")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
            },
            "optional": {
                "include_upsampled_codes": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "There isn't anything that can use this yet.",
                    },
                ),
            },
        }

    @staticmethod
    @torch.no_grad()
    def get_audio_codes(
        model: object,
        latent: torch.Tensor,
        *,
        upsample_codes: bool,
    ) -> tuple[list[int], torch.Tensor | None]:
        audio_codes, indices = model.tokenizer.tokenize(latent.mT)
        print(f"\nGot: {audio_codes.shape}, {indices.shape}")
        # mask = (indices > 0) & (indices < 64000)
        mask = (indices < 64000) & (indices >= 0)
        indices = indices[mask].reshape(latent.shape[0], -1).detach().cpu().tolist()
        if upsample_codes:
            raw_upsampled_codes = model.detokenizer(audio_codes)
            print(f"\nUpsampled: {raw_upsampled_codes.shape}")
            upsampled_codes = raw_upsampled_codes.mT.detach().float().cpu()
        else:
            upsampled_codes = None
        # print(
        #     f"\nGOT:\nCODES={audio_codes}\nUPSAMPLED:{upsampled_codes}\nINDICES: {indices}"
        # )
        return indices, upsampled_codes

    @classmethod
    def go(
        cls,
        *,
        model: object,
        latent: dict,
        include_upsampled_codes: bool = False,
    ) -> tuple[list[dict], dict]:
        samples = latent["samples"]
        if samples.ndim == 4 and samples.shape[-2] == 1:
            samples = samples.squeeze(-2)
        if samples.ndim != 3 or samples.shape[1] != 64:
            raise ValueError(
                "Incorrect latent format, doesn't appear to be an ACE-Steps 1.5 latent."
            )
        model_management.load_model_gpu(model)
        mmodel = model.model
        device = mmodel.device
        dtype = mmodel.get_dtype()
        samples = samples.to(device=device, dtype=dtype)
        ac, uac = cls.get_audio_codes(
            mmodel.diffusion_model,
            samples,
            upsample_codes=include_upsampled_codes,
        )
        result = {"audio_codes": ac}
        if uac is not None:
            result["upsampled_audio_codes"] = uac
            latent = {"samples": uac.clone(), "type": "audio"}
        else:
            latent = {
                "samples": torch.zeros(
                    1,
                    64,
                    1,
                    device="cpu",
                    dtype=torch.float32,
                )
            }
        ac_tokens = "".join(f"<|audio_code_{i}|>" for i in ac)
        return ([result], latent, ac_tokens)
