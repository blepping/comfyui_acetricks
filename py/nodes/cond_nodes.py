SPLIT_KEYS = ("conditioning_lyrics", "lyrics_strength", "audio_codes")


class SplitOutLyricsNode:
    DESCRIPTION = "Allows splitting out lyrics and lyrics strength from ACE-Steps CONDITIONING objects. Note that you will only be able to join it back again if it is the same shape."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING_ACE_LYRICS")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "add_fake_pooled": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def go(cls, *, conditioning, add_fake_pooled) -> tuple:
        tags_result, lyrics_result = [], []
        for cond_t, cond_d in conditioning:
            cond_d = cond_d.copy()
            split_d = {k: cond_d.pop(k) for k in SPLIT_KEYS if k in cond_d}
            if add_fake_pooled and cond_d.get("pooled_output") is None:
                cond_d["pooled_output"] = cond_t.new_zeros(1, 1)
                split_d["pooled_ouput"] = None
            tags_result.append([cond_t.clone(), cond_d])
            lyrics_result.append(split_d)
        return (tags_result, lyrics_result)


class JoinLyricsNode:
    DESCRIPTION = "Allows joining CONDITIONING_ACE_LYRICS back into CONDITIONING. Will overwrite any lyrics that exist. Must be the same shape as the conditioning the lyrics were split from."
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("CONDITIONING",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "conditioning_tags": ("CONDITIONING",),
                "conditioning_lyrics": ("CONDITIONING_ACE_LYRICS",),
                "mode": (("matching", "add_missing"), {"default": "matching"}),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "end_time": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
        }

    @classmethod
    def go(
        cls,
        *,
        conditioning_tags: list,
        conditioning_lyrics: list,
        mode: str,
        start_time: float,
        end_time: float,
    ) -> tuple[list]:
        ct_len, cl_len = len(conditioning_tags), len(conditioning_lyrics)
        if mode == "add_missing":
            if not cl_len:
                raise ValueError("conditioning_lyrics must have at least one item")
            conditioning_lyrics = (
                conditioning_lyrics + [conditioning_lyrics[-1]] * (ct_len - cl_len)
            )[:ct_len]
        elif ct_len != cl_len:
            raise ValueError(
                f"Different lengths for tags {ct_len} vs conditioning lyrics {cl_len}"
            )
        result = [
            [
                cond_t.clone(),
                cond_d
                | {
                    k: v.clone()
                    if isinstance(v, torch.Tensor)
                    else (v.copy() if hasattr(v, "copy") else v)
                    for k, v in cond_l.items()
                    if mode != "add_missing" or k not in cond_d
                }
                if cond_d.get("start_percent", 0.0) >= start_time
                and cond_d.get("end_percent", 1.0) <= end_time
                else cond_d.copy(),
            ]
            for (cond_t, cond_d), cond_l in zip(conditioning_tags, conditioning_lyrics)
        ]
        return (result,)


class EncodeLyricsNode:
    DESCRIPTION = "Encode lyrics for ACE-Steps 1.0"
    FUNCTION = "go"
    CATEGORY = "audio/acetricks"
    RETURN_TYPES = ("CONDITIONING_ACE_LYRICS",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "clip": ("CLIP",),
                "lyrics_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "lyrics": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
        }

    @classmethod
    def go(cls, *, clip, lyrics_strength: float, lyrics: str) -> tuple:
        conditioning = clip.encode_from_tokens_scheduled(
            clip.tokenize("", lyrics=lyrics)
        )
        lyrics_result = [
            {
                "conditioning_lyrics": cond[1]["conditioning_lyrics"],
                "lyrics_strength": lyrics_strength,
            }
            for cond in conditioning
        ]
        return (lyrics_result,)
