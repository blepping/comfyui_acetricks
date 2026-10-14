from .ace15_nodes import (
    Ace15AudioCodesToLatentNode,
    Ace15CompressDuplicateAudioCodesNode,
    Ace15GetGlobalProjectionNode,
    Ace15LatentToAudioCodesNode,
    Ace15LMHintsToLatentForVisualizationNode,
    EmptyAce15LatentFromConditioningNode,
    ModelPatchAce15Use4dLatentNode,
    RawTextEncodeAce15Node,
    TextEncodeAce15Node,
)
from .audio_nodes import (
    AudioAsLatentNode,
    AudioBlendNode,
    AudioFromBatchNode,
    AudioLevelsNode,
    LatentAsAudioNode,
    MonoToStereoNode,
    SetAudioDtypeNode,
    WaveformNode,
)
from .cond_nodes import (
    EncodeLyricsNode,
    JoinLyricsNode,
    SplitOutLyricsNode,
)
from .latent_nodes import (
    SilentLatentNode,
    SqueezeUnsqueezeLatentDimensionNode,
    VisualizeLatentNode,
)
from .llm_nodes import Ace15LLMInferenceNode
from .misc_nodes import CutAudioCodesNode, MaskNode, TimeOffsetNode
from .yue2_nodes import (
    LoadYuE2TokenizerAudioEncoderNode,
    RawTextEncodeYuE2Node,
    YuE2AudioToCodesNode,
    YuE2LLMInferenceNode,
)

NODE_CLASS_MAPPINGS = {
    "ACETricks Ace15AudioCodesToLatent": Ace15AudioCodesToLatentNode,
    "ACETricks Ace15CompressDuplicateAudioCodes": Ace15CompressDuplicateAudioCodesNode,
    "ACETricks Ace15GetGlobalProjection": Ace15GetGlobalProjectionNode,
    "ACETricks Ace15LatentToAudioCodes": Ace15LatentToAudioCodesNode,
    "ACETricks Ace15LLMInference": Ace15LLMInferenceNode,
    "ACETricks Ace15LMHintsToLatentForVisualization": Ace15LMHintsToLatentForVisualizationNode,
    "ACETricks AudioAsLatent": AudioAsLatentNode,
    "ACETricks AudioBlend": AudioBlendNode,
    "ACETricks AudioFromBatch": AudioFromBatchNode,
    "ACETricks AudioLevels": AudioLevelsNode,
    "ACETricks CondJoinLyrics": JoinLyricsNode,
    "ACETricks CondSplitOutLyrics": SplitOutLyricsNode,
    "ACETricks CutAudioCodes": CutAudioCodesNode,
    "ACETricks EmptyAce15LatentFromConditioning": EmptyAce15LatentFromConditioningNode,
    "ACETricks EncodeLyrics": EncodeLyricsNode,
    "ACETricks LatentAsAudio": LatentAsAudioNode,
    "ACETricks LoadYuE2TokenizerAudioEncoder": LoadYuE2TokenizerAudioEncoderNode,
    "ACETricks Mask": MaskNode,
    "ACETricks ModelPatchAce15Use4dLatent": ModelPatchAce15Use4dLatentNode,
    "ACETricks MonoToStereo": MonoToStereoNode,
    "ACETricks RawTextEncodeAce15": RawTextEncodeAce15Node,
    "ACETricks RawTextEncodeYuE2": RawTextEncodeYuE2Node,
    "ACETricks SetAudioDtype": SetAudioDtypeNode,
    "ACETricks SilentLatent": SilentLatentNode,
    "ACETricks SqueezeUnsqueezeLatentDimension": SqueezeUnsqueezeLatentDimensionNode,
    "ACETricks TextEncodeAce15": TextEncodeAce15Node,
    "ACETricks Time Offset": TimeOffsetNode,
    "ACETricks VisualizeLatent": VisualizeLatentNode,
    "ACETricks Waveform Image": WaveformNode,
    "ACETricks YuE2AudioToCodes": YuE2AudioToCodesNode,
    "ACETricks YuE2LLMInference": YuE2LLMInferenceNode,
}

__all__ = ("NODE_CLASS_MAPPINGS",)
