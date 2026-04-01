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
from .misc_nodes import MaskNode, TimeOffsetNode

NODE_CLASS_MAPPINGS = {
    "ACETricks SilentLatent": SilentLatentNode,
    "ACETricks VisualizeLatent": VisualizeLatentNode,
    "ACETricks CondSplitOutLyrics": SplitOutLyricsNode,
    "ACETricks CondJoinLyrics": JoinLyricsNode,
    "ACETricks EncodeLyrics": EncodeLyricsNode,
    "ACETricks SetAudioDtype": SetAudioDtypeNode,
    "ACETricks AudioLevels": AudioLevelsNode,
    "ACETricks AudioAsLatent": AudioAsLatentNode,
    "ACETricks LatentAsAudio": LatentAsAudioNode,
    "ACETricks MonoToStereo": MonoToStereoNode,
    "ACETricks AudioBlend": AudioBlendNode,
    "ACETricks AudioFromBatch": AudioFromBatchNode,
    "ACETricks Mask": MaskNode,
    "ACETricks Time Offset": TimeOffsetNode,
    "ACETricks Waveform Image": WaveformNode,
    "ACETricks SqueezeUnsqueezeLatentDimension": SqueezeUnsqueezeLatentDimensionNode,
    "ACETricks ModelPatchAce15Use4dLatent": ModelPatchAce15Use4dLatentNode,
    "ACETricks EmptyAce15LatentFromConditioning": EmptyAce15LatentFromConditioningNode,
    "ACETricks Ace15CompressDuplicateAudioCodes": Ace15CompressDuplicateAudioCodesNode,
    "ACETricks TextEncodeAce15": TextEncodeAce15Node,
    "ACETricks RawTextEncodeAce15": RawTextEncodeAce15Node,
    "ACETricks Ace15LatentToAudioCodes": Ace15LatentToAudioCodesNode,
    "ACETricks Ace15AudioCodesToLatent": Ace15AudioCodesToLatentNode,
    "ACETricks Ace15LLMInference": Ace15LLMInferenceNode,
    "ACETricks Ace15GetGlobalProjection": Ace15GetGlobalProjectionNode,
    "ACETricks Ace15LMHintsToLatentForVisualization": Ace15LMHintsToLatentForVisualizationNode,
}

__all__ = ("NODE_CLASS_MAPPINGS",)
