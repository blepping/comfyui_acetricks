import sys

import nodes
import torch

BLEND_MODES: dict | None = None


def ensure_blend_modes():
    global BLEND_MODES  # noqa: PLW0603
    if BLEND_MODES is None:
        bi = sys.modules.get("_blepping_integrations", {}) or getattr(
            nodes,
            "_blepping_integrations",
            {},
        )
        bleh = bi.get("bleh")
        if bleh is not None:
            BLEND_MODES = bleh.py.latent_utils.BLENDING_MODES
        else:
            BLEND_MODES = {
                "lerp": torch.lerp,
                "a_only": lambda a, _b, _t: a,
                "b_only": lambda _a, b, _t: b,
                "subtract_b": lambda a, b, t: a - b * t,
                "inject": lambda a, b, t: (b * t).add_(a),
            }


__all__ = (
    "BLEND_MODES",
    "ensure_blend_modes",
)
