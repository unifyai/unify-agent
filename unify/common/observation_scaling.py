"""Model-aware observation scaling for images shown to the LLM.

Images the sandbox displays or passes to ``query_llm`` are downscaled to the
observation space the configured model accepts. Policy is resolved from the
model so scaling stays aligned with provider limits.

The policy constants live in the sibling JSON file
``observation_scaling_policy.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as _PILImage

_POLICY_PATH = Path(__file__).with_name("observation_scaling_policy.json")

_policy_cache: dict | None = None


def _load_policy() -> dict:
    global _policy_cache
    if _policy_cache is None:
        with open(_POLICY_PATH) as f:
            _policy_cache = json.load(f)
    return _policy_cache


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AspectTarget:
    width: int
    height: int


@dataclass(frozen=True)
class ObservationScalingPolicy:
    """Resolved scaling policy for a given model."""

    aspect_targets: tuple[AspectTarget, ...]
    max_edge: int
    model: str
    provider: str


@dataclass(frozen=True)
class NativeObservationScale:
    """Mapping between display pixels and observation pixels."""

    display_width: int
    display_height: int
    observation_width: int
    observation_height: int
    model: str
    provider: str


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------


def parse_model_provider(model: str) -> str:
    """Parse ``name@provider`` or infer provider from model name."""
    trimmed = model.strip().lower()
    at = trimmed.rfind("@")
    if at >= 0:
        return trimmed[at + 1 :]
    if "claude" in trimmed:
        return "anthropic"
    if "gpt" in trimmed:
        return "openai"
    if "gemini" in trimmed:
        return "google"
    if "minimax" in trimmed:
        return "minimax"
    return "default"


def _resolve_max_edge(model: str, provider: str) -> int:
    policy = _load_policy()
    normalized = model.strip().lower()

    model_overrides: dict[str, int] = policy["modelObservationMaxEdge"]
    if normalized in model_overrides:
        return model_overrides[normalized]

    env_override = os.environ.get("UNIFY_OBSERVATION_MAX_EDGE", "").strip()
    if env_override:
        try:
            parsed = int(env_override)
            if parsed > 0:
                return parsed
        except ValueError:
            pass

    provider_edges: dict[str, int] = policy["providerMaxEdge"]
    return provider_edges.get(provider, provider_edges["default"])


def resolve_observation_scaling_policy(model: str) -> ObservationScalingPolicy:
    """Build the full scaling policy for a model identifier."""
    policy = _load_policy()
    provider = parse_model_provider(model)
    targets = tuple(
        AspectTarget(width=t["width"], height=t["height"])
        for t in policy["defaultAspectTargets"]
    )
    return ObservationScalingPolicy(
        aspect_targets=targets,
        max_edge=_resolve_max_edge(model, provider),
        model=model.strip(),
        provider=provider,
    )


# ---------------------------------------------------------------------------
# Observation model resolution (picks the model driving coordinate space)
# ---------------------------------------------------------------------------


def resolve_observation_model() -> str:
    """The model whose vision limits define the observation space: the run's ``UNIFY_MODEL``."""
    from unify.settings import SETTINGS

    return SETTINGS.UNIFY_MODEL


# ---------------------------------------------------------------------------
# Scaling math
# ---------------------------------------------------------------------------


def pick_aspect_target(
    display_width: int,
    display_height: int,
    targets: tuple[AspectTarget, ...],
) -> AspectTarget | None:
    """Pick the closest aspect-ratio target that scales down from display size."""
    ratio = display_width / display_height
    for target in targets:
        if abs(target.width / target.height - ratio) < 0.02:
            if target.width < display_width:
                return target
    return None


def _cap_dimensions(width: int, height: int, max_edge: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= max_edge:
        return width, height
    factor = max_edge / longest
    return round(width * factor), round(height * factor)


def compute_native_observation_scale(
    display_width: int,
    display_height: int,
    policy: ObservationScalingPolicy,
) -> NativeObservationScale:
    """Compute the observation image dimensions for a given display size."""
    aspect = pick_aspect_target(display_width, display_height, policy.aspect_targets)

    if aspect:
        obs_w, obs_h = aspect.width, aspect.height
    else:
        longest = max(display_width, display_height)
        if longest > policy.max_edge:
            factor = policy.max_edge / longest
            obs_w = round(display_width * factor)
            obs_h = round(display_height * factor)
        else:
            obs_w, obs_h = display_width, display_height

    obs_w, obs_h = _cap_dimensions(obs_w, obs_h, policy.max_edge)

    return NativeObservationScale(
        display_width=display_width,
        display_height=display_height,
        observation_width=obs_w,
        observation_height=obs_h,
        model=policy.model,
        provider=policy.provider,
    )


# ---------------------------------------------------------------------------
# Image fitting (used by capture.py display path)
# ---------------------------------------------------------------------------


def fit_image_to_observation_space(
    img: "_PILImage.Image",
    model: str | None = None,
) -> "_PILImage.Image":
    """Resize a PIL Image to the observation space for the given model.

    Idempotent: returns the image unchanged if it already fits within the
    observation dimensions.
    """
    from PIL import Image as _Image

    if model is None:
        model = resolve_observation_model()

    policy = resolve_observation_scaling_policy(model)
    w, h = img.size
    scale = compute_native_observation_scale(w, h, policy)

    if scale.observation_width == w and scale.observation_height == h:
        return img

    return img.resize(
        (scale.observation_width, scale.observation_height),
        _Image.LANCZOS,
    )
