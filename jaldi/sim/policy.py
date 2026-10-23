"""Policy families. Each maps (TurnView, parameter) -> boolean trigger per partial."""

from __future__ import annotations

import numpy as np

from .simulate import TurnView


def never(v: TurnView, _=None) -> np.ndarray:
    return np.zeros(len(v.t), bool)


def every(v: TurnView, _=None) -> np.ndarray:
    return np.ones(len(v.t), bool)


def pause(v: TurnView, x_ms: float) -> np.ndarray:
    """Speculate once the caller has been silent for at least x ms."""
    return v.silence >= x_ms


def stable(v: TurnView, k: float) -> np.ndarray:
    """Speculate once the transcript has not changed for k consecutive partials."""
    return v.stable >= k


def turn_detector(v: TurnView, tau: float) -> np.ndarray:
    return np.nan_to_num(v.turn_p, nan=-1.0) >= tau


def jaldi(scores: np.ndarray, tau: float) -> np.ndarray:
    return scores >= tau


FAMILIES = {
    "pause": pause,
    "stable": stable,
    "turn_detector": turn_detector,
}
