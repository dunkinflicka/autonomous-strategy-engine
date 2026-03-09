"""
Utility probability distributions used across the simulator.
All distributions are parameterised to be easily swapped.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class TruncatedNormal:
    """Normal distribution clipped to [low, high]."""
    mean: float
    std: float
    low: float = -np.inf
    high: float = np.inf

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray | float:
        raw = rng.normal(self.mean, self.std, size=size)
        return np.clip(raw, self.low, self.high)


@dataclass
class PoissonProcess:
    """Poisson arrival process with rate lambda (events per interval)."""
    rate: float  # expected events per interval

    def event_occurred(self, rng: np.random.Generator) -> bool:
        """Return True if at least one event occurs in this interval."""
        return rng.poisson(self.rate) > 0

    def expected_events(self, n_intervals: int) -> float:
        return self.rate * n_intervals


def logistic(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def beta_sample(rng: np.random.Generator, mean: float, variance: float,
                size: int | None = None) -> np.ndarray | float:
    """
    Sample from a Beta distribution parameterised by mean and variance.
    Useful for sampling probabilities (support [0,1]).
    """
    if variance <= 0 or mean <= 0 or mean >= 1:
        raise ValueError(f"Invalid Beta params: mean={mean}, variance={variance}")
    alpha = mean * (mean * (1 - mean) / variance - 1)
    beta  = (1 - mean) * (mean * (1 - mean) / variance - 1)
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"Derived Beta shape params non-positive: a={alpha:.3f}, b={beta:.3f}")
    return rng.beta(alpha, beta, size=size)
