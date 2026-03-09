"""
Centralised random state management.
Ensures reproducibility across all stochastic components.
"""
from __future__ import annotations

import numpy as np


class RandomManager:
    """
    Holds a seeded numpy Generator and provides named sub-streams
    via SeedSequence spawning. This guarantees that adding new
    stochastic components doesn't break existing stream ordering.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._seq  = np.random.SeedSequence(seed)
        self._streams: dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        """Return (or create) a reproducible Generator for a named stream."""
        if name not in self._streams:
            child_seq = self._seq.spawn(1)[0]
            # Deterministic child seed from name hash
            named_seq = np.random.SeedSequence(
                hash(name) ^ int(self._seq.entropy)  # type: ignore[arg-type]
            )
            self._streams[name] = np.random.default_rng(named_seq)
        return self._streams[name]

    def reset(self, seed: int | None = None) -> None:
        """Re-initialise all streams (call before each simulation)."""
        if seed is not None:
            self._seed = seed
        self._seq = np.random.SeedSequence(self._seed)
        self._streams.clear()

    @property
    def seed(self) -> int:
        return self._seed


# Module-level default manager (override seed via reset())
_default_manager = RandomManager(seed=42)


def get_rng(name: str = "default") -> np.random.Generator:
    return _default_manager.stream(name)


def set_global_seed(seed: int) -> None:
    _default_manager.reset(seed)
