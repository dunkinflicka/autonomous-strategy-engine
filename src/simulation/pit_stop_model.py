"""
Pit Stop Model
==============
Models the time cost of a pit stop including:
  - stationary time (tyre change)
  - pit lane entry/exit delta
  - unsafe release probability
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class PitStopParams:
    stationary_time_mean_s: float  = 2.40
    stationary_time_std_s: float   = 0.18
    pit_lane_delta_mean_s: float   = 20.5
    pit_lane_delta_std_s: float    = 0.5
    unsafe_release_probability: float = 0.005
    unsafe_release_penalty_s: float   = 5.0  # time lost to unsafe release penalty


class PitStopModel:
    """
    Computes total time cost for a pit stop event.

    The pit lane delta represents the extra time spent driving through the
    pit lane vs the circuit (entry + exit + speed limiter).
    """

    def __init__(self, params: PitStopParams) -> None:
        self.params = params

    def sample_stop_time(
        self,
        rng: np.random.Generator,
    ) -> tuple[float, bool]:
        """
        Returns (total_time_delta_s, unsafe_release).

        total_time_delta_s = pit_lane_delta + stationary_time
        This replaces the lap time for the pit lap portion.
        """
        stationary = max(
            1.8,
            rng.normal(self.params.stationary_time_mean_s,
                       self.params.stationary_time_std_s)
        )
        pit_lane_delta = max(
            18.0,
            rng.normal(self.params.pit_lane_delta_mean_s,
                       self.params.pit_lane_delta_std_s)
        )

        unsafe_release = rng.random() < self.params.unsafe_release_probability
        penalty = self.params.unsafe_release_penalty_s if unsafe_release else 0.0

        total_delta = pit_lane_delta + stationary + penalty
        return total_delta, unsafe_release

    @property
    def expected_delta(self) -> float:
        """Expected total pit stop time delta (deterministic estimate)."""
        return self.params.pit_lane_delta_mean_s + self.params.stationary_time_mean_s
