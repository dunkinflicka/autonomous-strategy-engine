"""
Tyre Degradation Model
======================
Physics-informed model tracking wear, temperature, and performance penalty.

Wear accumulation:
    w(t+1) = w(t) + k_c * A_track * push_factor * (1 + thermal_penalty)

Performance loss (seconds added to lap time):
    Δt_wear(w) = a*w + b*w²

Cliff behaviour:
    When w exceeds cliff_threshold, wear_rate is multiplied by cliff_multiplier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Optional

from src.utils.distributions import TruncatedNormal


@dataclass
class TyreCompoundParams:
    id: str
    wear_rate_base: float            # k_c
    thermal_degradation_factor: float
    optimal_temp_c: float
    temp_sensitivity: float
    perf_loss_linear: float          # coefficient a
    perf_loss_quadratic: float       # coefficient b
    cliff_threshold: float           # wear fraction at cliff onset
    cliff_multiplier: float
    max_stint_laps: int
    initial_perf_advantage_s: float  # delta vs medium (negative = faster)

    @classmethod
    def from_dict(cls, data: dict) -> "TyreCompoundParams":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TyreState:
    """Mutable state for a single tyre set during a stint."""
    compound: str
    wear: float = 0.0           # fraction [0, 1]; 1 = fully worn
    temperature_c: float = 80.0
    stint_age: int = 0          # laps completed on this set
    is_new: bool = True

    def copy(self) -> "TyreState":
        return TyreState(
            compound=self.compound,
            wear=self.wear,
            temperature_c=self.temperature_c,
            stint_age=self.stint_age,
            is_new=self.is_new,
        )


class TyreModel:
    """
    Computes tyre wear evolution and lap time penalty per lap.

    Parameters
    ----------
    compound_params : TyreCompoundParams
    track_abrasion  : float  Track surface abrasion factor (0–1)
    ambient_temp_c  : float  Ambient air temperature
    """

    def __init__(
        self,
        compound_params: TyreCompoundParams,
        track_abrasion: float = 0.65,
        ambient_temp_c: float = 25.0,
    ) -> None:
        self.params = compound_params
        self.track_abrasion = track_abrasion
        self.ambient_temp_c = ambient_temp_c

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        state: TyreState,
        push_factor: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> TyreState:
        """
        Advance tyre state by one lap.

        Parameters
        ----------
        state       : current TyreState
        push_factor : driving aggression (1.0 = nominal)
        rng         : random generator for stochastic noise

        Returns
        -------
        Updated TyreState (new object)
        """
        new_state = state.copy()

        # 1. Thermal model: temperature converges toward optimal
        thermal_penalty = self._thermal_penalty(state.temperature_c)
        new_temp = self._update_temperature(state, push_factor)

        # 2. Wear accumulation
        cliff_factor = self._cliff_factor(state.wear)
        delta_wear = (
            self.params.wear_rate_base
            * self.track_abrasion
            * push_factor
            * (1.0 + thermal_penalty)
            * cliff_factor
        )
        if rng is not None:
            # Small stochastic noise on wear rate (±5%)
            delta_wear *= rng.uniform(0.95, 1.05)

        new_state.wear = min(state.wear + delta_wear, 1.0)
        new_state.temperature_c = new_temp
        new_state.stint_age = state.stint_age + 1
        new_state.is_new = False
        return new_state

    def lap_time_delta(self, state: TyreState) -> float:
        """
        Returns seconds added to base lap time due to tyre state.

        Δt = initial_perf_advantage + a*w + b*w²
        """
        w = state.wear
        p = self.params
        wear_penalty = p.perf_loss_linear * w + p.perf_loss_quadratic * (w ** 2)
        return p.initial_perf_advantage_s + wear_penalty

    def is_cliff(self, state: TyreState) -> bool:
        return state.wear >= self.params.cliff_threshold

    def remaining_life(self, state: TyreState) -> float:
        """Fraction of usable life remaining (1 = fresh, 0 = at cliff)."""
        return max(0.0, (self.params.cliff_threshold - state.wear) / self.params.cliff_threshold)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _thermal_penalty(self, temp_c: float) -> float:
        """Extra wear fraction due to temperature deviation from optimal."""
        delta = abs(temp_c - self.params.optimal_temp_c)
        return self.params.thermal_degradation_factor * self.params.temp_sensitivity * delta

    def _update_temperature(self, state: TyreState, push_factor: float) -> float:
        """
        Simple first-order thermal model.
        High push_factor heats tyres; ambient pulls them back.
        """
        target_temp = self.params.optimal_temp_c * push_factor + self.ambient_temp_c * 0.1
        alpha = 0.25  # thermal inertia constant
        return state.temperature_c + alpha * (target_temp - state.temperature_c)

    def _cliff_factor(self, wear: float) -> float:
        """Multiplier applied to wear rate when past cliff threshold."""
        if wear >= self.params.cliff_threshold:
            # Linear ramp from 1x to cliff_multiplier past threshold
            overshoot = (wear - self.params.cliff_threshold) / (1.0 - self.params.cliff_threshold)
            return 1.0 + (self.params.cliff_multiplier - 1.0) * overshoot
        return 1.0
