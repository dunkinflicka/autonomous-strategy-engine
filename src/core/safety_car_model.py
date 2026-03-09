"""
Safety Car Model
================
Stochastic model for safety car (SC) and virtual safety car (VSC) deployments.

Two approaches implemented:
1. Poisson process — simple, interpretable, calibrated to historical rates
2. Logistic regression — conditioned on lap, track, and weather inputs

P(SC event on lap l | track, conditions)

Safety car deployment significantly impacts pit strategy windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

from src.utils.distributions import logistic, PoissonProcess


class SafetyCarStatus(Enum):
    NONE    = "none"
    VSC     = "virtual_safety_car"
    SC      = "safety_car"


@dataclass
class SafetyCarEvent:
    start_lap: int
    end_lap: int
    status: SafetyCarStatus


@dataclass
class SafetyCarModelParams:
    base_rate: float = 0.18          # P(event) per race (historical)
    safety_car_laps_mean: float = 4.5
    safety_car_laps_std: float  = 1.5
    vsc_probability: float = 0.4     # P(VSC | event triggered)
    min_duration_laps: int = 2
    max_duration_laps: int = 8
    # Logistic model weights (lap_number, track_abrasion, is_wet)
    logistic_intercept: float = -2.5
    logistic_lap_coeff: float = 0.008
    logistic_abrasion_coeff: float = 0.4
    logistic_wet_coeff: float = 0.6


class PoissonSafetyCarModel:
    """
    Poisson process safety car model.
    Rate is set from historical track data.
    """

    def __init__(self, params: SafetyCarModelParams, total_laps: int) -> None:
        self.params = params
        self.total_laps = total_laps
        # Convert race probability to per-lap rate
        self._per_lap_rate = params.base_rate / total_laps
        self._process = PoissonProcess(rate=self._per_lap_rate)

    def sample_event(
        self,
        lap: int,
        rng: np.random.Generator,
    ) -> Optional[SafetyCarEvent]:
        """
        Returns a SafetyCarEvent if one is triggered on this lap, else None.
        Does not allow overlapping events (caller must check active status).
        """
        if not self._process.event_occurred(rng):
            return None

        is_vsc = rng.random() < self.params.vsc_probability
        status = SafetyCarStatus.VSC if is_vsc else SafetyCarStatus.SC

        duration = int(np.clip(
            rng.normal(self.params.safety_car_laps_mean, self.params.safety_car_laps_std),
            self.params.min_duration_laps,
            self.params.max_duration_laps,
        ))

        return SafetyCarEvent(
            start_lap=lap,
            end_lap=min(lap + duration, self.total_laps),
            status=status,
        )


class LogisticSafetyCarModel:
    """
    Logistic regression model for safety car probability.
    Conditioned on lap number, track abrasion, and weather.
    Enables calibration from historical race data.
    """

    def __init__(
        self,
        params: SafetyCarModelParams,
        total_laps: int,
        track_abrasion: float = 0.65,
    ) -> None:
        self.params = params
        self.total_laps = total_laps
        self.track_abrasion = track_abrasion
        # Coefficient vector [intercept, lap, abrasion, is_wet]
        self._coefs = np.array([
            params.logistic_intercept,
            params.logistic_lap_coeff,
            params.logistic_abrasion_coeff,
            params.logistic_wet_coeff,
        ])

    def probability(self, lap: int, is_wet: bool = False) -> float:
        """P(SC event | lap, conditions)"""
        x = np.array([
            1.0,
            lap / self.total_laps,  # normalised lap position
            self.track_abrasion,
            float(is_wet),
        ])
        return float(logistic(self._coefs @ x))

    def sample_event(
        self,
        lap: int,
        rng: np.random.Generator,
        is_wet: bool = False,
    ) -> Optional[SafetyCarEvent]:
        p = self.probability(lap, is_wet)
        if rng.random() >= p:
            return None

        is_vsc = rng.random() < self.params.vsc_probability
        status = SafetyCarStatus.VSC if is_vsc else SafetyCarStatus.SC
        duration = int(np.clip(
            rng.normal(self.params.safety_car_laps_mean, self.params.safety_car_laps_std),
            self.params.min_duration_laps,
            self.params.max_duration_laps,
        ))
        return SafetyCarEvent(
            start_lap=lap,
            end_lap=min(lap + duration, self.total_laps),
            status=status,
        )

    def update_coefficients(self, coefs: np.ndarray) -> None:
        """Allow ML calibration to update logistic weights."""
        if coefs.shape != (4,):
            raise ValueError(f"Expected 4 coefficients, got {coefs.shape}")
        self._coefs = coefs.copy()
