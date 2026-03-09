"""
Lap Time Model
==============
Composes all contributing factors into a predicted lap time.

LapTime = BaseLapTime
        + TyrePenalty(wear, compound)
        + FuelPenalty(fuel_load)
        + TrafficPenalty(gap_ahead, overtaking_difficulty)
        + TrackEvolution(lap_number)
        + StochasticNoise(sigma)

All terms are in seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Optional

from src.core.tyre_model import TyreModel, TyreState
from src.core.fuel_model import FuelModel


@dataclass
class LapTimeModelParams:
    base_lap_time_s: float = 80.5
    track_evolution_rate: float = 0.003   # s improvement per lap
    overtaking_difficulty: float = 0.5    # 0=easy, 1=impossible
    lap_time_noise_std_s: float = 0.08    # stochastic driver variability


class LapTimeModel:
    """
    Computes the lap time for a given race state.

    Parameters
    ----------
    params       : LapTimeModelParams
    tyre_model   : TyreModel (carries compound info)
    fuel_model   : FuelModel
    """

    def __init__(
        self,
        params: LapTimeModelParams,
        tyre_model: TyreModel,
        fuel_model: FuelModel,
    ) -> None:
        self.params = params
        self.tyre_model  = tyre_model
        self.fuel_model  = fuel_model

    def predict(
        self,
        lap: int,
        tyre_state: TyreState,
        fuel_kg: float,
        gap_ahead_s: float = 999.0,
        safety_car_active: bool = False,
        rng: Optional[np.random.Generator] = None,
        ml_residual_s: float = 0.0,
    ) -> float:
        """
        Predict lap time for given race conditions.

        Parameters
        ----------
        lap              : current lap number (1-indexed)
        tyre_state       : current TyreState
        fuel_kg          : current fuel load
        gap_ahead_s      : gap to car ahead (seconds); large value = free air
        safety_car_active: whether safety car is deployed
        rng              : Generator for stochastic noise
        ml_residual_s    : optional ML-predicted residual correction

        Returns
        -------
        Predicted lap time in seconds
        """
        if safety_car_active:
            # Under SC, lap time is dictated by SC speed
            return self.params.base_lap_time_s * 1.35

        # --- Deterministic components ---
        base        = self.params.base_lap_time_s
        tyre_delta  = self.tyre_model.lap_time_delta(tyre_state)
        fuel_delta  = self.fuel_model.lap_time_delta(fuel_kg)
        track_evo   = self._track_evolution_delta(lap)
        traffic     = self._traffic_penalty(gap_ahead_s)

        # --- Stochastic noise ---
        noise = 0.0
        if rng is not None:
            noise = rng.normal(0.0, self.params.lap_time_noise_std_s)

        lap_time = base + tyre_delta + fuel_delta + track_evo + traffic + noise + ml_residual_s

        # Physical floor: can't be faster than theoretical minimum
        return max(lap_time, self.params.base_lap_time_s * 0.97)

    def predict_stint(
        self,
        start_lap: int,
        n_laps: int,
        tyre_state: TyreState,
        fuel_kg_start: float,
        push_factor: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> list[float]:
        """
        Predict lap times for an entire stint without pit stops.
        Useful for offline strategy evaluation.
        """
        lap_times: list[float] = []
        state = tyre_state

        for i in range(n_laps):
            lap = start_lap + i
            fuel = max(0.0, fuel_kg_start - self.fuel_model.params.consumption_kg_per_lap * i)
            lt = self.predict(lap, state, fuel, rng=rng)
            lap_times.append(lt)
            state = self.tyre_model.step(state, push_factor=push_factor, rng=rng)

        return lap_times

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _track_evolution_delta(self, lap: int) -> float:
        """
        Negative delta (lap times improve) as rubber is laid down.
        Saturates after ~30 laps.
        """
        saturation_lap = 30
        effect = min(lap, saturation_lap) * self.params.track_evolution_rate
        return -effect

    def _traffic_penalty(self, gap_ahead_s: float) -> float:
        """
        Time lost driving behind another car (dirty air).
        Penalty is significant within ~1.5 s, zero in free air.
        """
        if gap_ahead_s >= 2.0:
            return 0.0
        # Exponential decay: max ~0.4 s at 0 gap, zero at 2 s
        max_penalty = 0.4 * self.params.overtaking_difficulty
        return max_penalty * np.exp(-gap_ahead_s * 1.5)
