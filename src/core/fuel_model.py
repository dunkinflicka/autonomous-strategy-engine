"""
Fuel Model
==========
Models fuel consumption and its impact on lap time.

Physics:
    fuel_load(lap) = fuel_load_start - consumption_rate * lap
    lap_time_delta(fuel_kg) = fuel_penalty_s_per_kg * fuel_load_kg
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FuelModelParams:
    consumption_kg_per_lap: float = 1.85    # average consumption
    time_penalty_per_kg_s: float  = 0.034   # seconds per kg

    # Consumption variability (safety margin built into start load)
    consumption_variability: float = 0.05   # fraction


class FuelModel:
    """
    Tracks fuel load and computes its lap time contribution.

    Parameters
    ----------
    params          : FuelModelParams
    initial_load_kg : fuel load at race start (kg)
    """

    def __init__(
        self,
        params: FuelModelParams,
        initial_load_kg: float = 105.0,
    ) -> None:
        self.params = params
        self.initial_load_kg = initial_load_kg
        self._current_load_kg = initial_load_kg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> float:
        """
        Consume one lap's worth of fuel.
        Returns current fuel load BEFORE consumption (used for lap time calc).
        """
        load_before = self._current_load_kg
        self._current_load_kg = max(
            0.0,
            self._current_load_kg - self.params.consumption_kg_per_lap,
        )
        return load_before

    def lap_time_delta(self, fuel_kg: float | None = None) -> float:
        """
        Seconds added to lap time due to fuel weight.
        Uses current load if fuel_kg not specified.
        """
        load = fuel_kg if fuel_kg is not None else self._current_load_kg
        return self.params.time_penalty_per_kg_s * load

    def fuel_at_lap(self, lap: int) -> float:
        """Estimate fuel load at a given lap (0-indexed from start)."""
        return max(
            0.0,
            self.initial_load_kg - self.params.consumption_kg_per_lap * lap,
        )

    def reset(self, initial_load_kg: float | None = None) -> None:
        if initial_load_kg is not None:
            self.initial_load_kg = initial_load_kg
        self._current_load_kg = self.initial_load_kg

    @property
    def current_load_kg(self) -> float:
        return self._current_load_kg
