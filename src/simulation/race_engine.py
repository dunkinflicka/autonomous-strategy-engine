"""
Race Simulation Engine
======================
Core deterministic-with-stochastic-inputs race simulator.

Each call to simulate_race() produces a full race result from
start to finish, updating all physics models lap by lap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import numpy as np

from src.core.tyre_model import TyreModel, TyreState, TyreCompoundParams
from src.core.fuel_model import FuelModel, FuelModelParams
from src.core.lap_time_model import LapTimeModel, LapTimeModelParams
from src.core.safety_car_model import (
    PoissonSafetyCarModel, SafetyCarModelParams, SafetyCarStatus, SafetyCarEvent
)
from src.core.weather_model import WeatherModel, WeatherModelParams
from src.simulation.pit_stop_model import PitStopModel, PitStopParams
from src.simulation.race_state import DriverRaceState, RaceSnapshot
from src.utils.logging import SimulationEventLog


@dataclass
class TrackConfig:
    name: str = "monza"
    total_laps: int = 53
    base_lap_time_s: float = 80.5
    fuel_load_kg_start: float = 105.0
    track_abrasion: float = 0.55
    track_evolution_rate: float = 0.003
    overtaking_difficulty: float = 0.3
    safety_car_base_rate: float = 0.18
    safety_car_laps_mean: float = 4.5
    safety_car_laps_std: float = 1.5
    weather_rain_probability: float = 0.05


@dataclass
class DriverStrategy:
    """Encodes a driver's race strategy as a sequence of pit windows."""
    driver_id: int
    pit_laps: List[int]              # laps on which to pit
    compounds: List[str]             # compound for each stint (len = len(pit_laps) + 1)
    starting_compound: str = "medium"

    def compound_at_lap(self, lap: int) -> str:
        """Return compound being used at a given lap."""
        stint = 0
        for pit_lap in self.pit_laps:
            if lap > pit_lap:
                stint += 1
            else:
                break
        return self.compounds[min(stint, len(self.compounds) - 1)]

    def should_pit(self, lap: int) -> bool:
        return lap in self.pit_laps

    def next_compound(self, pit_stop_number: int) -> str:
        """Compound to fit after the Nth pit stop."""
        idx = pit_stop_number  # 0-indexed post-stop stint
        return self.compounds[min(idx + 1, len(self.compounds) - 1)]


class RaceEngine:
    """
    Simulates a full F1 race for multiple drivers.

    Parameters
    ----------
    track_config      : TrackConfig
    compound_params   : dict mapping compound name -> TyreCompoundParams
    lap_noise_std_s   : per-lap stochastic noise std (seconds)
    sc_model_type     : 'poisson' or 'logistic'
    """

    def __init__(
        self,
        track_config: TrackConfig,
        compound_params: Dict[str, TyreCompoundParams],
        lap_noise_std_s: float = 0.08,
        sc_model_type: str = "poisson",
        fuel_model_params: Optional[FuelModelParams] = None,
        pit_stop_params: Optional[PitStopParams] = None,
        weather_params: Optional[WeatherModelParams] = None,
    ) -> None:
        self.track = track_config
        self.compound_params = compound_params
        self.lap_noise_std_s = lap_noise_std_s
        self.fuel_params = fuel_model_params or FuelModelParams()
        self.pit_params = pit_stop_params or PitStopParams()

        self._sc_model = PoissonSafetyCarModel(
            SafetyCarModelParams(
                base_rate=track_config.safety_car_base_rate,
                safety_car_laps_mean=track_config.safety_car_laps_mean,
                safety_car_laps_std=track_config.safety_car_laps_std,
            ),
            total_laps=track_config.total_laps,
        )

        self._weather_params = weather_params or WeatherModelParams()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_race(
        self,
        strategies: List[DriverStrategy],
        rng: np.random.Generator,
        ml_residual_fn: Optional[Callable] = None,
        event_log: Optional[SimulationEventLog] = None,
    ) -> List[DriverRaceState]:
        """
        Simulate a complete race.

        Parameters
        ----------
        strategies     : one DriverStrategy per driver
        rng            : seeded random generator
        ml_residual_fn : optional function(driver_id, lap, state) -> float
        event_log      : optional event logger

        Returns
        -------
        List of final DriverRaceState, sorted by finishing position
        """
        # Initialise per-driver state
        driver_states = self._initialise_drivers(strategies)
        active_sc: Optional[SafetyCarEvent] = None
        weather_model = WeatherModel(self._weather_params)

        for lap in range(1, self.track.total_laps + 1):

            # --- Weather update ---
            weather_model.step(lap, rng)

            # --- Safety car check (only when no SC active) ---
            if active_sc is None:
                active_sc = self._sc_model.sample_event(lap, rng)
                if active_sc and event_log:
                    event_log.log(lap, "safety_car", status=active_sc.status.value,
                                  end_lap=active_sc.end_lap)

            sc_status = SafetyCarStatus.NONE
            if active_sc:
                if lap <= active_sc.end_lap:
                    sc_status = active_sc.status
                else:
                    active_sc = None

            # --- Simulate each driver's lap ---
            for ds in driver_states:
                if ds.dnf:
                    continue
                strategy = self._get_strategy(strategies, ds.driver_id)
                self._simulate_driver_lap(
                    ds, strategy, lap, sc_status, weather_model, rng,
                    ml_residual_fn, event_log
                )

            # --- Update positions ---
            self._update_positions(driver_states)

        return sorted(driver_states, key=lambda d: d.cumulative_time_s)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initialise_drivers(self, strategies: List[DriverStrategy]) -> List[DriverRaceState]:
        states = []
        for i, strat in enumerate(strategies):
            tyre_state = TyreState(
                compound=strat.starting_compound,
                wear=0.02 if strat.starting_compound == "soft" else 0.0,
                temperature_c=75.0,
                stint_age=0,
                is_new=True,
            )
            ds = DriverRaceState(
                driver_id=strat.driver_id,
                lap=0,
                position=i + 1,
                cumulative_time_s=i * 0.3,  # staggered grid start
                last_lap_time_s=0.0,
                tyre_state=tyre_state,
                fuel_kg=self.track.fuel_load_kg_start,
                pit_stops_completed=0,
                laps_since_pit=0,
            )
            states.append(ds)
        return states

    def _simulate_driver_lap(
        self,
        ds: DriverRaceState,
        strategy: DriverStrategy,
        lap: int,
        sc_status: SafetyCarStatus,
        weather_model: WeatherModel,
        rng: np.random.Generator,
        ml_residual_fn,
        event_log,
    ) -> None:
        """Mutates ds in-place for the current lap."""
        # Build per-driver models (compound-specific)
        compound = ds.tyre_state.compound
        c_params = self.compound_params[compound]
        tyre_model = TyreModel(c_params, self.track.track_abrasion)
        fuel_model = FuelModel(self.fuel_params, ds.fuel_kg)
        fuel_model._current_load_kg = ds.fuel_kg

        lt_params = LapTimeModelParams(
            base_lap_time_s=self.track.base_lap_time_s,
            track_evolution_rate=self.track.track_evolution_rate,
            overtaking_difficulty=self.track.overtaking_difficulty,
            lap_time_noise_std_s=self.lap_noise_std_s,
        )
        lt_model = LapTimeModel(lt_params, tyre_model, fuel_model)

        # ML residual
        ml_residual = 0.0
        if ml_residual_fn is not None:
            ml_residual = ml_residual_fn(ds.driver_id, lap, ds)

        # Gap to car ahead
        gap_ahead = 999.0  # default: free air (simplified)

        # Pit stop decision
        is_pitting = strategy.should_pit(lap)
        pit_delta = 0.0
        if is_pitting:
            pit_model = PitStopModel(self.pit_params)
            pit_delta, unsafe = pit_model.sample_stop_time(rng)
            new_compound = strategy.next_compound(ds.pit_stops_completed)
            ds.tyre_state = TyreState(
                compound=new_compound,
                wear=0.0,
                temperature_c=70.0,
                stint_age=0,
                is_new=True,
            )
            ds.pit_stops_completed += 1
            ds.laps_since_pit = 0
            if event_log:
                event_log.log(lap, "pit_stop", driver_id=ds.driver_id,
                              compound=new_compound, unsafe=unsafe)
        else:
            ds.laps_since_pit += 1

        sc_active = sc_status != SafetyCarStatus.NONE
        lap_time = lt_model.predict(
            lap=lap,
            tyre_state=ds.tyre_state,
            fuel_kg=ds.fuel_kg,
            gap_ahead_s=gap_ahead,
            safety_car_active=sc_active,
            rng=rng,
            ml_residual_s=ml_residual,
        )

        # Add weather penalty
        lap_time += weather_model.lap_time_delta() if not sc_active else 0.0
        lap_time += pit_delta

        # Consume fuel
        ds.fuel_kg = max(0.0, ds.fuel_kg - self.fuel_params.consumption_kg_per_lap)

        # Update tyre wear
        push = 1.0 if not sc_active else 0.7
        ds.tyre_state = tyre_model.step(ds.tyre_state, push_factor=push, rng=rng)

        # Update cumulative state
        ds.last_lap_time_s = lap_time
        ds.cumulative_time_s += lap_time
        ds.lap = lap
        ds.is_pitting_this_lap = is_pitting

    def _update_positions(self, driver_states: List[DriverRaceState]) -> None:
        sorted_drivers = sorted(driver_states, key=lambda d: d.cumulative_time_s)
        for pos, d in enumerate(sorted_drivers):
            d.position = pos + 1
            leader_time = sorted_drivers[0].cumulative_time_s
            d.gap_to_leader_s = d.cumulative_time_s - leader_time

    @staticmethod
    def _get_strategy(strategies: List[DriverStrategy], driver_id: int) -> DriverStrategy:
        for s in strategies:
            if s.driver_id == driver_id:
                return s
        raise KeyError(f"No strategy for driver {driver_id}")
