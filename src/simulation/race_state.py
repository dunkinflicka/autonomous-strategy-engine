"""
Race State
==========
Immutable snapshot of a driver's state at a given point in the race.
The race engine produces a new RaceState each lap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.core.tyre_model import TyreState
from src.core.safety_car_model import SafetyCarStatus


@dataclass
class DriverRaceState:
    """Per-driver state for a single lap."""
    driver_id: int
    lap: int
    position: int
    cumulative_time_s: float        # total elapsed race time
    last_lap_time_s: float
    tyre_state: TyreState
    fuel_kg: float
    pit_stops_completed: int = 0
    laps_since_pit: int = 0
    is_pitting_this_lap: bool = False
    dnf: bool = False
    gap_to_leader_s: float = 0.0

    def copy(self) -> "DriverRaceState":
        return DriverRaceState(
            driver_id=self.driver_id,
            lap=self.lap,
            position=self.position,
            cumulative_time_s=self.cumulative_time_s,
            last_lap_time_s=self.last_lap_time_s,
            tyre_state=self.tyre_state.copy(),
            fuel_kg=self.fuel_kg,
            pit_stops_completed=self.pit_stops_completed,
            laps_since_pit=self.laps_since_pit,
            is_pitting_this_lap=self.is_pitting_this_lap,
            dnf=self.dnf,
            gap_to_leader_s=self.gap_to_leader_s,
        )


@dataclass
class RaceSnapshot:
    """Complete race state at end of a given lap."""
    lap: int
    total_laps: int
    safety_car_status: SafetyCarStatus
    drivers: List[DriverRaceState]

    @property
    def laps_remaining(self) -> int:
        return self.total_laps - self.lap

    @property
    def leader(self) -> DriverRaceState:
        return min(self.drivers, key=lambda d: d.cumulative_time_s)

    def driver_by_id(self, driver_id: int) -> Optional[DriverRaceState]:
        for d in self.drivers:
            if d.driver_id == driver_id:
                return d
        return None

    def sorted_by_position(self) -> List[DriverRaceState]:
        return sorted(self.drivers, key=lambda d: d.cumulative_time_s)
