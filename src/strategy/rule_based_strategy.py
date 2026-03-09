"""
Rule-Based Strategy Templates
==============================
Predefined strategy templates covering common F1 race strategies:
  - one-stop (medium → hard)
  - two-stop (soft → medium → hard)
  - undercut (pit earlier than opponent to gain track position)
  - overcut (stay out longer to gain position during rivals' stops)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from src.simulation.race_engine import DriverStrategy


def one_stop_strategy(
    driver_id: int,
    total_laps: int,
    pit_lap: Optional[int] = None,
    opening_compound: str = "medium",
    closing_compound: str = "hard",
) -> DriverStrategy:
    """
    Classic one-stop: open on medium, switch to hard mid-race.
    Default pit window is 40-50% race distance.
    """
    if pit_lap is None:
        pit_lap = int(total_laps * 0.45)
    return DriverStrategy(
        driver_id=driver_id,
        pit_laps=[pit_lap],
        compounds=[opening_compound, closing_compound],
        starting_compound=opening_compound,
    )


def two_stop_strategy(
    driver_id: int,
    total_laps: int,
    pit_lap_1: Optional[int] = None,
    pit_lap_2: Optional[int] = None,
    compounds: Optional[List[str]] = None,
) -> DriverStrategy:
    """
    Two-stop: split race into three stints.
    Default windows at ~30% and ~65% of race distance.
    """
    if pit_lap_1 is None:
        pit_lap_1 = int(total_laps * 0.30)
    if pit_lap_2 is None:
        pit_lap_2 = int(total_laps * 0.65)
    if compounds is None:
        compounds = ["soft", "medium", "hard"]
    return DriverStrategy(
        driver_id=driver_id,
        pit_laps=[pit_lap_1, pit_lap_2],
        compounds=compounds,
        starting_compound=compounds[0],
    )


def undercut_strategy(
    driver_id: int,
    opponent_pit_lap: int,
    total_laps: int,
    undercut_offset: int = -3,
    compounds: Optional[List[str]] = None,
) -> DriverStrategy:
    """
    Undercut: pit `undercut_offset` laps BEFORE the opponent.
    Fresh tyres allow faster laps while opponent is on degraded rubber.
    """
    pit_lap = max(5, opponent_pit_lap + undercut_offset)
    if compounds is None:
        compounds = ["medium", "hard"]
    return DriverStrategy(
        driver_id=driver_id,
        pit_laps=[pit_lap],
        compounds=compounds,
        starting_compound=compounds[0],
    )


def overcut_strategy(
    driver_id: int,
    opponent_pit_lap: int,
    total_laps: int,
    overcut_offset: int = 3,
    compounds: Optional[List[str]] = None,
) -> DriverStrategy:
    """
    Overcut: stay out LONGER than the opponent, accumulating free laps
    on track while the opponent is in the pit lane.
    """
    pit_lap = min(total_laps - 5, opponent_pit_lap + overcut_offset)
    if compounds is None:
        compounds = ["medium", "hard"]
    return DriverStrategy(
        driver_id=driver_id,
        pit_laps=[pit_lap],
        compounds=compounds,
        starting_compound=compounds[0],
    )


def safety_car_strategy(
    driver_id: int,
    sc_start_lap: int,
    compounds: Optional[List[str]] = None,
) -> DriverStrategy:
    """
    Opportunistic safety car pit: pit on SC lap + 1 to minimise time loss.
    """
    pit_lap = sc_start_lap + 1
    if compounds is None:
        compounds = ["medium", "hard"]
    return DriverStrategy(
        driver_id=driver_id,
        pit_laps=[pit_lap],
        compounds=compounds,
        starting_compound=compounds[0],
    )
