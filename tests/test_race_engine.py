"""Tests for the race simulation engine."""
import numpy as np
import pytest

from src.simulation.race_engine import RaceEngine, DriverStrategy, TrackConfig
from src.core.tyre_model import TyreCompoundParams

COMPOUND_PARAMS = {
    "soft": TyreCompoundParams("soft", 0.028, 0.15, 90.0, 0.008, 0.12, 0.18,
                                0.72, 2.8, 25, -0.9),
    "medium": TyreCompoundParams("medium", 0.018, 0.10, 85.0, 0.006, 0.08, 0.12,
                                  0.78, 2.3, 35, 0.0),
    "hard": TyreCompoundParams("hard", 0.012, 0.07, 80.0, 0.005, 0.05, 0.08,
                                0.85, 1.8, 50, 0.7),
}

TRACK = TrackConfig(
    name="test_track",
    total_laps=20,   # short race for testing
    base_lap_time_s=80.5,
    fuel_load_kg_start=50.0,
    track_abrasion=0.65,
    safety_car_base_rate=0.0,   # disable SC for deterministic tests
)


@pytest.fixture
def engine():
    return RaceEngine(TRACK, COMPOUND_PARAMS)


@pytest.fixture
def two_driver_strategies():
    s1 = DriverStrategy(0, pit_laps=[10], compounds=["medium", "hard"])
    s2 = DriverStrategy(1, pit_laps=[12], compounds=["soft", "hard"])
    return [s1, s2]


def test_race_produces_results(engine, two_driver_strategies):
    rng = np.random.default_rng(42)
    results = engine.simulate_race(two_driver_strategies, rng)
    assert len(results) == 2
    assert all(r.lap == TRACK.total_laps for r in results)


def test_finishing_positions_unique(engine, two_driver_strategies):
    rng = np.random.default_rng(42)
    results = engine.simulate_race(two_driver_strategies, rng)
    positions = [r.position for r in results]
    assert len(set(positions)) == len(positions), "Positions must be unique"


def test_cumulative_time_positive(engine, two_driver_strategies):
    rng = np.random.default_rng(42)
    results = engine.simulate_race(two_driver_strategies, rng)
    assert all(r.cumulative_time_s > 0 for r in results)


def test_pit_stop_count_matches_strategy(engine):
    strategy = DriverStrategy(0, pit_laps=[7, 14], compounds=["soft", "medium", "hard"])
    rng = np.random.default_rng(42)
    results = engine.simulate_race([strategy], rng)
    driver = results[0]
    assert driver.pit_stops_completed == 2


def test_fuel_decreases_each_lap(engine):
    s = DriverStrategy(0, pit_laps=[], compounds=["medium"])
    rng = np.random.default_rng(0)
    results = engine.simulate_race([s], rng)
    assert results[0].fuel_kg < TRACK.fuel_load_kg_start


def test_deterministic_with_same_seed(engine, two_driver_strategies):
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    r1 = engine.simulate_race(two_driver_strategies, rng1)
    r2 = engine.simulate_race(two_driver_strategies, rng2)
    assert r1[0].cumulative_time_s == r2[0].cumulative_time_s


def test_different_seeds_produce_different_results(engine, two_driver_strategies):
    results = []
    for seed in [42, 43, 44]:
        rng = np.random.default_rng(seed)
        r = engine.simulate_race(two_driver_strategies, rng)
        results.append(r[0].cumulative_time_s)
    # At least some variance expected
    assert len(set(f"{t:.6f}" for t in results)) > 1
