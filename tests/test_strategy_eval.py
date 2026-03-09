"""Tests for the strategy evaluation layer."""
import numpy as np
import pytest

from src.simulation.race_engine import RaceEngine, DriverStrategy, TrackConfig
from src.simulation.monte_carlo import MonteCarloEngine, MonteCarloResult
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
    name="test_track", total_laps=15, base_lap_time_s=80.5,
    fuel_load_kg_start=35.0, track_abrasion=0.65, safety_car_base_rate=0.0,
)


@pytest.fixture
def mc_engine():
    engine = RaceEngine(TRACK, COMPOUND_PARAMS)
    return MonteCarloEngine(engine, n_simulations=200, base_seed=0, show_progress=False)


@pytest.fixture
def strategies():
    return [
        DriverStrategy(0, pit_laps=[8], compounds=["medium", "hard"]),
        DriverStrategy(1, pit_laps=[7], compounds=["soft", "hard"]),
    ]


def test_mc_returns_correct_shape(mc_engine, strategies):
    result = mc_engine.run(strategies)
    assert result.position_matrix.shape == (200, 2)
    assert result.race_time_matrix.shape == (200, 2)


def test_win_probability_sums_correctly(mc_engine, strategies):
    result = mc_engine.run(strategies)
    total_win_prob = sum(result.win_probability(s.driver_id) for s in strategies)
    assert abs(total_win_prob - 1.0) < 0.01, "Win probabilities must sum to 1"


def test_positions_always_valid(mc_engine, strategies):
    result = mc_engine.run(strategies)
    n_drivers = len(strategies)
    assert np.all(result.position_matrix >= 1)
    assert np.all(result.position_matrix <= n_drivers)


def test_expected_position_in_range(mc_engine, strategies):
    result = mc_engine.run(strategies)
    for s in strategies:
        ep = result.expected_position(s.driver_id)
        assert 1 <= ep <= len(strategies), f"Expected position out of range: {ep}"


def test_expected_points_non_negative(mc_engine, strategies):
    result = mc_engine.run(strategies)
    for s in strategies:
        assert result.expected_points(s.driver_id) >= 0


def test_reproducibility(mc_engine, strategies):
    r1 = MonteCarloEngine(
        RaceEngine(TRACK, COMPOUND_PARAMS), n_simulations=50, base_seed=99, show_progress=False
    ).run(strategies)
    r2 = MonteCarloEngine(
        RaceEngine(TRACK, COMPOUND_PARAMS), n_simulations=50, base_seed=99, show_progress=False
    ).run(strategies)
    np.testing.assert_array_equal(r1.position_matrix, r2.position_matrix)
