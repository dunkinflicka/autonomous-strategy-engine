"""Tests for lap time composition model."""
import numpy as np
import pytest

from src.core.tyre_model import TyreModel, TyreState, TyreCompoundParams
from src.core.fuel_model import FuelModel, FuelModelParams
from src.core.lap_time_model import LapTimeModel, LapTimeModelParams

MEDIUM_PARAMS = TyreCompoundParams(
    id="medium", wear_rate_base=0.018, thermal_degradation_factor=0.10,
    optimal_temp_c=85.0, temp_sensitivity=0.006, perf_loss_linear=0.08,
    perf_loss_quadratic=0.12, cliff_threshold=0.78, cliff_multiplier=2.3,
    max_stint_laps=35, initial_perf_advantage_s=0.0,
)


@pytest.fixture
def lt_model():
    tyre = TyreModel(MEDIUM_PARAMS, track_abrasion=0.65)
    fuel = FuelModel(FuelModelParams(), initial_load_kg=100.0)
    params = LapTimeModelParams(
        base_lap_time_s=80.5,
        track_evolution_rate=0.003,
        lap_time_noise_std_s=0.0,  # deterministic
    )
    return LapTimeModel(params, tyre, fuel)


def test_base_lap_time_reasonable(lt_model):
    state = TyreState("medium", wear=0.0, temperature_c=85.0, stint_age=0)
    lt = lt_model.predict(lap=1, tyre_state=state, fuel_kg=100.0)
    # Should be around 80.5 + fuel penalty (~3.4 s) = ~83.9
    assert 80.0 < lt < 90.0, f"Unreasonable lap time: {lt}"


def test_lap_time_increases_with_tyre_wear(lt_model):
    fresh = TyreState("medium", wear=0.0, temperature_c=85.0, stint_age=0)
    worn  = TyreState("medium", wear=0.7, temperature_c=85.0, stint_age=30)
    lt_fresh = lt_model.predict(1, fresh, 100.0)
    lt_worn  = lt_model.predict(1, worn,  100.0)
    assert lt_worn > lt_fresh, "Worn tyre must be slower"


def test_lap_time_decreases_with_fuel_burn(lt_model):
    state = TyreState("medium", wear=0.0, temperature_c=85.0, stint_age=0)
    lt_full  = lt_model.predict(1, state, fuel_kg=100.0)
    lt_empty = lt_model.predict(1, state, fuel_kg=10.0)
    assert lt_empty < lt_full, "Lighter fuel load should be faster"


def test_safety_car_slower(lt_model):
    state = TyreState("medium", wear=0.0, temperature_c=85.0, stint_age=0)
    lt_normal = lt_model.predict(1, state, fuel_kg=80.0, safety_car_active=False)
    lt_sc     = lt_model.predict(1, state, fuel_kg=80.0, safety_car_active=True)
    assert lt_sc > lt_normal, "Safety car must produce slower lap times"


def test_track_evolution_effect(lt_model):
    state = TyreState("medium", wear=0.1, temperature_c=85.0, stint_age=5)
    lt_early = lt_model.predict(lap=1,  tyre_state=state, fuel_kg=80.0)
    lt_late  = lt_model.predict(lap=30, tyre_state=state, fuel_kg=80.0)
    assert lt_late < lt_early, "Track evolution should improve lap times"


def test_traffic_penalty_in_dirty_air(lt_model):
    state = TyreState("medium", wear=0.0, temperature_c=85.0, stint_age=0)
    lt_free    = lt_model.predict(1, state, 80.0, gap_ahead_s=999.0)
    lt_traffic = lt_model.predict(1, state, 80.0, gap_ahead_s=0.5)
    assert lt_traffic > lt_free, "Traffic should add time"


def test_stochastic_noise_non_zero_with_rng():
    tyre = TyreModel(MEDIUM_PARAMS, 0.65)
    fuel = FuelModel(FuelModelParams(), 100.0)
    params = LapTimeModelParams(base_lap_time_s=80.5, lap_time_noise_std_s=0.1)
    model = LapTimeModel(params, tyre, fuel)
    state = TyreState("medium", wear=0.1, temperature_c=85.0, stint_age=5)

    rng = np.random.default_rng(0)
    samples = [model.predict(1, state, 80.0, rng=rng) for _ in range(50)]
    assert np.std(samples) > 0.01, "Stochastic noise should produce variance"
