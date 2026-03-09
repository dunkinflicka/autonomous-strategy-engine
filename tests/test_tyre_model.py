"""Tests for tyre degradation model."""
import numpy as np
import pytest
from src.core.tyre_model import TyreModel, TyreState, TyreCompoundParams

MEDIUM_PARAMS = TyreCompoundParams(
    id="medium",
    wear_rate_base=0.018,
    thermal_degradation_factor=0.10,
    optimal_temp_c=85.0,
    temp_sensitivity=0.006,
    perf_loss_linear=0.08,
    perf_loss_quadratic=0.12,
    cliff_threshold=0.78,
    cliff_multiplier=2.3,
    max_stint_laps=35,
    initial_perf_advantage_s=0.0,
)


@pytest.fixture
def medium_model():
    return TyreModel(MEDIUM_PARAMS, track_abrasion=0.65, ambient_temp_c=25.0)


@pytest.fixture
def fresh_state():
    return TyreState(compound="medium", wear=0.0, temperature_c=80.0,
                     stint_age=0, is_new=True)


def test_wear_increases_each_lap(medium_model, fresh_state):
    rng = np.random.default_rng(42)
    state = fresh_state
    for _ in range(10):
        new_state = medium_model.step(state, rng=rng)
        assert new_state.wear > state.wear, "Wear must increase each lap"
        state = new_state


def test_wear_bounded_at_one(medium_model):
    """Extreme wear should never exceed 1.0."""
    state = TyreState("medium", wear=0.99, temperature_c=90.0, stint_age=50)
    rng = np.random.default_rng(0)
    new_state = medium_model.step(state, push_factor=2.0, rng=rng)
    assert new_state.wear <= 1.0


def test_lap_time_delta_fresh_tyre(medium_model, fresh_state):
    """Fresh medium tyre should give ~0 delta vs baseline."""
    delta = medium_model.lap_time_delta(fresh_state)
    assert abs(delta) < 0.5, f"Fresh tyre delta too large: {delta}"


def test_lap_time_delta_increases_with_wear(medium_model):
    """Performance loss must increase monotonically with wear."""
    deltas = []
    for wear in np.linspace(0, 0.9, 10):
        state = TyreState("medium", wear=wear, temperature_c=85.0, stint_age=0)
        deltas.append(medium_model.lap_time_delta(state))
    for i in range(1, len(deltas)):
        assert deltas[i] > deltas[i-1], "Lap time penalty must increase with wear"


def test_cliff_detection(medium_model):
    """Cliff should be detected above threshold."""
    state_safe = TyreState("medium", wear=0.70, temperature_c=85.0, stint_age=30)
    state_cliff = TyreState("medium", wear=0.82, temperature_c=85.0, stint_age=40)
    assert not medium_model.is_cliff(state_safe)
    assert medium_model.is_cliff(state_cliff)


def test_cliff_accelerates_wear(medium_model):
    """Wear accumulation should be faster past the cliff threshold."""
    rng = np.random.default_rng(42)
    safe_state  = TyreState("medium", wear=0.60, temperature_c=85.0, stint_age=25)
    cliff_state = TyreState("medium", wear=0.85, temperature_c=85.0, stint_age=40)

    safe_new  = medium_model.step(safe_state, rng=np.random.default_rng(1))
    cliff_new = medium_model.step(cliff_state, rng=np.random.default_rng(1))

    safe_delta  = safe_new.wear  - safe_state.wear
    cliff_delta = cliff_new.wear - cliff_state.wear
    assert cliff_delta > safe_delta, "Cliff should accelerate wear"


def test_stint_age_increments(medium_model, fresh_state):
    rng = np.random.default_rng(0)
    state = fresh_state
    for expected_age in range(1, 6):
        state = medium_model.step(state, rng=rng)
        assert state.stint_age == expected_age
