"""
Weather Model
=============
Stochastic weather evolution model.
Weather affects:
  - tyre degradation rate (wet tyres behave differently)
  - lap time (wet conditions add ~5-15 s)
  - safety car probability
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Optional


class WeatherCondition(Enum):
    DRY         = "dry"
    DAMP        = "damp"       # transition state
    LIGHT_RAIN  = "light_rain"
    HEAVY_RAIN  = "heavy_rain"


# Weather transition matrix: P(next | current)
_TRANSITION_MATRIX = {
    WeatherCondition.DRY:         {WeatherCondition.DRY: 0.95,
                                    WeatherCondition.DAMP: 0.05},
    WeatherCondition.DAMP:        {WeatherCondition.DRY: 0.30,
                                    WeatherCondition.DAMP: 0.40,
                                    WeatherCondition.LIGHT_RAIN: 0.30},
    WeatherCondition.LIGHT_RAIN:  {WeatherCondition.DAMP: 0.25,
                                    WeatherCondition.LIGHT_RAIN: 0.60,
                                    WeatherCondition.HEAVY_RAIN: 0.15},
    WeatherCondition.HEAVY_RAIN:  {WeatherCondition.LIGHT_RAIN: 0.45,
                                    WeatherCondition.HEAVY_RAIN: 0.55},
}

# Lap time delta (seconds) by weather condition (on dry tyres)
WEATHER_LAP_TIME_DELTA = {
    WeatherCondition.DRY:         0.0,
    WeatherCondition.DAMP:        3.5,
    WeatherCondition.LIGHT_RAIN:  8.0,
    WeatherCondition.HEAVY_RAIN:  15.0,
}

# Tyre wear rate multiplier under each condition (wet conditions = lower wear on dry tyres
# but adds tyre stress; full implementation would switch to wet compound)
WEATHER_WEAR_MULTIPLIER = {
    WeatherCondition.DRY:         1.0,
    WeatherCondition.DAMP:        1.15,
    WeatherCondition.LIGHT_RAIN:  0.7,   # Dry tyres: grip loss, but some aquaplaning
    WeatherCondition.HEAVY_RAIN:  0.5,   # Not meaningful — driver would pit for wets
}


@dataclass
class WeatherModelParams:
    initial_condition: WeatherCondition = WeatherCondition.DRY
    rain_probability: float = 0.0       # Used to bias initial transition probabilities
    update_interval_laps: int = 5       # Re-sample weather every N laps


class WeatherModel:
    """
    Markov chain weather model.
    Weather transitions are sampled every update_interval_laps laps.
    """

    def __init__(self, params: WeatherModelParams) -> None:
        self.params = params
        self._condition = params.initial_condition

    def step(self, lap: int, rng: np.random.Generator) -> WeatherCondition:
        """
        Update weather state. Only transitions on update interval boundaries.
        Returns current (possibly updated) condition.
        """
        if lap % self.params.update_interval_laps == 0:
            self._condition = self._sample_transition(rng)
        return self._condition

    def lap_time_delta(self) -> float:
        return WEATHER_LAP_TIME_DELTA[self._condition]

    def wear_multiplier(self) -> float:
        return WEATHER_WEAR_MULTIPLIER[self._condition]

    def is_wet(self) -> bool:
        return self._condition in (WeatherCondition.LIGHT_RAIN, WeatherCondition.HEAVY_RAIN)

    def reset(self) -> None:
        self._condition = self.params.initial_condition

    @property
    def condition(self) -> WeatherCondition:
        return self._condition

    # ------------------------------------------------------------------

    def _sample_transition(self, rng: np.random.Generator) -> WeatherCondition:
        transitions = _TRANSITION_MATRIX[self._condition]
        states = list(transitions.keys())
        probs  = np.array(list(transitions.values()))
        probs /= probs.sum()  # normalise for numerical safety
        return states[rng.choice(len(states), p=probs)]
