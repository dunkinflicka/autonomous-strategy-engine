# # """
# # F1 Strategy RL Environment
# # ==========================
# # Gymnasium-compatible environment where an agent controls pit strategy.

# # State space (normalised):
# #     [lap/total_laps, tyre_wear, compound_enc, fuel_fraction,
# #      position/n_drivers, gap_ahead_s/60, safety_car_active,
# #      laps_since_pit/total_laps]

# # Action space (Discrete 4):
# #     0: stay_out
# #     1: pit → soft
# #     2: pit → medium
# #     3: pit → hard

# # Reward:
# #     - Final reward: f(finishing_position) on terminal step
# #     - Intermediate shaping: -0.01 per lap (encourage efficiency)
# #     - Penalty for tyre cliff exploitation: -0.05 per cliff lap
# # """
# # from __future__ import annotations

# # from typing import Any, Dict, Optional, Tuple
# # import numpy as np
# # import gymnasium as gym
# # from gymnasium import spaces

# # from src.simulation.race_engine import RaceEngine, DriverStrategy, TrackConfig
# # from src.simulation.race_state import DriverRaceState
# # from src.core.tyre_model import TyreModel, TyreState, TyreCompoundParams
# # from src.core.fuel_model import FuelModel, FuelModelParams
# # from src.core.lap_time_model import LapTimeModel, LapTimeModelParams
# # from src.core.safety_car_model import (
# #     PoissonSafetyCarModel, SafetyCarModelParams, SafetyCarStatus
# # )
# # from src.core.weather_model import WeatherModel, WeatherModelParams
# # from src.simulation.pit_stop_model import PitStopModel, PitStopParams
# # from src.rl.reward_function import RewardFunction

# # COMPOUND_ENCODING = {"soft": 0.0, "medium": 0.5, "hard": 1.0}
# # ACTION_TO_COMPOUND = {0: None, 1: "soft", 2: "medium", 3: "hard"}


# # class F1StrategyEnv(gym.Env):
# #     """
# #     Single-driver F1 strategy environment.
# #     The RL agent controls one driver; opponents follow fixed strategies.
# #     """
# #     metadata = {"render_modes": []}

# #     def __init__(
# #         self,
# #         track_config: TrackConfig,
# #         compound_params: Dict[str, TyreCompoundParams],
# #         opponent_strategies: Optional[list] = None,
# #         n_opponents: int = 5,
# #         lap_noise_std_s: float = 0.08,
# #         seed: int = 42,
# #     ) -> None:
# #         super().__init__()
# #         self.track = track_config
# #         self.compound_params = compound_params
# #         self.n_opponents = n_opponents
# #         self.lap_noise_std_s = lap_noise_std_s

# #         # Observation: 8-dimensional normalised state vector
# #         self.observation_space = spaces.Box(
# #             low=np.zeros(8, dtype=np.float32),
# #             high=np.ones(8, dtype=np.float32),
# #             dtype=np.float32,
# #         )
# #         # Action: 0=stay, 1=pit_soft, 2=pit_medium, 3=pit_hard
# #         self.action_space = spaces.Discrete(4)

# #         self._reward_fn = RewardFunction(n_drivers=n_opponents + 1)
# #         self._rng = np.random.default_rng(seed)

# #         # Mutable episode state (reset on each reset())
# #         self._lap = 0
# #         self._tyre_state: Optional[TyreState] = None
# #         self._fuel_kg = track_config.fuel_load_kg_start
# #         self._position = 1
# #         self._gap_ahead_s = 999.0
# #         self._laps_since_pit = 0
# #         self._pit_stops = 0
# #         self._sc_active = False
# #         self._cumulative_time_s = 0.0

# #         self._sc_model = PoissonSafetyCarModel(
# #             SafetyCarModelParams(base_rate=track_config.safety_car_base_rate),
# #             total_laps=track_config.total_laps,
# #         )
# #         self._weather = WeatherModel(WeatherModelParams())
# #         self._pit_model = PitStopModel(PitStopParams())
# #         self._fuel_model = FuelModel(FuelModelParams(), track_config.fuel_load_kg_start)

# #     # ------------------------------------------------------------------
# #     # Gymnasium interface
# #     # ------------------------------------------------------------------

# #     def reset(
# #         self,
# #         seed: Optional[int] = None,
# #         options: Optional[Dict] = None,
# #     ) -> Tuple[np.ndarray, Dict]:
# #         if seed is not None:
# #             self._rng = np.random.default_rng(seed)

# #         self._lap = 0
# #         self._tyre_state = TyreState(
# #             compound="medium", wear=0.0, temperature_c=80.0, stint_age=0, is_new=True
# #         )
# #         self._fuel_kg = self.track.fuel_load_kg_start
# #         self._fuel_model.reset(self.track.fuel_load_kg_start)
# #         self._position = self._rng.integers(1, self.n_opponents + 2)
# #         self._gap_ahead_s = float(self._rng.exponential(2.0)) if self._position > 1 else 999.0
# #         self._laps_since_pit = 0
# #         self._pit_stops = 0
# #         self._sc_active = False
# #         self._cumulative_time_s = 0.0
# #         self._weather.reset()

# #         return self._get_obs(), {}

# #     def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
# #         self._lap += 1
# #         pit_compound = ACTION_TO_COMPOUND[action]
# #         is_pitting = pit_compound is not None

# #         # --- Safety car ---
# #         sc_event = self._sc_model.sample_event(self._lap, self._rng)
# #         self._sc_active = sc_event is not None

# #         # --- Weather ---
# #         # weather = self._weather.step(self._lap, self._rng)
# #         self._weather.step(self._lap, self._rng)
# #         # --- Pit stop ---
# #         pit_delta = 0.0
# #         if is_pitting:
# #             pit_delta, _ = self._pit_model.sample_stop_time(self._rng)
# #             self._tyre_state = TyreState(
# #                 compound=pit_compound, wear=0.0,
# #                 temperature_c=70.0, stint_age=0, is_new=True
# #             )
# #             self._pit_stops += 1
# #             self._laps_since_pit = 0
# #         else:
# #             self._laps_since_pit += 1

# #         # --- Lap time ---
# #         c_params = self.compound_params[self._tyre_state.compound]
# #         tyre_model = TyreModel(c_params, self.track.track_abrasion)
# #         lt_params = LapTimeModelParams(
# #             base_lap_time_s=self.track.base_lap_time_s,
# #             track_evolution_rate=self.track.track_evolution_rate,
# #             lap_time_noise_std_s=self.lap_noise_std_s,
# #         )
# #         lt_model = LapTimeModel(lt_params, tyre_model, self._fuel_model)

# #         lap_time = lt_model.predict(
# #             lap=self._lap,
# #             tyre_state=self._tyre_state,
# #             fuel_kg=self._fuel_kg,
# #             gap_ahead_s=self._gap_ahead_s,
# #             safety_car_active=self._sc_active,
# #             rng=self._rng,
# #         ) + pit_delta + self._weather.lap_time_delta()

# #         self._cumulative_time_s += lap_time

# #         # Update tyre
# #         self._tyre_state = tyre_model.step(
# #             self._tyre_state,
# #             push_factor=0.7 if self._sc_active else 1.0,
# #             rng=self._rng,
# #         )

# #         # Consume fuel
# #         self._fuel_kg = max(0.0, self._fuel_kg - 1.85)

# #         # Simulate position change (simplified)
# #         self._position = self._simulate_position_update()

# #         # Terminal condition
# #         terminated = self._lap >= self.track.total_laps
# #         truncated  = False

# #         # Reward
# #         reward = self._reward_fn.compute(
# #             lap=self._lap,
# #             total_laps=self.track.total_laps,
# #             position=self._position,
# #             tyre_state=self._tyre_state,
# #             is_terminal=terminated,
# #             pit_stops=self._pit_stops,
# #         )

# #         info = {
# #             "lap": self._lap,
# #             "position": self._position,
# #             "tyre_wear": self._tyre_state.wear,
# #             "fuel_kg": self._fuel_kg,
# #             "lap_time_s": lap_time,
# #         }

# #         return self._get_obs(), reward, terminated, truncated, info

# #     # ------------------------------------------------------------------

# #     def _get_obs(self) -> np.ndarray:
# #         compound_enc = COMPOUND_ENCODING.get(
# #             self._tyre_state.compound if self._tyre_state else "medium", 0.5
# #         )
# #         wear = self._tyre_state.wear if self._tyre_state else 0.0
# #         obs = np.array([
# #             self._lap / self.track.total_laps,
# #             wear,
# #             compound_enc,
# #             self._fuel_kg / self.track.fuel_load_kg_start,
# #             self._position / (self.n_opponents + 1),
# #             min(self._gap_ahead_s / 60.0, 1.0),
# #             float(self._sc_active),
# #             self._laps_since_pit / self.track.total_laps,
# #         ], dtype=np.float32)
# #         return np.clip(obs, 0.0, 1.0)

# #     # def _simulate_position_update(self) -> int:
# #     #     """
# #     #     Simplified position model: position changes stochastically
# #     #     based on recent lap time performance.
# #     #     Full multi-driver simulation used in production experiments.
# #     #     """
# #     #     delta = self._rng.integers(-1, 2)  # -1, 0, or +1
# #     #     new_pos = int(np.clip(self._position + delta, 1, self.n_opponents + 1))
# #     #     return new_pos
# #     # def _simulate_position_update(self) -> int:
# #     #     """
# #     #     Position estimated from cumulative lap time disadvantage.
# #     #     Cars with more pit stops have spent more time in pit lane.
# #     #     """
# #     #     # Each pit stop costs ~22s; tyre advantage is ~0.3s/lap
# #     #     pit_time_cost = self._pit_stops * 22.0
# #     #     tyre_advantage = self._tyre_state.wear * 3.0  # worn tyres = slower = lose positions
        
# #     #     # Estimate position from time costs relative to field
# #     #     base_pos = 3  # start mid-field
# #     #     pos_from_pits = base_pos + int(pit_time_cost / 22.0)
# #     #     pos_from_tyres = int(tyre_advantage * 2)
        
# #     #     estimated = base_pos + pos_from_pits + pos_from_tyres
        
# #     #     # Add small noise
# #     #     noise = int(self._rng.integers(-1, 2))
# #     #     return int(np.clip(estimated + noise, 1, self.n_opponents + 1))
# #     def _simulate_position_update(self) -> int:
# #         """
# #         Heuristic position model based on tyre state and pit strategy.
# #         Gives the agent a meaningful signal without cumulative time comparison.
# #         """
# #         # Base starting position (mid-field)
# #         pos = 3.0

# #         # --- Tyre wear penalty ---
# #         # As wear builds, we lose time to competitors on fresher rubber
# #         # Above 0.4 wear, positions start slipping
# #         if self._tyre_state.wear > 0.4:
# #             pos += (self._tyre_state.wear - 0.4) * 6.0  # up to +3.6 positions at full wear

# #         # --- No-stop penalty late in race ---
# #         # If we haven't pitted by lap 30, we're losing to pit-cycled cars
# #         if self._pit_stops == 0 and self._lap > 28:
# #             pos += (self._lap - 28) * 0.15

# #         # --- Pit stop cost (temporary) ---
# #         # In the lap immediately after pitting, we drop back from pit lane delta
# #         if self._laps_since_pit == 1:
# #             pos += 2.0  # just exited pit lane, temporarily behind

# #         # --- Fresh tyre recovery ---
# #         # After a stop, we recover positions as fresh tyres are faster
# #         elif 1 < self._laps_since_pit <= 8 and self._pit_stops >= 1:
# #             recovery = min(2.0, (self._laps_since_pit - 1) * 0.3)
# #             pos -= recovery

# #         # --- Over-pitting penalty ---
# #         if self._pit_stops > 2:
# #             pos += (self._pit_stops - 2) * 1.5

# #         # Small stochastic noise
# #         noise = self._rng.normal(0, 0.5)
# #         pos += noise

# #         return int(np.clip(round(pos), 1, self.n_opponents + 1))



# """
# F1 Strategy RL Environment — Fixed
# ====================================
# Key fix: position computed from lap-time gap to precomputed opponent trajectories.
# Pit deltas and tyre wear have genuine, observable effects on relative position.
# """
# from __future__ import annotations

# from typing import Any, Dict, Optional, Tuple
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces

# from src.core.tyre_model import TyreModel, TyreState, TyreCompoundParams
# from src.core.fuel_model import FuelModel, FuelModelParams
# from src.core.lap_time_model import LapTimeModel, LapTimeModelParams
# from src.core.safety_car_model import PoissonSafetyCarModel, SafetyCarModelParams
# from src.core.weather_model import WeatherModel, WeatherModelParams
# from src.simulation.pit_stop_model import PitStopModel, PitStopParams
# from src.rl.reward_function import RewardFunction
# from src.simulation.race_engine import TrackConfig

# COMPOUND_ENCODING = {"soft": 0.0, "medium": 0.5, "hard": 1.0}
# ACTION_TO_COMPOUND = {0: None, 1: "soft", 2: "medium", 3: "hard"}
# ACTION_NAMES       = {0: "stay_out", 1: "pit_soft", 2: "pit_medium", 3: "pit_hard"}


# def _precompute_opponent_times(
#     total_laps: int,
#     base_lap_time: float,
#     fuel_start: float,
#     pit_lap: int,
#     pace_offset: float = 0.0,
# ) -> np.ndarray:
#     """
#     Precompute cumulative race time for one opponent doing a single pit stop.
#     Returns array of shape (total_laps,) — cumulative time at end of each lap.
#     """
#     times = np.zeros(total_laps)
#     cum = 0.0
#     fuel = fuel_start
#     wear = 0.0
#     for lap in range(1, total_laps + 1):
#         fuel = max(0.0, fuel - 1.85)
#         fuel_pen = 0.034 * fuel
#         if lap == pit_lap:
#             # Pit stop lap: add pit lane delta
#             wear_pen = 0.08 * wear + 0.12 * wear ** 2
#             cum += base_lap_time + fuel_pen + wear_pen + 22.0 + pace_offset
#             wear = 0.0
#         else:
#             wear_pen = 0.08 * wear + 0.12 * wear ** 2
#             cum += base_lap_time + fuel_pen + wear_pen + pace_offset
#             wear = min(wear + 0.018 * 0.75, 1.0)
#         times[lap - 1] = cum
#     return times


# class F1StrategyEnv(gym.Env):
#     """
#     Single-driver F1 strategy environment.
#     Position is derived from lap-time gap to 5 precomputed opponent trajectories.
#     """
#     metadata = {"render_modes": []}

#     def __init__(
#         self,
#         track_config: TrackConfig,
#         compound_params: Dict[str, TyreCompoundParams],
#         n_opponents: int = 5,
#         lap_noise_std_s: float = 0.08,
#         seed: int = 42,
#     ) -> None:
#         super().__init__()
#         self.track          = track_config
#         self.compound_params = compound_params
#         self.n_opponents    = n_opponents
#         self.lap_noise_std_s = lap_noise_std_s

#         self.observation_space = spaces.Box(
#             low=np.zeros(8, dtype=np.float32),
#             high=np.ones(8, dtype=np.float32),
#             dtype=np.float32,
#         )
#         self.action_space = spaces.Discrete(4)

#         self._reward_fn  = RewardFunction(n_drivers=n_opponents + 1)
#         self._rng        = np.random.default_rng(seed)
#         self._pit_model  = PitStopModel(PitStopParams())
#         self._fuel_model = FuelModel(FuelModelParams(), track_config.fuel_load_kg_start)
#         self._sc_model   = PoissonSafetyCarModel(
#             SafetyCarModelParams(base_rate=track_config.safety_car_base_rate),
#             total_laps=track_config.total_laps,
#         )
#         self._weather = WeatherModel(WeatherModelParams())

#         # Precompute opponent cumulative times — each does a 1-stop at a different lap
#         pit_laps    = [20, 23, 26, 29, 32]
#         pace_offsets = [-0.10, -0.05, 0.0, 0.05, 0.10]
#         self._opponent_times = [
#             _precompute_opponent_times(
#                 track_config.total_laps,
#                 track_config.base_lap_time_s,
#                 track_config.fuel_load_kg_start,
#                 pit_lap=pit_laps[i],
#                 pace_offset=pace_offsets[i],
#             )
#             for i in range(n_opponents)
#         ]

#         # Episode state
#         self._lap            = 0
#         self._tyre_state     = None
#         self._fuel_kg        = track_config.fuel_load_kg_start
#         self._pit_stops      = 0
#         self._laps_since_pit = 0
#         self._sc_active      = False
#         self._position       = 3
#         self._our_cum_time   = 0.0   # cumulative race time including pit deltas

#     # ------------------------------------------------------------------
#     # Gymnasium interface
#     # ------------------------------------------------------------------

#     def reset(self, seed: Optional[int] = None, options=None) -> Tuple[np.ndarray, Dict]:
#         if seed is not None:
#             self._rng = np.random.default_rng(seed)

#         self._lap            = 0
#         self._tyre_state     = TyreState("medium", 0.0, 80.0, 0, True)
#         self._fuel_kg        = self.track.fuel_load_kg_start
#         self._fuel_model.reset(self.track.fuel_load_kg_start)
#         self._pit_stops      = 0
#         self._laps_since_pit = 0
#         self._sc_active      = False
#         self._our_cum_time   = 0.0
#         self._position       = 3
#         self._weather.reset()
#         return self._get_obs(), {}

#     def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
#         self._lap += 1
#         pit_compound = ACTION_TO_COMPOUND[int(action)]
#         is_pitting   = pit_compound is not None

#         # Safety car
#         sc_event = self._sc_model.sample_event(self._lap, self._rng)
#         self._sc_active = sc_event is not None

#         # Weather
#         self._weather.step(self._lap, self._rng)

#         # Pit stop
#         pit_delta = 0.0
#         if is_pitting:
#             pit_delta, _ = self._pit_model.sample_stop_time(self._rng)
#             self._tyre_state = TyreState(pit_compound, 0.0, 70.0, 0, True)
#             self._pit_stops      += 1
#             self._laps_since_pit  = 0
#         else:
#             self._laps_since_pit += 1

#         # Lap time from physics
#         c_params   = self.compound_params[self._tyre_state.compound]
#         tyre_model = TyreModel(c_params, self.track.track_abrasion)
#         lt_params  = LapTimeModelParams(
#             base_lap_time_s=self.track.base_lap_time_s,
#             track_evolution_rate=self.track.track_evolution_rate,
#             lap_time_noise_std_s=self.lap_noise_std_s,
#         )
#         lt_model  = LapTimeModel(lt_params, tyre_model, self._fuel_model)
#         lap_time  = lt_model.predict(
#             lap=self._lap,
#             tyre_state=self._tyre_state,
#             fuel_kg=self._fuel_kg,
#             safety_car_active=self._sc_active,
#             rng=self._rng,
#         )
#         if not self._sc_active:
#             lap_time += self._weather.lap_time_delta()

#         # Accumulate our total race time (lap time + pit delta)
#         self._our_cum_time += lap_time + pit_delta

#         # Advance tyre state
#         push = 0.7 if self._sc_active else 1.0
#         self._tyre_state = tyre_model.step(self._tyre_state, push_factor=push, rng=self._rng)

#         # Consume fuel
#         self._fuel_kg = max(0.0, self._fuel_kg - 1.85)

#         # Position: count how many opponents have lower cumulative time than us
#         lap_idx = self._lap - 1
#         opponents_ahead = sum(
#             1 for opp in self._opponent_times
#             if opp[lap_idx] < self._our_cum_time
#         )
#         self._position = int(np.clip(1 + opponents_ahead, 1, self.n_opponents + 1))

#         terminated = self._lap >= self.track.total_laps
#         truncated  = False

#         reward = self._reward_fn.compute(
#             lap=self._lap,
#             total_laps=self.track.total_laps,
#             position=self._position,
#             tyre_state=self._tyre_state,
#             is_terminal=terminated,
#             pit_stops=self._pit_stops,
#         )

#         info = {
#             "lap": self._lap,
#             "position": self._position,
#             "tyre_wear": self._tyre_state.wear,
#             "fuel_kg": self._fuel_kg,
#             "lap_time_s": lap_time,
#             "our_cum_time": self._our_cum_time,
#         }
#         return self._get_obs(), reward, terminated, truncated, info

#     # ------------------------------------------------------------------

#     def _get_obs(self) -> np.ndarray:
#         wear        = self._tyre_state.wear if self._tyre_state else 0.0
#         compound_enc = COMPOUND_ENCODING.get(
#             self._tyre_state.compound if self._tyre_state else "medium", 0.5
#         )
#         obs = np.array([
#             self._lap / self.track.total_laps,
#             wear,
#             compound_enc,
#             self._fuel_kg / self.track.fuel_load_kg_start,
#             self._position / (self.n_opponents + 1),
#             float(self._sc_active),
#             self._laps_since_pit / self.track.total_laps,
#             min(self._pit_stops / 3.0, 1.0),
#         ], dtype=np.float32)
#         return np.clip(obs, 0.0, 1.0)



from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.core.tyre_model import TyreModel, TyreState, TyreCompoundParams
from src.core.fuel_model import FuelModel, FuelModelParams
from src.core.lap_time_model import LapTimeModel, LapTimeModelParams
from src.core.safety_car_model import PoissonSafetyCarModel, SafetyCarModelParams
from src.core.weather_model import WeatherModel, WeatherModelParams
from src.simulation.pit_stop_model import PitStopModel, PitStopParams
from src.rl.reward_function import RewardFunction
from src.simulation.race_engine import TrackConfig

COMPOUND_ENCODING = {"soft": 0.0, "medium": 0.5, "hard": 1.0}
ACTION_TO_COMPOUND = {0: None, 1: "soft", 2: "medium", 3: "hard"}
ACTION_NAMES       = {0: "stay_out", 1: "pit_soft", 2: "pit_medium", 3: "pit_hard"}

# Minimum laps on a set before pitting is allowed
MIN_STINT_LAPS = 10


def _build_opponent_times(total_laps, base_lap_time, fuel_start, pit_lap, pace_delta=0.0):
    """Cumulative race time for one opponent doing a clean 1-stop."""
    cum, fuel, wear = 0.0, fuel_start, 0.0
    times = np.zeros(total_laps)
    for lap in range(1, total_laps + 1):
        fuel = max(0.0, fuel - 1.85)
        wear_pen = 0.10 * wear + 0.20 * wear ** 2
        fuel_pen = 0.034 * fuel
        lt = base_lap_time + wear_pen + fuel_pen + pace_delta
        if lap == pit_lap:
            lt += 22.0   # pit lane delta
            wear = 0.0
        else:
            wear = min(wear + 0.022, 1.0)
        cum += lt
        times[lap - 1] = cum
    return times


class F1StrategyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        track_config: TrackConfig,
        compound_params: Dict[str, TyreCompoundParams],
        n_opponents: int = 5,
        lap_noise_std_s: float = 0.05,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.track           = track_config
        self.compound_params = compound_params
        self.n_opponents     = n_opponents
        self.lap_noise_std_s = lap_noise_std_s

        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self._reward_fn  = RewardFunction(n_drivers=n_opponents + 1)
        self._rng        = np.random.default_rng(seed)
        self._pit_model  = PitStopModel(PitStopParams())
        self._fuel_model = FuelModel(FuelModelParams(), track_config.fuel_load_kg_start)
        self._sc_model   = PoissonSafetyCarModel(
            SafetyCarModelParams(base_rate=track_config.safety_car_base_rate),
            total_laps=track_config.total_laps,
        )
        self._weather = WeatherModel(WeatherModelParams())

        # Opponents pit at laps 20–32, with pace spread ±0.15s/lap around our baseline
        # Pace deltas slightly positive = they are a little slower than us on average
        pit_laps    = [20, 23, 26, 29, 32]
        pace_deltas = [0.05, 0.10, 0.15, 0.20, 0.25]   # opponents all slightly slower
        self._opp_times = [
            _build_opponent_times(
                track_config.total_laps,
                track_config.base_lap_time_s,
                track_config.fuel_load_kg_start,
                pit_lap=pit_laps[i],
                pace_delta=pace_deltas[i],
            )
            for i in range(n_opponents)
        ]

        self._reset_episode()

    def _reset_episode(self):
        self._lap            = 0
        self._tyre_state     = TyreState("medium", 0.0, 80.0, 0, True)
        self._fuel_kg        = self.track.fuel_load_kg_start
        self._fuel_model.reset(self.track.fuel_load_kg_start)
        self._pit_stops      = 0
        self._laps_since_pit = 0
        self._sc_active      = False
        self._position       = 1        # start at front (we are fast early on fresh tyres)
        self._our_cum_time   = 0.0
        self._weather.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_episode()
        return self._get_obs(), {}

    def step(self, action: int):
        self._lap += 1
        pit_compound = ACTION_TO_COMPOUND[int(action)]

        # --- Enforce minimum stint: ignore pit action if too early ---
        if pit_compound is not None and self._laps_since_pit < MIN_STINT_LAPS:
            pit_compound = None   # force stay_out

        is_pitting = pit_compound is not None

        # Safety car
        sc_event        = self._sc_model.sample_event(self._lap, self._rng)
        self._sc_active = sc_event is not None

        # Weather
        self._weather.step(self._lap, self._rng)

        # Pit stop
        pit_delta = 0.0
        if is_pitting:
            pit_delta, _ = self._pit_model.sample_stop_time(self._rng)
            self._tyre_state     = TyreState(pit_compound, 0.0, 70.0, 0, True)
            self._pit_stops     += 1
            self._laps_since_pit = 0
        else:
            self._laps_since_pit += 1

        # Physics lap time
        c_params   = self.compound_params[self._tyre_state.compound]
        tyre_model = TyreModel(c_params, self.track.track_abrasion)
        lt_params  = LapTimeModelParams(
            base_lap_time_s=self.track.base_lap_time_s,
            track_evolution_rate=self.track.track_evolution_rate,
            lap_time_noise_std_s=self.lap_noise_std_s,
        )
        lt_model = LapTimeModel(lt_params, tyre_model, self._fuel_model)
        lap_time = lt_model.predict(
            lap=self._lap,
            tyre_state=self._tyre_state,
            fuel_kg=self._fuel_kg,
            safety_car_active=self._sc_active,
            rng=self._rng,
        )
        if not self._sc_active:
            lap_time += self._weather.lap_time_delta()

        # Accumulate our race time
        self._our_cum_time += lap_time + pit_delta

        # Advance tyre and fuel
        push = 0.7 if self._sc_active else 1.0
        self._tyre_state = tyre_model.step(self._tyre_state, push_factor=push, rng=self._rng)
        self._fuel_kg    = max(0.0, self._fuel_kg - 1.85)

        # Position from cumulative time vs opponents
        lap_idx    = self._lap - 1
        ahead      = sum(1 for opp in self._opp_times if opp[lap_idx] < self._our_cum_time)
        self._position = int(np.clip(1 + ahead, 1, self.n_opponents + 1))

        terminated = self._lap >= self.track.total_laps
        reward = self._reward_fn.compute(
            lap=self._lap,
            total_laps=self.track.total_laps,
            position=self._position,
            tyre_state=self._tyre_state,
            is_terminal=terminated,
            pit_stops=self._pit_stops,
        )

        info = {
            "lap": self._lap,
            "position": self._position,
            "tyre_wear": self._tyre_state.wear,
            "fuel_kg": self._fuel_kg,
            "lap_time_s": lap_time,
        }
        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        wear         = self._tyre_state.wear if self._tyre_state else 0.0
        compound_enc = COMPOUND_ENCODING.get(
            self._tyre_state.compound if self._tyre_state else "medium", 0.5
        )
        obs = np.array([
            self._lap / self.track.total_laps,
            wear,
            compound_enc,
            self._fuel_kg / self.track.fuel_load_kg_start,
            self._position / (self.n_opponents + 1),
            float(self._sc_active),
            self._laps_since_pit / self.track.total_laps,
            min(self._pit_stops / 3.0, 1.0),
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)