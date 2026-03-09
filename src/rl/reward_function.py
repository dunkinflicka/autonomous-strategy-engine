"""
Reward Function
===============
Defines the reward signal for the RL strategy agent.

Design principles:
  - Large terminal reward based on finishing position
  - Small per-lap shaping terms to guide exploration
  - Penalty for tyre cliff abuse
  - Bonus for safety car exploitation
"""
from __future__ import annotations

import numpy as np
from src.core.tyre_model import TyreState


# F1 championship points per position
POINTS_MAP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
              6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


class RewardFunction:
    """
    Configurable reward function for the F1 strategy environment.

    Parameters
    ----------
    n_drivers           : total drivers in the race
    terminal_scale      : multiplier for the terminal position reward
    cliff_penalty       : per-lap penalty when tyre is beyond cliff
    sc_pit_bonus        : bonus for pitting under safety car
    per_lap_shaping     : small negative shaping per lap (encourages speed)
    """

    def __init__(
        self,
        n_drivers: int = 20,
        terminal_scale: float = 1.0,
        cliff_penalty: float = 0.05,
        sc_pit_bonus: float = 0.2,
        per_lap_shaping: float = -0.01,
    ) -> None:
        self.n_drivers = n_drivers
        self.terminal_scale = terminal_scale
        self.cliff_penalty = cliff_penalty
        self.sc_pit_bonus = sc_pit_bonus
        self.per_lap_shaping = per_lap_shaping

    # def compute(
    #     self,
    #     lap: int,
    #     total_laps: int,
    #     position: int,
    #     tyre_state: TyreState,
    #     is_terminal: bool,
    #     pit_stops: int,
    #     pitting_under_sc: bool = False,
    # ) -> float:
    #     reward = 0.0

    #     # --- Terminal reward ---
    #     if is_terminal:
    #         # Normalised position reward: P1 = 1.0, last = 0.0
    #         pos_reward = (self.n_drivers - position) / (self.n_drivers - 1)
    #         # Championship points bonus (harder signal)
    #         pts = POINTS_MAP.get(position, 0)
    #         points_reward = pts / 25.0  # normalise to [0, 1]
    #         reward += self.terminal_scale * (pos_reward + points_reward * 0.5)

    #     # --- Tyre cliff penalty ---
    #     if tyre_state.wear >= 0.72:  # generic cliff threshold
    #         cliff_excess = (tyre_state.wear - 0.72) / (1.0 - 0.72)
    #         reward -= self.cliff_penalty * cliff_excess

    #             # --- Excessive pit stop penalty ---
    #     if pit_stops > 3:
    #         reward -= 0.15 * (pit_stops - 3)   # heavy penalty beyond 3 stops
    #     elif pit_stops > 1:
    #         reward -= 0.02 * pit_stops          # mild penalty for each stop

    #     # --- Safety car pit bonus ---
    #     if pitting_under_sc:
    #         reward += self.sc_pit_bonus

    #     # --- Per-lap shaping ---
    #     reward += self.per_lap_shaping

    #     return float(reward)
    def compute(self, lap, total_laps, position, tyre_state, is_terminal, pit_stops,
            pitting_under_sc=False):
        reward = 0.0

        # Terminal reward — this is all that matters
        if is_terminal:
            pos_reward = (self.n_drivers - position) / (self.n_drivers - 1)
            pts = POINTS_MAP.get(position, 0)
            reward += self.terminal_scale * (pos_reward + (pts / 25.0) * 0.5)

        # Tyre cliff penalty only (discourages running into cliff)
        if tyre_state.wear >= 0.72:
            cliff_excess = (tyre_state.wear - 0.72) / (1.0 - 0.72)
            reward -= self.cliff_penalty * cliff_excess

        # SC pit bonus
        if pitting_under_sc:
            reward += self.sc_pit_bonus

        # Tiny per-lap shaping
        reward += self.per_lap_shaping

        return float(reward)
    def terminal_reward_from_position(self, position: int) -> float:
        """Convenience: compute terminal reward only (for evaluation)."""
        pos_reward = (self.n_drivers - position) / (self.n_drivers - 1)
        pts = POINTS_MAP.get(position, 0)
        return self.terminal_scale * (pos_reward + (pts / 25.0) * 0.5)
