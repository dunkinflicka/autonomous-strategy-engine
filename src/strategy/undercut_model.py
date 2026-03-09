"""
Undercut / Overcut Model
========================
Analytical model for evaluating undercut and overcut opportunities.

The undercut works when:
    fresh_tyre_advantage × remaining_laps_on_old_tyre > pit_stop_time_loss

We formalise this as an expected time gain calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class UndercutOpportunity:
    driver_id: int
    opponent_id: int
    current_lap: int
    recommended_pit_lap: int
    expected_time_gain_s: float
    confidence: float           # [0, 1]
    strategy_type: str          # 'undercut' or 'overcut'


class UndercutModel:
    """
    Evaluates undercut and overcut opportunities given current race state.

    Parameters
    ----------
    pit_lane_delta_s     : expected time lost in pit lane
    tyre_advantage_per_lap : seconds per lap advantage of fresh vs worn tyre
    """

    def __init__(
        self,
        pit_lane_delta_s: float = 22.0,
        tyre_advantage_per_lap: float = 0.3,
    ) -> None:
        self.pit_lane_delta_s = pit_lane_delta_s
        self.tyre_advantage_per_lap = tyre_advantage_per_lap

    def evaluate_undercut(
        self,
        driver_gap_s: float,
        opponent_laps_since_pit: int,
        opponent_predicted_pit_lap: int,
        current_lap: int,
        total_laps: int,
        driver_id: int = 0,
        opponent_id: int = 1,
    ) -> UndercutOpportunity:
        """
        Evaluate whether an undercut is profitable.

        An undercut gains time if:
            (advantage_per_lap × stint_length) > pit_lane_delta + driver_gap

        Parameters
        ----------
        driver_gap_s              : current gap behind opponent (positive = behind)
        opponent_laps_since_pit   : how many laps opponent has been on current set
        opponent_predicted_pit_lap: lap on which opponent is expected to pit
        current_lap               : current race lap
        total_laps                : total race laps
        """
        laps_until_opponent_pits = max(1, opponent_predicted_pit_lap - current_lap)
        remaining_race_laps = total_laps - opponent_predicted_pit_lap

        # Fresh tyre advantage over old tyre (opponent's worn rubber)
        tyre_age_diff = opponent_laps_since_pit
        worn_tyre_handicap = self.tyre_advantage_per_lap * min(tyre_age_diff, 20)

        # Expected time gain from undercut:
        # We pit now, opponent pits in N laps.
        # During those N laps, we gain worn_tyre_handicap per lap vs opponent.
        # But we lose pit_lane_delta immediately.
        undercut_gain = (worn_tyre_handicap * laps_until_opponent_pits) - self.pit_lane_delta_s

        # Only viable if we can close the gap
        net_gain = undercut_gain - driver_gap_s
        confidence = float(np.clip(net_gain / (self.pit_lane_delta_s * 0.5), 0, 1))

        return UndercutOpportunity(
            driver_id=driver_id,
            opponent_id=opponent_id,
            current_lap=current_lap,
            recommended_pit_lap=current_lap,
            expected_time_gain_s=net_gain,
            confidence=confidence,
            strategy_type="undercut" if net_gain > 0 else "stay_out",
        )

    def evaluate_overcut(
        self,
        opponent_pit_lap: int,
        current_lap: int,
        driver_tyre_wear: float,
        driver_id: int = 0,
        opponent_id: int = 1,
    ) -> UndercutOpportunity:
        """
        Evaluate whether an overcut is viable.
        The overcut works when the driver can lap faster than the opponent
        during the opponent's out-lap (typically 1-3 laps of slow out-lap).
        """
        opponent_out_lap_loss = 2.0  # seconds lost on out lap from cold tyres
        free_laps_on_track = max(0, opponent_pit_lap - current_lap)

        # If we're on old tyres but tyre wear isn't at cliff, we can push
        cliff_risk = max(0.0, driver_tyre_wear - 0.65) * 5.0  # increasing risk > cliff
        net_gain = opponent_out_lap_loss * free_laps_on_track - cliff_risk

        confidence = float(np.clip(net_gain / 3.0, 0, 1))

        return UndercutOpportunity(
            driver_id=driver_id,
            opponent_id=opponent_id,
            current_lap=current_lap,
            recommended_pit_lap=opponent_pit_lap + 2,
            expected_time_gain_s=net_gain,
            confidence=confidence,
            strategy_type="overcut" if net_gain > 0 else "stay_out",
        )
