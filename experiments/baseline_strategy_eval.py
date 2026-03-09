"""
Baseline Strategy Evaluation Experiment
========================================
Compares standard 1-stop, 2-stop, undercut, and overcut strategies at Monza
using Monte Carlo simulation across 10,000 races.

Usage:
    python -m experiments.baseline_strategy_eval
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.race_engine import RaceEngine, DriverStrategy, TrackConfig
from src.simulation.monte_carlo import MonteCarloEngine
from src.strategy.strategy_evaluator import StrategyEvaluator
from src.strategy.rule_based_strategy import (
    one_stop_strategy, two_stop_strategy, undercut_strategy
)
from src.core.tyre_model import TyreCompoundParams


COMPOUND_PARAMS = {
    "soft": TyreCompoundParams("soft", 0.028, 0.15, 90.0, 0.008, 0.12, 0.18,
                                0.72, 2.8, 25, -0.9),
    "medium": TyreCompoundParams("medium", 0.018, 0.10, 85.0, 0.006, 0.08, 0.12,
                                  0.78, 2.3, 35, 0.0),
    "hard": TyreCompoundParams("hard", 0.012, 0.07, 80.0, 0.005, 0.05, 0.08,
                                0.85, 1.8, 50, 0.7),
}

MONZA = TrackConfig(
    name="monza",
    total_laps=53,
    base_lap_time_s=80.5,
    fuel_load_kg_start=105.0,
    track_abrasion=0.55,
    track_evolution_rate=0.003,
    overtaking_difficulty=0.3,
    safety_car_base_rate=0.18,
    safety_car_laps_mean=4.5,
    safety_car_laps_std=1.5,
)


def main():
    print("=" * 65)
    print("F1 Strategy Simulator — Baseline Evaluation (Monza)")
    print("=" * 65)

    engine = RaceEngine(MONZA, COMPOUND_PARAMS)
    evaluator = StrategyEvaluator(engine, n_simulations=2000, base_seed=42)

    DRIVER_ID = 0

    # Define 5 competitor strategies (fixed)
    competitors = [
        DriverStrategy(i + 1, pit_laps=[25], compounds=["medium", "hard"])
        for i in range(5)
    ]

    # Candidate strategies
    strategies = {
        "1-stop M→H (lap 24)": one_stop_strategy(DRIVER_ID, 53, pit_lap=24,
                                                    opening_compound="medium",
                                                    closing_compound="hard"),
        "1-stop S→H (lap 20)": one_stop_strategy(DRIVER_ID, 53, pit_lap=20,
                                                    opening_compound="soft",
                                                    closing_compound="hard"),
        "2-stop S→M→H":        two_stop_strategy(DRIVER_ID, 53),
        "Undercut vs lap 25":  undercut_strategy(DRIVER_ID, 25, 53, -4),
    }

    reports = evaluator.compare_all(strategies, competitors)

    print(f"\n{'Rank':<5} {'Strategy':<25} {'Win%':>6} {'Podium%':>8} "
          f"{'E[Pos]':>7} {'E[Pts]':>7} {'Sharpe':>7}")
    print("-" * 65)

    for rank, report in enumerate(reports, 1):
        print(f"{rank:<5} {report.strategy_name:<25} "
              f"{report.win_probability:>6.1%} "
              f"{report.podium_probability:>8.1%} "
              f"{report.expected_position:>7.2f} "
              f"{report.expected_points:>7.1f} "
              f"{report.sharpe_ratio:>7.2f}")

    print("\nRecommended strategy:", reports[0].strategy_name)


if __name__ == "__main__":
    main()
