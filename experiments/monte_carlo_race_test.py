"""
Monte Carlo Race Simulation Test
==================================
Validates the Monte Carlo engine at scale and reports convergence.

Usage:
    python -m experiments.monte_carlo_race_test
"""
from __future__ import annotations

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.simulation.race_engine import RaceEngine, DriverStrategy, TrackConfig
from src.simulation.monte_carlo import MonteCarloEngine
from src.core.tyre_model import TyreCompoundParams


COMPOUND_PARAMS = {
    "soft":   TyreCompoundParams("soft",   0.028, 0.15, 90.0, 0.008, 0.12, 0.18, 0.72, 2.8, 25, -0.9),
    "medium": TyreCompoundParams("medium", 0.018, 0.10, 85.0, 0.006, 0.08, 0.12, 0.78, 2.3, 35,  0.0),
    "hard":   TyreCompoundParams("hard",   0.012, 0.07, 80.0, 0.005, 0.05, 0.08, 0.85, 1.8, 50,  0.7),
}

TRACK = TrackConfig(
    name="silverstone", total_laps=52, base_lap_time_s=89.0,
    fuel_load_kg_start=105.0, track_abrasion=0.75,
    track_evolution_rate=0.0025, overtaking_difficulty=0.5,
    safety_car_base_rate=0.22,
)


def run_convergence_test(n_sims: int):
    engine = RaceEngine(TRACK, COMPOUND_PARAMS)
    strategies = [
        DriverStrategy(0, pit_laps=[26], compounds=["medium", "hard"]),
        DriverStrategy(1, pit_laps=[28], compounds=["soft",   "hard"]),
        DriverStrategy(2, pit_laps=[17, 36], compounds=["soft", "medium", "hard"]),
    ]
    mc = MonteCarloEngine(engine, n_simulations=n_sims, base_seed=42, show_progress=True)

    t0 = time.perf_counter()
    result = mc.run(strategies)
    elapsed = time.perf_counter() - t0

    print(f"\n{'─'*55}")
    print(f"  {n_sims:,} simulations | {elapsed:.2f}s | {n_sims/elapsed:.0f} sim/s")
    print(f"{'─'*55}")

    summary = result.summary()
    for did, stats in summary.items():
        strat_name = ["1-stop M", "1-stop S→H", "2-stop"][int(did)]
        print(f"  Driver {did} ({strat_name}):")
        print(f"    Win:    {stats['win_probability']:.1%}")
        print(f"    Podium: {stats['podium_probability']:.1%}")
        print(f"    E[Pos]: {stats['expected_position']:.2f}")
        print(f"    E[Pts]: {stats['expected_points']:.1f}")
    print()


if __name__ == "__main__":
    for n in [500, 2000]:
        run_convergence_test(n)
