"""
Monte Carlo Race Simulation Engine
====================================
Executes N race simulations and aggregates probabilistic outcomes.

Target: 10,000 – 100,000 simulations.
Uses independent seeded sub-RNG per simulation for full reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

from src.simulation.race_engine import RaceEngine, DriverStrategy
from src.simulation.race_state import DriverRaceState


@dataclass
class MonteCarloResult:
    """Aggregated probabilistic outcomes from N simulations."""
    n_simulations: int
    driver_ids: List[int]

    # Raw results per simulation [n_sims x n_drivers] – finishing position
    position_matrix: np.ndarray          # shape (n_sims, n_drivers)
    race_time_matrix: np.ndarray         # shape (n_sims, n_drivers)

    def win_probability(self, driver_id: int) -> float:
        idx = self.driver_ids.index(driver_id)
        return float(np.mean(self.position_matrix[:, idx] == 1))

    def podium_probability(self, driver_id: int) -> float:
        idx = self.driver_ids.index(driver_id)
        return float(np.mean(self.position_matrix[:, idx] <= 3))

    def expected_position(self, driver_id: int) -> float:
        idx = self.driver_ids.index(driver_id)
        return float(np.mean(self.position_matrix[:, idx]))

    def position_variance(self, driver_id: int) -> float:
        idx = self.driver_ids.index(driver_id)
        return float(np.var(self.position_matrix[:, idx]))

    def position_distribution(self, driver_id: int) -> Dict[int, float]:
        """Returns {position: probability} for all observed positions."""
        idx = self.driver_ids.index(driver_id)
        positions = self.position_matrix[:, idx]
        n_drivers = len(self.driver_ids)
        return {
            pos: float(np.mean(positions == pos))
            for pos in range(1, n_drivers + 1)
        }

    def expected_points(self, driver_id: int,
                         points_system: Optional[Dict[int, float]] = None) -> float:
        """Expected championship points using standard F1 scoring."""
        if points_system is None:
            points_system = {
                1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
                6: 8, 7: 6, 8: 4, 9: 2, 10: 1
            }
        dist = self.position_distribution(driver_id)
        return sum(
            prob * points_system.get(pos, 0)
            for pos, prob in dist.items()
        )

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Human-readable summary keyed by driver_id."""
        return {
            str(did): {
                "win_probability": self.win_probability(did),
                "podium_probability": self.podium_probability(did),
                "expected_position": self.expected_position(did),
                "position_variance": self.position_variance(did),
                "expected_points": self.expected_points(did),
            }
            for did in self.driver_ids
        }


class MonteCarloEngine:
    """
    Executes multiple race simulations and returns aggregate statistics.

    Parameters
    ----------
    race_engine      : RaceEngine — the deterministic-core simulator
    n_simulations    : number of Monte Carlo rollouts
    base_seed        : master seed for reproducibility
    show_progress    : whether to display tqdm progress bar
    """

    def __init__(
        self,
        race_engine: RaceEngine,
        n_simulations: int = 10_000,
        base_seed: int = 42,
        show_progress: bool = True,
    ) -> None:
        self.race_engine = race_engine
        self.n_simulations = n_simulations
        self.base_seed = base_seed
        self.show_progress = show_progress

    def run(
        self,
        strategies: List[DriverStrategy],
        ml_residual_fn: Optional[Callable] = None,
    ) -> MonteCarloResult:
        """
        Execute n_simulations races and return aggregated results.

        Each simulation uses an independent seeded Generator derived
        from base_seed + simulation_index — fully reproducible.
        """
        n_drivers = len(strategies)
        driver_ids = [s.driver_id for s in strategies]

        position_matrix  = np.zeros((self.n_simulations, n_drivers), dtype=np.int32)
        race_time_matrix = np.zeros((self.n_simulations, n_drivers), dtype=np.float64)

        iterator = range(self.n_simulations)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Monte Carlo simulations", unit="sim")

        for sim_idx in iterator:
            rng = np.random.default_rng(self.base_seed + sim_idx)

            final_states = self.race_engine.simulate_race(
                strategies=strategies,
                rng=rng,
                ml_residual_fn=ml_residual_fn,
            )

            # Map results to matrix
            id_to_idx = {did: i for i, did in enumerate(driver_ids)}
            for driver_state in final_states:
                col = id_to_idx[driver_state.driver_id]
                position_matrix[sim_idx, col]  = driver_state.position
                race_time_matrix[sim_idx, col] = driver_state.cumulative_time_s

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            driver_ids=driver_ids,
            position_matrix=position_matrix,
            race_time_matrix=race_time_matrix,
        )

    def compare_strategies(
        self,
        strategy_sets: List[List[DriverStrategy]],
        strategy_names: Optional[List[str]] = None,
        driver_of_interest: int = 0,
    ) -> Dict[str, MonteCarloResult]:
        """
        Compare multiple strategy variants for a driver.

        Parameters
        ----------
        strategy_sets        : list of strategy lists (one per scenario)
        strategy_names       : display names for each strategy set
        driver_of_interest   : driver_id to focus comparison on

        Returns
        -------
        Dict mapping strategy_name -> MonteCarloResult
        """
        if strategy_names is None:
            strategy_names = [f"strategy_{i}" for i in range(len(strategy_sets))]

        results: Dict[str, MonteCarloResult] = {}
        for name, strats in zip(strategy_names, strategy_sets):
            print(f"\nRunning: {name}")
            results[name] = self.run(strats)

        return results
