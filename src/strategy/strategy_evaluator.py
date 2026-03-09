"""
Strategy Evaluator
==================
Compares strategies using Monte Carlo results.
Produces a ranking with expected points, win probabilities, and risk metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from src.simulation.monte_carlo import MonteCarloResult, MonteCarloEngine
from src.simulation.race_engine import DriverStrategy, RaceEngine
from src.strategy.rule_based_strategy import (
    one_stop_strategy, two_stop_strategy, undercut_strategy, overcut_strategy
)


@dataclass
class StrategyReport:
    strategy_name: str
    driver_id: int
    win_probability: float
    podium_probability: float
    expected_position: float
    position_variance: float
    expected_points: float
    p10_position: float   # 10th percentile finishing position (optimistic)
    p90_position: float   # 90th percentile finishing position (pessimistic)
    sharpe_ratio: float   # expected_points / std(position) — risk-adjusted score

    def __str__(self) -> str:
        return (
            f"[{self.strategy_name}] "
            f"Win: {self.win_probability:.1%} | "
            f"Podium: {self.podium_probability:.1%} | "
            f"E[Pos]: {self.expected_position:.2f} | "
            f"E[Pts]: {self.expected_points:.1f} | "
            f"Sharpe: {self.sharpe_ratio:.2f}"
        )


class StrategyEvaluator:
    """
    High-level strategy comparison tool.
    Wraps the Monte Carlo engine to produce StrategyReport objects.
    """

    def __init__(
        self,
        race_engine: RaceEngine,
        n_simulations: int = 10_000,
        base_seed: int = 42,
    ) -> None:
        self.mc_engine = MonteCarloEngine(
            race_engine, n_simulations=n_simulations, base_seed=base_seed
        )

    def evaluate(
        self,
        strategy: DriverStrategy,
        competitor_strategies: List[DriverStrategy],
        strategy_name: str = "candidate",
    ) -> StrategyReport:
        """
        Evaluate a single strategy against a field of competitors.
        """
        all_strategies = [strategy] + competitor_strategies
        result = self.mc_engine.run(all_strategies)
        return self._build_report(result, strategy.driver_id, strategy_name)

    def compare_all(
        self,
        strategy_variants: Dict[str, DriverStrategy],
        competitor_strategies: List[DriverStrategy],
    ) -> List[StrategyReport]:
        """
        Compare multiple strategy options for the same driver.
        Returns reports sorted by expected points (descending).
        """
        reports = []
        for name, strat in strategy_variants.items():
            report = self.evaluate(strat, competitor_strategies, strategy_name=name)
            reports.append(report)

        return sorted(reports, key=lambda r: r.expected_points, reverse=True)

    def standard_strategy_comparison(
        self,
        driver_id: int,
        total_laps: int,
        competitor_strategies: List[DriverStrategy],
    ) -> List[StrategyReport]:
        """
        Run all standard strategy types and compare.
        Useful for baseline strategy exploration.
        """
        variants = {
            "1-stop (M→H)": one_stop_strategy(driver_id, total_laps,
                                               opening_compound="medium",
                                               closing_compound="hard"),
            "1-stop (S→H)": one_stop_strategy(driver_id, total_laps,
                                               opening_compound="soft",
                                               closing_compound="hard"),
            "2-stop (S→M→H)": two_stop_strategy(driver_id, total_laps),
            "2-stop (M→S→H)": two_stop_strategy(driver_id, total_laps,
                                                  compounds=["medium", "soft", "hard"]),
            "undercut vs P2": undercut_strategy(driver_id,
                                                 opponent_pit_lap=int(total_laps * 0.45),
                                                 total_laps=total_laps),
        }
        return self.compare_all(variants, competitor_strategies)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_report(
        self,
        result: MonteCarloResult,
        driver_id: int,
        strategy_name: str,
    ) -> StrategyReport:
        idx = result.driver_ids.index(driver_id)
        positions = result.position_matrix[:, idx]

        expected_pts = result.expected_points(driver_id)
        std_pos = float(np.std(positions))
        sharpe = expected_pts / (std_pos + 1e-6)

        return StrategyReport(
            strategy_name=strategy_name,
            driver_id=driver_id,
            win_probability=result.win_probability(driver_id),
            podium_probability=result.podium_probability(driver_id),
            expected_position=result.expected_position(driver_id),
            position_variance=result.position_variance(driver_id),
            expected_points=expected_pts,
            p10_position=float(np.percentile(positions, 10)),
            p90_position=float(np.percentile(positions, 90)),
            sharpe_ratio=sharpe,
        )
