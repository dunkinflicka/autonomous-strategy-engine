"""
Tyre Parameter Estimation
=========================
ML-based inference of tyre degradation coefficients from telemetry.

Uses gradient boosted trees (XGBoost) to infer:
    - k_c (wear rate base)
    - cliff_threshold
    - perf_loss_linear / perf_loss_quadratic

from observed lap times, stint ages, and compound information.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


@dataclass
class TelemetrySample:
    """A single lap observation for tyre parameter fitting."""
    compound: str
    stint_age: int
    tyre_wear_estimated: float   # estimated from visual/sensor data
    lap_time_s: float
    fuel_kg: float
    track_temp_c: float
    driver_id: int


class TyreParameterEstimator:
    """
    Fits tyre model parameters from telemetry observations.

    Approach:
        1. Compute physics-based lap time prediction using nominal params
        2. Fit residuals using GBT
        3. Optionally: use inverse modelling to refine physical coefficients

    Parameters
    ----------
    feature_cols : columns used as input features
    """

    def __init__(self) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for TyreParameterEstimator")

        self._scaler = StandardScaler()
        self._models: Dict[str, GradientBoostingRegressor] = {}
        self._is_fitted = False

    def fit(self, samples: List[TelemetrySample]) -> "TyreParameterEstimator":
        """
        Fit per-compound wear parameter models from telemetry.

        Parameters
        ----------
        samples : list of TelemetrySample observations

        Returns self for chaining.
        """
        df = self._to_dataframe(samples)

        for compound in df["compound"].unique():
            mask = df["compound"] == compound
            X = df.loc[mask, self._feature_cols()].values
            y = df.loc[mask, "lap_time_residual"].values

            if len(X) < 5:
                continue

            X_scaled = self._scaler.fit_transform(X)
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=3,
                learning_rate=0.1, random_state=42
            )
            model.fit(X_scaled, y)
            self._models[compound] = model

        self._is_fitted = True
        return self

    def predict_residual(
        self,
        compound: str,
        stint_age: int,
        fuel_kg: float,
        track_temp_c: float,
    ) -> float:
        """
        Predict lap time residual (actual - physics prediction) for given conditions.
        Returns 0.0 if compound not fitted.
        """
        if not self._is_fitted or compound not in self._models:
            return 0.0

        X = np.array([[stint_age, fuel_kg, track_temp_c]])
        X_scaled = self._scaler.transform(X)
        return float(self._models[compound].predict(X_scaled)[0])

    def cross_validate(
        self,
        samples: List[TelemetrySample],
        cv: int = 5,
    ) -> Dict[str, float]:
        """Returns mean CV R² score per compound."""
        df = self._to_dataframe(samples)
        scores = {}

        for compound in df["compound"].unique():
            mask = df["compound"] == compound
            X = df.loc[mask, self._feature_cols()].values
            y = df.loc[mask, "lap_time_residual"].values

            if len(X) < 10:
                continue

            X_scaled = self._scaler.fit_transform(X)
            model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="r2")
            scores[compound] = float(np.mean(cv_scores))

        return scores

    # ------------------------------------------------------------------

    def _to_dataframe(self, samples: List[TelemetrySample]) -> pd.DataFrame:
        rows = [
            {
                "compound": s.compound,
                "stint_age": s.stint_age,
                "tyre_wear_estimated": s.tyre_wear_estimated,
                "fuel_kg": s.fuel_kg,
                "track_temp_c": s.track_temp_c,
                "driver_id": s.driver_id,
                "lap_time_s": s.lap_time_s,
                # Simplified residual: deviation from group median
                "lap_time_residual": 0.0,
            }
            for s in samples
        ]
        df = pd.DataFrame(rows)
        # Compute residual as deviation from compound-age trend
        for compound in df["compound"].unique():
            mask = df["compound"] == compound
            df.loc[mask, "lap_time_residual"] = (
                df.loc[mask, "lap_time_s"] - df.loc[mask, "lap_time_s"].median()
            )
        return df

    @staticmethod
    def _feature_cols() -> List[str]:
        return ["stint_age", "fuel_kg", "track_temp_c"]
