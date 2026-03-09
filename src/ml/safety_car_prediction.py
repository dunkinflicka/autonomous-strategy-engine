"""
Safety Car Prediction (ML)
==========================
Trains a probabilistic model to estimate P(safety_car | lap, track, conditions).
Intended to calibrate the logistic safety car model from historical data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


@dataclass
class SafetyCarObservation:
    """One lap observation for safety car model training."""
    circuit: str
    lap: int
    total_laps: int
    track_abrasion: float
    is_wet: bool
    safety_car_occurred: bool    # label


class SafetyCarPredictor:
    """
    Logistic regression model for safety car probability estimation.
    Calibrated coefficients can be passed back to LogisticSafetyCarModel.
    """

    def __init__(self) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")
        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=500, random_state=42
        )
        self._is_fitted = False

    def fit(self, observations: List[SafetyCarObservation]) -> "SafetyCarPredictor":
        X, y = self._to_arrays(observations)
        if len(np.unique(y)) < 2:
            raise ValueError("Training data must contain both positive and negative examples")
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)
        self._is_fitted = True
        return self

    def predict_proba(
        self,
        lap: int,
        total_laps: int,
        track_abrasion: float,
        is_wet: bool,
    ) -> float:
        """Returns P(safety car occurs on this lap)."""
        if not self._is_fitted:
            return 0.0
        X = np.array([[lap / total_laps, track_abrasion, float(is_wet)]])
        X_scaled = self._scaler.transform(X)
        return float(self._model.predict_proba(X_scaled)[0, 1])

    def get_calibrated_logistic_coefficients(self) -> np.ndarray:
        """
        Returns [intercept, lap_coeff, abrasion_coeff, wet_coeff]
        for use in LogisticSafetyCarModel.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        coef = self._model.coef_[0]
        intercept = self._model.intercept_[0]
        # Prepend intercept to match LogisticSafetyCarModel convention
        return np.array([intercept, coef[0], coef[1], coef[2]])

    def evaluate(self, observations: List[SafetyCarObservation]) -> Dict[str, float]:
        """Returns AUC-ROC and accuracy on provided observations."""
        X, y = self._to_arrays(observations)
        X_scaled = self._scaler.transform(X)
        y_pred = self._model.predict(X_scaled)
        y_proba = self._model.predict_proba(X_scaled)[:, 1]
        return {
            "accuracy": float(np.mean(y == y_pred)),
            "auc_roc": float(roc_auc_score(y, y_proba)),
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _to_arrays(
        observations: List[SafetyCarObservation],
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array([
            [o.lap / o.total_laps, o.track_abrasion, float(o.is_wet)]
            for o in observations
        ])
        y = np.array([int(o.safety_car_occurred) for o in observations])
        return X, y


def generate_synthetic_training_data(
    n_races: int = 100,
    total_laps: int = 53,
    sc_base_rate: float = 0.20,
    seed: int = 42,
) -> List[SafetyCarObservation]:
    """
    Generate synthetic safety car observations for testing/demonstration.
    Uses a logistic data-generating process.
    """
    rng = np.random.default_rng(seed)
    observations = []

    for _ in range(n_races):
        abrasion = float(rng.uniform(0.4, 0.9))
        is_wet   = rng.random() < 0.25

        for lap in range(1, total_laps + 1):
            # True probability from logistic process
            logit = (-2.5
                     + 0.008 * (lap / total_laps)
                     + 0.4  * abrasion
                     + 0.6  * float(is_wet))
            prob = 1 / (1 + np.exp(-logit))
            # Scale to per-lap rate
            p_lap = 1 - (1 - sc_base_rate) ** (1 / total_laps) * (1 - prob * 0.1)
            occurred = rng.random() < p_lap

            observations.append(SafetyCarObservation(
                circuit="synthetic",
                lap=lap,
                total_laps=total_laps,
                track_abrasion=abrasion,
                is_wet=is_wet,
                safety_car_occurred=bool(occurred),
            ))

    return observations
