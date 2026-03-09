"""
Lap Time Residual Model
=======================
Models the difference between physics-predicted and actual lap times.

    LapTime_actual = LapTime_physics + ML_residual

This captures:
  - driver-specific style effects
  - track evolution not captured by the linear model
  - wind, temperature deviations
  - errors in the nominal physics model

Supported backends: GradientBoostingRegressor, GaussianProcess
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple
import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class ResidualModelType(Enum):
    GBT = "gradient_boosted_trees"
    GP  = "gaussian_process"


class LapTimeResidualModel:
    """
    Fits and predicts lap time residuals to complement the physics model.

    Features: [lap, tyre_wear, fuel_kg, track_temp_c, stint_age, compound_enc]
    Target:   lap_time_residual_s
    """

    def __init__(
        self,
        model_type: ResidualModelType = ResidualModelType.GBT,
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")

        self.model_type = model_type
        self._scaler = StandardScaler()
        self._model = self._build_model()
        self._is_fitted = False

    def fit(
        self,
        features: np.ndarray,
        residuals: np.ndarray,
    ) -> "LapTimeResidualModel":
        """
        Fit the residual model.

        Parameters
        ----------
        features  : shape (n_samples, 6) — see feature order above
        residuals : shape (n_samples,) — observed lap time residuals
        """
        X_scaled = self._scaler.fit_transform(features)
        self._model.fit(X_scaled, residuals)
        self._is_fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict residuals for new observations.
        Returns zeros if not fitted.
        """
        if not self._is_fitted:
            return np.zeros(len(features))

        X_scaled = self._scaler.transform(features)
        return self._model.predict(X_scaled)

    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        For GP model, returns (mean, std). For GBT returns (mean, None).
        """
        if not self._is_fitted:
            return np.zeros(len(features)), None

        X_scaled = self._scaler.transform(features)

        if self.model_type == ResidualModelType.GP:
            mean, std = self._model.predict(X_scaled, return_std=True)
            return mean, std
        else:
            return self._model.predict(X_scaled), None

    # ------------------------------------------------------------------

    def _build_model(self):
        if self.model_type == ResidualModelType.GBT:
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        elif self.model_type == ResidualModelType.GP:
            kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
            return GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-4,
                n_restarts_optimizer=3,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def make_features(
    lap: int,
    tyre_wear: float,
    fuel_kg: float,
    track_temp_c: float,
    stint_age: int,
    compound: str,
) -> np.ndarray:
    """Construct feature vector for residual model prediction."""
    compound_enc = {"soft": 0.0, "medium": 0.5, "hard": 1.0}.get(compound, 0.5)
    return np.array([[lap, tyre_wear, fuel_kg, track_temp_c, stint_age, compound_enc]])
