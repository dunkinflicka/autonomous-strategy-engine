"""
Policy Inference
================
Load and query a trained RL policy for strategy recommendations.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

from src.rl.environment import F1StrategyEnv, ACTION_TO_COMPOUND
from src.core.tyre_model import TyreState


ACTION_NAMES = {0: "stay_out", 1: "pit_soft", 2: "pit_medium", 3: "pit_hard"}


class PolicyInference:
    """
    Wraps a trained SB3 model for strategy recommendations.
    Can also run deterministic rollouts for evaluation.
    """

    def __init__(self, model_path: str) -> None:
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError("stable-baselines3 required")
        self.model = PPO.load(model_path)
        print(f"Policy loaded from {model_path}")

    def recommend_action(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, str, np.ndarray]:
        """
        Returns (action_id, action_name, action_probabilities).
        """
        action, _ = self.model.predict(obs, deterministic=deterministic)
        action = int(action)
        probs = self._get_action_probs(obs)
        return action, ACTION_NAMES[action], probs

    def _get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract softmax action probabilities from the policy network.
        Returns uniform distribution if model access not available.
        """
        try:
            import torch
            obs_tensor = self.model.policy.obs_to_tensor(obs[None])[0]
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.numpy().flatten()
            return probs
        except Exception:
            return np.ones(4) / 4.0
