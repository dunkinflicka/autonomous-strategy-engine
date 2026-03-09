"""
RL Agent Training
=================
Trains a PPO agent on the F1StrategyEnv using Stable-Baselines3.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict

import numpy as np

from src.rl.environment import F1StrategyEnv
from src.simulation.race_engine import TrackConfig
from src.core.tyre_model import TyreCompoundParams


def train_ppo_agent(
    track_config: TrackConfig,
    compound_params: Dict[str, TyreCompoundParams],
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    save_path: Optional[str] = None,
    seed: int = 42,
) -> "stable_baselines3.PPO":
    """
    Train a PPO agent on the F1 strategy environment.

    Parameters
    ----------
    track_config      : TrackConfig for the target circuit
    compound_params   : Tyre compound parameter dict
    total_timesteps   : training budget
    n_envs            : number of parallel environments
    learning_rate     : PPO learning rate
    save_path         : path to save final model (optional)
    seed              : random seed

    Returns
    -------
    Trained PPO model
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 required for RL training. "
            "Install with: pip install stable-baselines3"
        ) from e

    def make_env():
        return F1StrategyEnv(
            track_config=track_config,
            compound_params=compound_params,
        )

    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=seed,
        tensorboard_log="./logs/ppo_f1/",
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        print(f"Model saved to {save_path}")

    return model
