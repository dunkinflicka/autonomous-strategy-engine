"""
RL Training Experiment
========================
Trains and evaluates a PPO strategy agent at Silverstone.

Usage:
    python -m experiments.rl_training_experiment
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.train_agent import train_ppo_agent
from src.simulation.race_engine import TrackConfig
from src.core.tyre_model import TyreCompoundParams


# COMPOUND_PARAMS = {
#     "soft":   TyreCompoundParams("soft",   0.028, 0.15, 90.0, 0.008, 0.12, 0.18, 0.72, 2.8, 25, -0.9),
#     "medium": TyreCompoundParams("medium", 0.018, 0.10, 85.0, 0.006, 0.08, 0.12, 0.78, 2.3, 35,  0.0),
#     "hard":   TyreCompoundParams("hard",   0.012, 0.07, 80.0, 0.005, 0.05, 0.08, 0.85, 1.8, 50,  0.7),
# }

COMPOUND_PARAMS = {
    "soft":   TyreCompoundParams("soft",   0.028, 0.15, 90.0, 0.008, 0.12, 0.18, 0.72, 2.8, 25, -0.9),
    "medium": TyreCompoundParams("medium", 0.022, 0.12, 85.0, 0.007, 0.14, 0.28, 0.68, 3.0, 30,  0.0),
    "hard":   TyreCompoundParams("hard",   0.014, 0.08, 80.0, 0.005, 0.08, 0.14, 0.78, 2.2, 42,  0.7),
}




SILVERSTONE = TrackConfig(
    name="silverstone", total_laps=52, base_lap_time_s=89.0,
    fuel_load_kg_start=105.0, track_abrasion=0.75,
    safety_car_base_rate=0.22,
)


if __name__ == "__main__":
    print("Training PPO strategy agent at Silverstone...")
    model = train_ppo_agent(
        track_config=SILVERSTONE,
        compound_params=COMPOUND_PARAMS,
        total_timesteps=500_000,
        n_envs=4,
        save_path="./models/ppo_silverstone",
        seed=42,
    )
    print("Training complete.")
