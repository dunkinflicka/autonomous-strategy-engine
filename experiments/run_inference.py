import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO

from src.rl.environment import F1StrategyEnv
from src.rl.policy_inference import ACTION_NAMES
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
    fuel_load_kg_start=105.0, track_abrasion=0.75, safety_car_base_rate=0.22,
)

model = PPO.load("./models/ppo_silverstone")
env = F1StrategyEnv(SILVERSTONE, COMPOUND_PARAMS, seed=42)

obs, _ = env.reset()
print(f"{'Lap':<5} {'Wear':>6} {'Fuel':>6} {'Pos':>4} {'Action':<14} {'Reward':>8}")
print("-" * 50)

total_reward = 0
for lap in range(1, 53):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    total_reward += reward
    action_name = ACTION_NAMES[int(action)]
    print(f"{lap:<5} {info['tyre_wear']:>6.3f} {info['fuel_kg']:>6.1f} "
          f"{info['position']:>4} {action_name:<14} {reward:>8.4f}")
    if terminated:
        break

print(f"\nFinal position: {info['position']} | Total reward: {total_reward:.3f}")