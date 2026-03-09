# RL Strategy Design

## MDP Formulation
State: [lap_frac, tyre_wear, compound_enc, fuel_frac, pos_frac, gap_frac, sc_active, laps_since_pit_frac]
Actions: stay_out | pit_soft | pit_medium | pit_hard
Reward: terminal position reward + tyre cliff penalty + SC pit bonus

## Training
Algorithm: PPO (Stable-Baselines3)
LR: 3e-4, gamma: 0.99, n_steps: 2048
