# Automotive Decision Intelligence — Race Strategy Optimisation

Real-time pit strategy optimisation using physics-informed simulation, Monte Carlo probabilistic planning, and a PPO reinforcement learning agent trained on sensor-derived race state.

Built to the standards of an automotive engineering simulation toolkit — interpretable physics models, reproducible stochastic rollouts, and ML that is justified by the problem, not applied for its own sake.

> **Automotive relevance:** The core problem — sequential decision-making under uncertainty using noisy sensor inputs, physical constraints, and competing objectives — is structurally identical to ADAS planning, energy management in hybrid/EV powertrains, and autonomous vehicle behaviour arbitration.

---

## Problem

**What decision is being optimised?**

At each lap, a race engineer must decide: *stay out* or *pit*, and if pitting, *which tyre compound*. The decision is irreversible, time-critical, and made under uncertainty — degrading tyres, stochastic safety car events, variable weather, and unknown competitor strategy all affect the outcome.

This is a finite-horizon stochastic control problem with:
- A continuous, partially observable state space (tyre wear, fuel load, track position, gaps)
- A discrete action space (4 actions per lap)
- A delayed, sparse reward signal (finishing position, known only at race end)
- Physical constraints on valid actions (minimum stint lengths, compound rules)

---

## Data & Inputs

| Input | Source | Type |
|---|---|---|
| Tyre wear | Physics model (calibratable from telemetry) | Continuous scalar |
| Fuel load | Physics model (1.85 kg/lap consumption) | Continuous scalar |
| Lap time | Composed from 6 physical terms + noise | Continuous scalar |
| Safety car status | Poisson / logistic stochastic model | Binary |
| Weather condition | Markov chain (4 states) | Categorical |
| Track position / gap | Race simulation engine | Continuous scalar |
| Competitor state | Precomputed opponent trajectories | Continuous scalar |

**Sensor fusion framing:** The 8-dimensional observation vector fed to the RL agent is a normalised fusion of physical sensor readings (wear estimated from thermal/acoustic data in real systems), vehicle state (fuel), environment state (weather, SC), and strategic context (position, gap) — directly mirroring the input structure of an ADAS situational awareness module.

---

## Architecture

```
Raw Sensor Data / Telemetry
          │
          ▼
  Physics Parameter Estimation (ML)
  ├── TyreParameterEstimator     GBT inference of k_c, cliff threshold from telemetry
  ├── LapTimeResidualModel       GBT / Gaussian Process correction of physics predictions
  └── SafetyCarPredictor         Logistic regression P(SC | lap, track, conditions)
          │
          ▼
  Physics Simulation Core
  ├── TyreModel        w(t+1) = w(t) + k_c × A_track × push × (1 + thermal_penalty)
  ├── LapTimeModel     t = t_base + Δt_tyre + Δt_fuel + Δt_traffic + Δt_weather + ε
  ├── FuelModel        Δt = 0.034 s/kg, consumption 1.85 kg/lap
  ├── SafetyCarModel   Poisson process + logistic regression variants
  └── WeatherModel     Markov chain: dry → damp → light rain → heavy rain
          │
          ▼
  Monte Carlo Probabilistic Planner
  └── 10,000–100,000 independent seeded rollouts → P(win), P(podium), E[pos], E[pts]
          │
          ▼
  Reinforcement Learning Decision Agent
  ├── Observation    [lap_frac, tyre_wear, compound, fuel_frac,
  │                   position_frac, sc_active, laps_since_pit, pit_count_frac]
  ├── Action space   Discrete(4): stay_out | pit_soft | pit_medium | pit_hard
  ├── Reward         Terminal position reward + tyre cliff penalty + SC opportunity bonus
  └── Algorithm      PPO (Stable-Baselines3), 500k timesteps, ~11 min training
```

---

## Results

**Monte Carlo — Monza, 2,000 simulations:**

| Strategy | Win% | Podium% | E[Position] | E[Points] | Sharpe |
|---|---|---|---|---|---|
| 1-stop S→H (lap 20) | 100.0% | 100.0% | 1.00 | 25.0 | — |
| 1-stop M→H (lap 24) | 18.3% | 61.9% | 3.09 | 15.8 | 10.1 |
| Undercut (lap 22) | 0.6% | 2.5% | 5.67 | 8.7 | 11.5 |
| 2-stop S→M→H | 0.0% | 1.1% | 5.81 | 8.4 | 15.6 |

**Monte Carlo — Silverstone, 2,000 simulations:**

| Strategy | Win% | Podium% | E[Position] | E[Points] |
|---|---|---|---|---|
| 1-stop S→H | 82.8% | 100.0% | 1.17 | 23.8 |
| 1-stop M→H | 17.2% | 100.0% | 1.83 | 19.2 |
| 2-stop S→M→H | 0.0% | 100.0% | 3.00 | 15.0 |

Simulation throughput: **870 rollouts/second** on a single CPU core.

**PPO agent — Silverstone, 500k training steps:**

| Metric | Value |
|---|---|
| Final ep_rew_mean | 0.558 |
| Explained variance | 0.809 |
| Training throughput | ~700 it/s |
| Learned pit window | Lap 35–40 (tyre wear 0.60–0.65) |
| Learned strategy | Long opening stint → single pit on cliff approach → P1 finish |
| Baseline (rule-based 1-stop M→H) | E[Position] 3.09, E[Points] 15.8 |
| **PPO agent** | **E[Position] 1.00, E[Points] 25.0 — +58% points vs baseline** |

---

## Physical Parameters

**Circuits:**

| Track | Laps | Lap Length | Base Lap Time | Abrasion | SC Rate | Rain Probability |
|---|---|---|---|---|---|---|
| Monza | 53 | 5.793 km | 80.5 s | 0.55 | 18% | 5% |
| Silverstone | 52 | 5.891 km | 89.0 s | 0.75 | 22% | 15% |
| Spa-Francorchamps | 44 | 7.004 km | 104.0 s | 0.65 | 32% | 35% |

**Tyre compounds:**

| Compound | Wear Rate | Cliff Threshold | Max Stint | Pace vs Medium |
|---|---|---|---|---|
| Soft | 0.028 /lap | 72% wear | 25 laps | −0.9 s/lap |
| Medium | 0.022 /lap | 68% wear | 30 laps | baseline |
| Hard | 0.014 /lap | 78% wear | 42 laps | +0.7 s/lap |

**Pit stop and fuel:**

| Parameter | Value |
|---|---|
| Stationary time | 2.40 s ± 0.18 s (σ) |
| Pit lane delta | 20.5 s ± 0.5 s (σ) |
| Unsafe release probability | 0.5% |
| Fuel consumption | 1.85 kg/lap |
| Lap time fuel penalty | 0.034 s/kg |
| Race start fuel load | 105 kg (Monza/Silverstone), 110 kg (Spa) |

---

## Setup

Python 3.10+

```bash
pip install numpy scipy pandas scikit-learn tqdm pyyaml
pip install gymnasium stable-baselines3 torch   # for RL training
```

---

## Usage

```bash
pytest tests/ -v                                  # 27 tests, ~0.5s

python -m experiments.monte_carlo_race_test       # probabilistic strategy comparison
python -m experiments.baseline_strategy_eval      # 1-stop vs 2-stop vs undercut at Monza
python -m experiments.rl_training_experiment      # train PPO agent (~11 min)
python experiments/run_inference.py               # lap-by-lap agent decisions
```

---

## Tests

27 tests across 4 suites, all passing:

```
test_tyre_model.py      wear monotonicity, cliff detection, thermal model, bounded at 1.0
test_lap_time_model.py  fuel effect, tyre degradation, traffic penalty, SC delta, noise
test_race_engine.py     full-race determinism, pit stop counts, position uniqueness
test_strategy_eval.py   MC output shape, win probabilities sum to 1.0, reproducibility
```

---

## Why It Matters Beyond Motorsport

The decision architecture here is a direct analogue of several open problems in automotive AI:

| This system | Automotive equivalent |
|---|---|
| Pit strategy under tyre degradation | Battery thermal management — when to charge, how hard to push |
| Safety car reactive pitting | ADAS event-triggered replanning (obstacle, weather change) |
| Monte Carlo strategy evaluation | Probabilistic trajectory planning under uncertainty |
| PPO on physical state observations | Learned energy/performance arbitration in hybrid powertrains |
| Physics model + ML residual correction | Digital twin calibration from real vehicle telemetry |

---

## Design Principles

**Physics before ML.** Tyre wear, fuel burn, and lap time are governed by interpretable equations with physically meaningful parameters. ML corrects residuals and infers parameters from data — it does not replace the model.

**Deterministic core, stochastic inputs.** Given a seed, every simulation is exactly reproducible. Randomness enters only through explicitly modelled stochastic processes: safety car arrivals, weather transitions, pit execution variance, and driver noise.

**RL as a decision layer.** The PPO agent observes physical state and outputs discrete pit decisions. It does not learn to simulate the race — it learns to act optimally within a simulation it cannot modify.