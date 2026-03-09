# F1 Race Strategy Simulator

Physics-informed Monte Carlo race simulator with a PPO-trained pit strategy agent.

Built as an engineering-grade simulation toolkit — not a dashboard, not a toy model. The focus is on interpretable physics, reproducible stochastic simulation, and ML that earns its place.

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

Throughput: **870 simulations/second** on a single CPU core.

**PPO agent — Silverstone, 500k training steps (~11 min):**

| Metric | Value |
|---|---|
| Final ep_rew_mean | 0.558 |
| Explained variance | 0.809 |
| Training throughput | ~700 it/s |
| Learned pit window | Lap 35–40 (wear 0.60–0.65) |
| Learned strategy | Long first stint → single soft stop → P1 finish |

---

## Setup

Python 3.10+

```bash
pip install numpy scipy pandas scikit-learn tqdm pyyaml
```

For RL training:
```bash
pip install gymnasium stable-baselines3 torch
```

---

## Usage

```bash
# Run all 27 tests (~0.5s)
pytest tests/ -v

# Monte Carlo strategy comparison at Silverstone
python -m experiments.monte_carlo_race_test

# Baseline strategy evaluation at Monza
python -m experiments.baseline_strategy_eval

# Train PPO pit strategy agent
python -m experiments.rl_training_experiment

# Lap-by-lap inference on trained agent
python experiments/run_inference.py
```

---

## Architecture

```
Physics Core
├── TyreModel          wear accumulation, cliff behaviour, thermal penalty
├── LapTimeModel       base + tyre + fuel + traffic + weather + noise
├── FuelModel          0.034 s/kg time penalty, per-lap consumption
├── SafetyCarModel     Poisson process + logistic regression variants
└── WeatherModel       Markov chain: dry → damp → rain → heavy rain

Simulation
├── RaceEngine         lap-by-lap state machine, all physics wired
├── PitStopModel       stationary time + pit lane delta, stochastic
└── MonteCarloEngine   N independent seeded simulations, parallel-ready

Strategy
├── RuleBasedStrategy  1-stop, 2-stop, undercut, overcut, SC-reactive templates
└── StrategyEvaluator  Monte Carlo wrapper → ranked reports with Sharpe ratio

Reinforcement Learning
├── F1StrategyEnv      Gymnasium Discrete(4): stay_out / pit_soft / medium / hard
├── RewardFunction     terminal position reward + tyre cliff penalty + SC bonus
└── train_agent.py     PPO via Stable-Baselines3, CPU-optimised

ML Calibration
├── TyreParameterEstimator    GBT inference of degradation coefficients from telemetry
├── LapTimeResidualModel      GBT or Gaussian Process residual correction
└── SafetyCarPredictor        logistic regression P(SC | lap, track, conditions)
```

---

## Physics Parameters

**Tracks:**

| Track | Laps | Lap Length | Base Lap Time | Abrasion | SC Rate | Rain Prob |
|---|---|---|---|---|---|---|
| Monza | 53 | 5.793 km | 80.5 s | 0.55 | 18% | 5% |
| Silverstone | 52 | 5.891 km | 89.0 s | 0.75 | 22% | 15% |
| Spa | 44 | 7.004 km | 104.0 s | 0.65 | 32% | 35% |

**Tyre compounds:**

| Compound | Wear Rate | Cliff Threshold | Max Stint | Pace vs Medium |
|---|---|---|---|---|
| Soft | 0.028 /lap | 72% | 25 laps | −0.9 s/lap |
| Medium | 0.022 /lap | 68% | 30 laps | baseline |
| Hard | 0.014 /lap | 78% | 42 laps | +0.7 s/lap |

**Pit stop and fuel:**

| Parameter | Value |
|---|---|
| Stationary time | 2.40 s ± 0.18 s |
| Pit lane delta | 20.5 s ± 0.5 s |
| Unsafe release probability | 0.5% |
| Fuel consumption | 1.85 kg/lap |
| Lap time penalty | 0.034 s/kg |
| Start load (Monza / Silverstone) | 105 kg |
| Start load (Spa) | 110 kg |

**Wear model:**
```
w(t+1) = w(t) + k_c × A_track × push_factor × (1 + thermal_penalty)
Δt_wear = a×w + b×w²
```

**Lap time composition:**
```
LapTime = BaseLapTime + TyrePenalty + FuelPenalty + TrafficPenalty + TrackEvolution + Noise
```

---

## Tests

```
tests/test_tyre_model.py      wear monotonicity, cliff detection, bounded at 1.0
tests/test_lap_time_model.py  fuel / tyre / traffic / SC / noise effects
tests/test_race_engine.py     determinism, pit counts, position uniqueness
tests/test_strategy_eval.py   MC shape, win prob sums to 1, reproducibility
```

```
27 passed in 0.56s
```

---

## Design Decisions

**Physics before ML.** Tyre wear, fuel penalty, and lap time are derived from interpretable equations, not learned models. ML is used only for residual correction and parameter inference from telemetry — never as a replacement for the physics.

**Deterministic core, stochastic inputs.** Given a seed, every simulation is fully reproducible. Uncertainty enters through sampled safety car events, weather transitions, pit stop execution, and lap time noise — not through the physics equations themselves.

**RL as a decision layer, not a replacement.** The PPO agent operates on top of the physics simulator. It learns when to pit and which compound to fit given tyre state, fuel load, position, and safety car status — it does not learn to predict lap times.