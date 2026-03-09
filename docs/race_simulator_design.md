# Race Simulator Design

## Architecture
Deterministic core with stochastic inputs.
RaceEngine drives: TyreModel, FuelModel, LapTimeModel, PitStopModel, SafetyCarModel, WeatherModel.

## Monte Carlo
Each sim uses independent seeded Generator: seed = base_seed + sim_idx.
Target: 1000+ sims/sec on a modern CPU.
