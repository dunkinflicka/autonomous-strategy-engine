# Tyre Degradation Model

## Wear Accumulation
w(t+1) = w(t) + k_c * A_track * push_factor * (1 + thermal_penalty)

## Performance Degradation
Lap time penalty: Δt = a*w + b*w²

## Cliff Behaviour
When w >= cliff_threshold, wear rate is multiplied by cliff_multiplier.

## Compound Parameters
| Param | Soft | Medium | Hard |
|---|---|---|---|
| wear_rate_base | 0.028 | 0.018 | 0.012 |
| cliff_threshold | 0.72 | 0.78 | 0.85 |
| max_stint_laps | 25 | 35 | 50 |
| perf vs medium | -0.9s | 0.0s | +0.7s |
