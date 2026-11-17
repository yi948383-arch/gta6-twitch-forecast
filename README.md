[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/yi948383-arch/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/yi948383-arch/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch forecast

Entry for make.fun bounty `gta6-twitch-ccu-forecast-01`: forecast how many
people watch Grand Theft Auto VI live on Twitch in each hour of its release
day (2026-11-19 UTC), hours 2–24, as five quantiles (5/25/50/75/95%), from
everything recorded up to the end of hour 1.

## Run

```sh
python3 main.py            # reads /input, writes /output
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. No network, no randomness, no model
calls. One `<id>.out` per `<id>.in`, deterministic across runs, ~0.1 s a
case.

## Model

A launch-day curve forecaster fitted at run time on the case's own
`comparables` (past big launches' hourly Twitch curves):

1. **Training pool** — comparables whose unlock was a rolling midnight-local
   launch (GTA VI's own pattern), with a substantial hour-1 print
   (≥ 5,000 viewers) and no absurd peak/hour-1 ratio (≤ 30). This drops
   simultaneous-global launches (hour 1 near zero, then a 100–2000×
   explosion), early-access launches (hour 1 already inflated), and tiny
   sparsely-sampled categories. In the final case the pool is ~21 launches,
   including the four practice launches.
2. **Per-hour log regression** — for each hour h = 2..24,
   `ln(viewers_h) ~ b0 + b1·ln(viewers_1) + b2·trend`, where
   `trend = ln(viewers_1 / viewers_0)` and `viewers_0` is the hour just
   before the release day (the CET midnight unlock hour, from
   `history_hourly`). The trend carries how the UK unlock stacked on the
   CET unlock, which predicts how the US unlocks stack later in the day.
   `b1 < 1` throughout: a bigger hour 1 means a flatter day.
3. **Quantiles** — the fitted value is the log-median; the five quantiles
   are `exp(mu + z_tau · sigma)` with standard-normal z-scores and
   `sigma` the regression's residual SD shrunk by 0.85 (tuned by
   leave-one-out pinball loss).

Fallbacks keep every case valid: looser training gates, then an
empirical median-ratio forecast, then the house persistence baseline.

## Validation

Held out on the four practice launches (true SullyGnome release-day
curves; hour-1 prints matched exactly to the kit's `known` rows):

| launch | this entry | house baseline |
|---|---|---|
| Marvel's Wolverine | 0.172 | 0.744 |
| Red Dead Redemption 2 | 0.144 | 0.287 |
| God of War Ragnarök | 0.125 | 0.215 |
| Marvel's Spider-Man 2 | 0.183 | 0.303 |
| **mean** | **0.156** | **0.387** |

Mean log pinball loss 0.156 vs the baseline's 0.387 (−60%); typical miss
23.5% vs 82.8%; 90% coverage 0.91. Leave-one-out over the training pool:
0.281 vs the baseline's 0.790 on the same launches.

## Files

- `main.py` — the whole entry (model + I/O), standard library only
- `LICENSE` — MIT (from the kit's starter)
