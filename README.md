# Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sillasill/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sillasill/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, the percent of the
players Exophase tracks who will have earned it a week after launch
(2026-11-26T00:00:00Z), as five quantiles per trophy (5/25/50/75/95%), scored by
pinball loss on the logit scale. `main.py` (Python 3.13, standard library only)
reads each case in `/input` and writes `/output/<id>.out`.

## How it forecasts

Exophase only refreshes its counts every few hours, so the history the house
records is a step function: frozen between refreshes, jumping at each one. The
model works from the *refresh points* — the distinct (players, rates) values —
not the many identical reads between them.

- **Trend.** Per trophy, a least-squares line of logit(rate) on ln(1 + hours
  since release), fitted on the most recent day of refresh points (the balance
  of dilution against players progressing shifts as the launch rush passes, so
  the early history would flatten the line). The slope is shrunk by one
  standard error and clamped; the extrapolated move is capped.
- **Launch dilution.** When the history is a single flat block — launch night,
  no refresh seen yet — persistence is wrong: the next refresh brings the
  launch rush of new players, who hold (almost) only the trophy everybody
  gets, so every other rate is diluted by roughly the growth of the tracked
  base. The forecast dilutes each rate by the expected player growth between
  the last refresh and the target.
- **Spread.** Quantiles sit on the logit scale with a spread combining the
  random-walk term (0.5 · √horizon in days), the trend fit's residual, and a
  refresh-timing term.

Deterministic: no randomness, no network, no wall-clock reads. Two runs on the
same case write the same bytes. Runs in ~0.05 s per case, far inside the 8 s
limit.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

(with the kit's `input/` beside it; the grader runs `python3 /bot/main.py` with
`/input` and `/output`).

## Local validation

- Practice cases p00/p01 run clean and byte-identical across two runs.
- Against practice answers reconstructed from the rules' own published
  refresh facts (638 → 3,868 players, 943 → 4,276 trophies earned, every rate
  but the 100% one falling), the forecast scores a pinball loss of roughly
  0.12–0.16 across five plausible reconstructions of the refreshed rates,
  against the house persistence baseline's 1.37–1.69 on the same answers.
  The true practice answers live on the server, so these are estimates, not
  site scores.
- On simulated final histories (cohort model: players join on a launch curve,
  earn trophies on per-trophy timescales, Exophase refreshes every few hours),
  the forecast beats the persistence baseline in every scenario and the house
  curve baseline in six of seven.

---

<sub>Entered on [make.fun](https://make.fun/?ref=bfyeqrq3&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
