"""Trophy Rarity Forecast — gta6-trophy-rarity-forecast-01 (Python 3.13, standard library only).

Forecasts, for every GTA VI PlayStation 5 trophy, its earn rate on Exophase at the
case's target, as five quantiles (5/25/50/75/95%), scored by pinball loss on the
logit scale. Reads every <id>.in in the input directory and writes /output/<id>.out.

How it forecasts
---------------
The house reads Exophase on a fixed clock, but Exophase only refreshes its counts
every few hours: between refreshes the page (and so the history) is frozen, and at a
refresh the rates jump to the current truth. So the history is a step function, and
the informative points are the *refresh points* — the distinct (players, rates)
values, one per refresh — not the many identical reads between them.

1. Refresh points. Consecutive reads with identical players_tracked and rates are
   collapsed to one point (the first read showing those values).

2. Trend. Per trophy, a least-squares line of logit(rate) on ln(1 + hours since the
   game's release), fitted on the refresh points only (the house curve baseline fits
   the same line on *all* reads, so its slope is dragged by stale duplicates).
   The fit uses the most recent day of refresh points: the balance of dilution
   against players progressing shifts as the launch rush passes, so the early
   history — where the rates first fell as the tracked base flooded in — would
   flatten a line fitted over the whole history. When fewer than three refresh
   points fall in the last day, the window widens to two days, then to all of
   them. The slope is shrunk by one standard error — an insignificant slope
   becomes flat — and clamped, and the extrapolated move from the last rate is
   capped at 3.5 logit units. With fewer than 3 refresh points, or no variation,
   the trend is flat.

3. Launch dilution (flat histories only). When the history is a single flat block
   — as on launch night, when no refresh has been seen yet — a trend cannot be
   fitted, and persistence would be wrong: the next refresh brings the launch rush
   of new players, who hold (almost) only the trophy everybody gets, so every other
   rate is diluted by roughly the growth of the tracked base. The forecast then
   dilutes each rate by the expected growth of players tracked between the last
   refresh and the target (fast in the first hours after launch, slower later),
   with the new cohort holding the universal trophy in full and a fiftieth of the old
   cohort's rate on every other trophy. This is what the practice cases need: their
   histories are perfectly flat, their answers are the first refreshed page, and
   every rate but the one at 100% falls.

4. Spread. Quantiles sit on the logit scale around the median with a spread that
   combines the house random-walk term (0.5 * sqrt(horizon in days)), the model's
   own uncertainty (large when the history is flat and the dilution is a prior,
   the trend fit's residual otherwise) and a refresh-timing term (the answer is
   as of the last refresh before the target, which may or may not land). A
   forecast that is too narrow pays for every trophy outside it; the spread is
   the honest price of not knowing. The random-walk term already grows with the
   horizon, so no separate extrapolation penalty is added on top of it.

Deterministic: no randomness, no wall-clock reads, no network. Two runs on the
same case write the same bytes.

Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The game's release, the origin of the log-time axis (the house curve baseline
# counts hours from 2026-11-19T00:00:00Z too). Overridden by the case's own
# release_date when it parses.
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(l: float) -> float:
    l = min(9.21, max(-9.21, l))
    return 100 / (1 + math.exp(-l))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since_release(when: datetime, release: datetime) -> float:
    return max(0.0, (when - release).total_seconds() / 3600.0)


def refresh_points(case: dict, release: datetime):
    """Collapse the history to one point per distinct (players, rates) value.

    Returns a list of (hours_since_release, players_tracked, rates) in time order.
    """
    points = []
    last_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != last_key:
            t = hours_since_release(parse_time(read["at"]), release)
            points.append((t, read["players_tracked"], list(read["rates"])))
            last_key = key
    return points


def fit_trend(xs, ys):
    """Least squares y = a + b*x, with the slope shrunk by one standard error.

    Returns (a, b, rmse) or None when the points cannot support a line.
    """
    n = len(xs)
    if n < 3:
        return None
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx < 0.09:  # the x range is too small to say anything about a slope
        return None
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    rmse = math.sqrt(sum(r * r for r in resid) / n)
    if max(ys) - min(ys) < 0.05:
        return None
    # Standard error of the slope; shrink the slope by one SE so a slope the
    # data cannot distinguish from flat becomes flat.
    if n > 2 and sxx > 0:
        se_b = math.sqrt(sum(r * r for r in resid) / (n - 2) / sxx)
        shrink = max(0.0, 1.0 - se_b / max(abs(b), 1e-9))
        b *= shrink
    b = min(2.0, max(-1.5, b))
    return (a, b, rmse)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    release = RELEASE
    try:
        release = parse_time(case["game"]["release_date"])
    except Exception:
        pass
    target = parse_time(case["target"])
    t_target = hours_since_release(target, release)
    x_target = math.log1p(t_target)
    horizon_days = case["horizon_minutes"] / 1440.0

    points = refresh_points(case, release)
    last_rates = points[-1][2]
    # The last refresh was seen first at this read; the target may be hours later.
    t_last_refresh = points[-1][0]
    flat = len(points) < 2

    # --- trend fits, one per trophy, on the recent refresh points only ------
    # The most recent day's refresh points carry the current regime; the early
    # history (rates first falling as the tracked base flooded in) would flatten
    # a line fitted over the whole history. Widen the window when too few points.
    t_last = points[-1][0]
    window = [p for p in points if p[0] >= t_last - 24.0]
    if len(window) < 3:
        window = [p for p in points if p[0] >= t_last - 48.0]
    if len(window) < 3:
        window = points
    trends = []  # (median_logit_at_target, rmse) or (None, None) when flat
    for i in range(len(case["trophies"])):
        xs, ys = [], []
        for (t, _n, rates) in window:
            r = rates[i]
            if r is not None:
                xs.append(math.log1p(t))
                ys.append(logit(r))
        fit = fit_trend(xs, ys)
        if fit is None:
            trends.append((None, None))
        else:
            a, b, rmse = fit
            trends.append((a + b * x_target, rmse))
    fit_rmses = [rm for (_m, rm) in trends if rm is not None]
    median_rmse = sorted(fit_rmses)[len(fit_rmses) // 2] if fit_rmses else 0.0

    # --- launch dilution, for flat histories --------------------------------
    # Expected growth of players tracked between the last refresh and the target.
    # Early after launch the tracked base grows about one doubling an hour; later
    # it grows far more slowly. The new cohort holds the universal trophy in full
    # and a tenth of the old cohort's rate on every other trophy.
    dilute = None
    if flat:
        dt = max(0.0, t_target - t_last_refresh)
        tau = 1.0 if t_target <= 12.0 else 6.0
        growth = 1.0 + dt / tau
        dilute = (growth, 0.02)

    # --- spread ---------------------------------------------------------------
    s_rw = 0.5 * math.sqrt(horizon_days)
    s_refresh = 0.25
    if flat:
        s_model = 0.4
    else:
        # The fit's own residual is the model uncertainty; the random-walk term
        # already grows with the horizon, so no separate extrapolation penalty.
        s_model = min(1.0, max(0.25, median_rmse))

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last = last_rates[i]
        if last is None:
            # Fall back to the last non-null rate for this trophy, else 0.01.
            last = 0.01
            for (_t, _n, rates) in reversed(points):
                if rates[i] is not None:
                    last = rates[i]
                    break
        universal = last >= 99.5
        med_logit, _rm = trends[i]
        if med_logit is None:
            med_logit = logit(last)
        else:
            # Cap the extrapolated move from the last rate.
            base = logit(last)
            med_logit = min(base + 3.5, max(base - 3.5, med_logit))
        if dilute is not None and not universal:
            growth, f_new = dilute
            rate = last * (1.0 + (growth - 1.0) * f_new) / growth
            med_logit = logit(rate)
        if universal:
            med_logit = logit(100.0)
            spread = 0.05
        else:
            spread = math.sqrt(s_rw * s_rw + s_model * s_model + s_refresh * s_refresh)
        qs = [round4(inv_logit(med_logit + z * spread)) for z in Z]
        # Rounding is monotone, but enforce non-decreasing explicitly.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
