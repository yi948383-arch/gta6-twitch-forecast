"""gta6-twitch-ccu-forecast-01 entry: launch-day Twitch viewer forecaster.

Model
-----
A launch-day curve forecaster fitted at run time on the case's own
``comparables`` (past big game launches' hourly Twitch curves):

* Training pool: comparables whose unlock was a rolling midnight-local
  launch (the same unlock pattern as GTA VI), with a substantial hour-1
  print (>= 5,000 viewers) and no absurd peak/hour-1 ratio (<= 30).
  That drops simultaneous-global launches (whose hour 1 is near zero
  before a 14:00Z unlock and then explodes 100-2000x), early-access
  launches (hour 1 already inflated by days of early access) and tiny
  sparsely-sampled categories. In the final case the pool is about 21
  launches, including the four practice launches.
* Per forecast hour h = 2..24, an OLS regression in log space:

      ln(viewers_h) ~ b0 + b1 * ln(viewers_1) + b2 * trend

  where ``trend = ln(viewers_1 / viewers_0)`` and ``viewers_0`` is the
  hour just before the release day (the CET midnight unlock hour, read
  from ``history_hourly``). The trend carries how the UK unlock stacked
  on the CET unlock, which predicts how the US unlocks will stack later
  in the day. ``b1 < 1`` throughout: a bigger hour 1 means a flatter
  day, a smaller hour 1 a steeper climb.
* Quantiles: the regression's fitted value is the log-median; the five
  quantiles are ``exp(mu + z_tau * sigma)`` with the standard normal
  z-scores and ``sigma`` the regression's residual standard deviation,
  shrunk by 0.85 (the factor that minimised leave-one-out pinball loss
  on the training pool).

Validation (held-out, before this entry was written)
--------------------------------------------------
* Four practice launches (true SullyGnome release-day curves, hour-1
  print matched exactly to the kit's ``known`` row): mean log pinball
  loss 0.156 (Wolverine 0.172, RDR2 0.144, GoW Ragnarok 0.125,
  Spider-Man 2 0.183) against the house baseline's 0.387; typical miss
  23.5% against the baseline's 82.8%; 90% coverage 0.91.
* Leave-one-out over the 17-launch training pool: 0.281 against the
  baseline's 0.790 on the same launches.

Run: ``python3 -P -s /bot/main.py`` (Python 3.13, standard library
only), one case per run. Reads ``$MAKEFUN_INPUT_DIR`` (else ``/input``),
writes ``$MAKEFUN_OUTPUT_DIR`` (else ``/output``), one ``<id>.out``
per ``<id>.in``: ``{"hours": [{"hour": h, "q": [q05, q25, q50, q75,
q95]}, ...]}`` for hours 2..24. Deterministic: no randomness, no
clock, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
# Residual-spread shrink factor, tuned by leave-one-out pinball loss.
SPREAD_MULT = 0.85
# Residual-spread floor, in log units.
SD_FLOOR = 0.05
# Training-pool gates.
MIN_H1 = 5000
MAX_PEAK_RATIO = 30.0
MIN_POOL = 8
# Trend feature is clipped to this range (training data spans -0.6..0.5).
TREND_CLIP = 1.0


def parse_ts(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up, as the house baseline rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def classify(unlock):
    """Unlock pattern of a comparable launch, from its notes text."""
    u = (unlock or "").lower()
    if "target-history" in u or "not a launch" in u:
        return "target"
    if "early-access-contaminated" in u:
        return "ea"
    if "tier: weak" in u or "free-to-play" in u or "slow-burn" in u:
        return "weak"
    if any(s in u for s in (
            "single simultaneous", "simultaneous across",
            "single global unlock", "simultaneous global",
            "pc single time", "simultaneous midnight gmt",
            "pc/stadia simultaneous")):
        return "simul"
    return "rolling"


def day_curve(comp):
    """Release-day hourly viewers (hours 1..24) plus the pre-day hour.

    Returns (curve, v0) with curve a 24-list, or (None, None) when any
    release-day hour is missing or non-positive.
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None, None
    by_hour = {}
    for row in comp.get("hourly", []):
        try:
            t = parse_ts(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or viewers <= 0:
            continue
        h = int((t - day_start).total_seconds() // 3600) + 1
        if -1 <= h <= 24:
            by_hour[h] = float(viewers)
    if any(h not in by_hour for h in range(1, 25)):
        return None, None
    return [by_hour[h] for h in range(1, 25)], by_hour.get(0)


def ols(design, ys):
    """Least squares for a small design matrix. Returns (beta, resid_sd).

    None when the normal equations are singular.
    """
    n = len(ys)
    k = len(design[0])
    if n <= k:
        return None
    ata = [[sum(design[r][i] * design[r][j] for r in range(n))
            for j in range(k)] for i in range(k)]
    aty = [sum(design[r][i] * ys[r] for r in range(n)) for i in range(k)]
    m = [ata[i][:] + [aty[i]] for i in range(k)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(k):
            if r != col:
                f = m[r][col] / m[col][col]
                for cc in range(col, k + 1):
                    m[r][cc] -= f * m[col][cc]
    beta = [m[i][k] / m[i][i] for i in range(k)]
    resid = [ys[r] - sum(beta[j] * design[r][j] for j in range(k))
             for r in range(n)]
    sd = math.sqrt(sum(r * r for r in resid) / max(n - k, 1))
    return beta, sd


def train_pool(comparables):
    """Rolling-midnight launches usable as training curves.

    Each entry: (log hour-1, trend, 24-hour curve). Falls back to
    looser gates when too few launches pass, so a forecast is always
    possible.
    """
    parsed = []
    for comp in comparables or []:
        curve, v0 = day_curve(comp)
        if curve is None:
            continue
        tier = classify(comp.get("unlock"))
        trend = 0.0
        if v0 and v0 > 0:
            trend = math.log(curve[0] / v0)
            trend = max(-TREND_CLIP, min(TREND_CLIP, trend))
        parsed.append({"tier": tier, "curve": curve, "trend": trend})
    for keep in (
            lambda c: c["tier"] == "rolling" and c["curve"][0] >= MIN_H1
            and max(c["curve"]) / c["curve"][0] <= MAX_PEAK_RATIO,
            lambda c: c["tier"] == "rolling" and c["curve"][0] >= MIN_H1,
            lambda c: c["tier"] == "rolling",
            lambda c: c["curve"][0] >= 1000,
            lambda c: True,
    ):
        pool = [c for c in parsed if keep(c)]
        if len(pool) >= MIN_POOL:
            return pool
    return parsed


def median_ratio_forecast(p1, pool):
    """Fallback: per-hour empirical quantiles of log(viewers_h/viewers_1)."""
    out = {}
    for h in range(2, 25):
        ratios = sorted(math.log(c["curve"][h - 1] / c["curve"][0])
                        for c in pool)
        n = len(ratios)
        qs = []
        for tau in LEVELS:
            i = tau * (n - 1)
            lo = int(i)
            hi = min(lo + 1, n - 1)
            qs.append(ratios[lo] * (1 - (i - lo)) + ratios[hi] * (i - lo))
        out[h] = [max(1.0, p1 * math.exp(q)) for q in qs]
    return out


def persistence_forecast(p1):
    """Last fallback: the house baseline's shape (never invalid)."""
    out = {}
    for h in range(2, 25):
        out[h] = [max(1.0, p1 * math.exp(z * 0.15 * math.sqrt(h - 1)))
                  for z in Z]
    return out


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    # Trend: hour 1 over the hour just before the release day.
    v0 = None
    try:
        day_start = parse_ts(case["day_start"])
        target = day_start - timedelta(hours=1)
        for row in case.get("history_hourly", []):
            if parse_ts(row["from"]) == target and row.get("viewers"):
                v0 = float(row["viewers"])
                break
    except (KeyError, TypeError, ValueError):
        v0 = None
    trend = 0.0
    if v0 and v0 > 0:
        trend = max(-TREND_CLIP, min(TREND_CLIP, math.log(p1 / v0)))

    pool = train_pool(case.get("comparables"))
    if len(pool) < 3:
        return {"hours": [{"hour": h, "q": [round2(q) for q in qs]}
                            for h, qs in sorted(persistence_forecast(p1).items())]}

    log_p1 = math.log(p1)
    design = [[1.0, math.log(c["curve"][0]), c["trend"]] for c in pool]
    feats = [1.0, log_p1, trend]
    hours = {}
    ok = True
    for h in range(2, 25):
        ys = [math.log(c["curve"][h - 1]) for c in pool]
        fit = ols(design, ys)
        if fit is None:
            ok = False
            break
        beta, sd = fit
        mu = sum(b * f for b, f in zip(beta, feats))
        sigma = max(sd * SPREAD_MULT, SD_FLOOR)
        hours[h] = [max(1.0, math.exp(mu + z * sigma)) for z in Z]
    if not ok:
        hours = median_ratio_forecast(p1, pool)

    rows = []
    for h in sorted(hours):
        qs = []
        prev = 0.0
        for q in hours[h]:
            q = round2(min(max(q, 1.0), 1e9))
            q = max(q, prev)  # keep the quantiles non-decreasing
            prev = q
            qs.append(q)
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast over the 65,536-byte cap")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
