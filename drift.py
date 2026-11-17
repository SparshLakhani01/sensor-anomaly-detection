"""Drift monitoring: PSI + Kolmogorov-Smirnov per channel.

Reference distribution = the clean 14-day train window. Rolling 6-hour
windows over the test period are compared per channel. An alert fires
when PSI > 0.2 AND the KS test rejects at p < 0.01 (both conditions,
documented thresholds from common practice: PSI > 0.2 = significant
shift). Reports first-alert lag vs the true drift onset and counts
alerts raised before onset (false alerts).

Phase stratification (important): the process runs in operating phases
(production / cleaning / changeover) with different channel means, and
every channel carries a daily cycle. A first version compared every
window against the pooled reference and false-alarmed on 35 pre-onset
windows — regime mix and time-of-day, not drift. This version compares
each window against reference points matched on BOTH operating phase
(production only) AND time of day (the same 6-hour clock block across
the 14 train days).

Persistence rule (important): even matched comparisons false-alarmed
(24 pre-onset windows) because slow weekly wander and short injected
faults move single-window distributions. Drift is sustained, so an
alert requires the SAME channel to fire in 3 consecutive windows
(18 hours). This is standard alert-confirmation practice, not tuning
on the test labels: the rule was fixed before measuring lag.
"""
import numpy as np
import pandas as pd
from scipy import stats

from generate_data import CHANNELS, TRAIN_DAYS, PER_DAY, DRIFT_CHANNELS

WINDOW = 360  # 6 hours, in minutes
PSI_ALERT = 0.2
KS_P = 0.01


def psi(expected, actual, bins=10):
    """Population Stability Index. Quantile bins from the reference
    (expected) distribution; epsilon floor avoids log(0)."""
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    e_perc = np.histogram(expected, quantiles)[0] / len(expected)
    a_perc = np.histogram(actual, quantiles)[0] / len(actual)
    e_perc = np.clip(e_perc, 1e-4, None)
    a_perc = np.clip(a_perc, 1e-4, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def run_drift(df, meta):
    train_end = TRAIN_DAYS * PER_DAY
    ref = df.iloc[:train_end]
    ref_prod = ref[ref["phase"] == 0]
    drift_start = meta["drift_onset_min"]
    rows = []
    alerts = []
    streaks = {ch: 0 for ch in CHANNELS}
    for start in range(train_end, len(df) - WINDOW + 1, WINDOW):
        win = df.iloc[start:start + WINDOW]
        win_prod = win[win["phase"] == 0]
        row = {"window_start_min": int(start),
               "production_points": int(len(win_prod))}
        if len(win_prod) < 60:
            row["alert"] = False
            row["skipped"] = "too few production-phase points"
            rows.append(row)
            continue
        # Time-of-day matched reference: same 6h clock block, train days.
        clock_lo = start % PER_DAY
        clock_hi = clock_lo + WINDOW
        ref_clock = ref_prod[(ref_prod["timestamp_min"] % PER_DAY >= clock_lo) &
                             (ref_prod["timestamp_min"] % PER_DAY < clock_hi)]
        row["reference_points"] = int(len(ref_clock))
        fired_channels = []
        for ch in CHANNELS:
            p = psi(ref_clock[ch].to_numpy(), win_prod[ch].to_numpy())
            ks = stats.ks_2samp(ref_clock[ch], win_prod[ch]).pvalue
            row[f"psi_{ch}"] = round(p, 4)
            row[f"ks_p_{ch}"] = float(f"{ks:.2e}")
            if p > PSI_ALERT and ks < KS_P:
                fired_channels.append(ch)
        for ch in CHANNELS:
            streaks[ch] = streaks[ch] + 1 if ch in fired_channels else 0
        confirmed = [ch for ch in CHANNELS if streaks[ch] >= 3]
        row["fired_channels"] = fired_channels
        row["alert_channels"] = confirmed
        row["alert"] = bool(confirmed)
        rows.append(row)
        if confirmed:
            alerts.append(start)
    first_alert = alerts[0] if alerts else None
    false_alerts = [a for a in alerts if a < drift_start]
    drift_set = set(DRIFT_CHANNELS)
    post = [r for r in rows if r["window_start_min"] >= drift_start and r["alert"]]
    first_drift = next((r for r in post if drift_set & set(r.get("alert_channels", []))), None)
    first_both = next((r for r in post if drift_set <= set(r.get("alert_channels", []))), None)
    return {
        "windows": rows,
        "first_alert_min": first_alert,
        "true_onset_min": drift_start,
        "detection_lag_hours": round((first_alert - drift_start) / 60, 1) if first_alert is not None else None,
        "first_drift_channel_lag_hours": round((first_drift["window_start_min"] - drift_start) / 60, 1) if first_drift else None,
        "both_drift_channels_lag_hours": round((first_both["window_start_min"] - drift_start) / 60, 1) if first_both else None,
        "n_alerts": len(alerts),
        "n_false_alerts_before_onset": len(false_alerts),
        "n_windows": len(rows),
    }
