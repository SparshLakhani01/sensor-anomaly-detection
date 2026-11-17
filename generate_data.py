"""Synthetic multivariate sensor-stream generator.

Simulates a generic industrial fluid process: 6 sensor channels sampled
once per minute for 30 days, with repeating operating phases, labeled
injected faults (spikes, stuck sensors, level shifts, noise bursts) and
a labeled gradual sensor-drift period. ALL data is synthetic, produced
by this script from a fixed seed. It is NOT real plant data, NOT
spectral data, and carries NO chemistry or process-engineering content:
channel names are generic labels for numeric streams.

Deterministic: the same seed always produces identical arrays (covered
by unit test). Ground-truth labels are written alongside the readings.
"""
import json

import numpy as np
import pandas as pd

SEED = 42
N_DAYS = 30
PER_DAY = 1440  # 1-minute sampling
N = N_DAYS * PER_DAY
CHANNELS = ["temperature", "pressure", "flow", "conductivity", "vibration", "valve"]
# (base mean, base std) per channel, in arbitrary display units
BASE = {
    "temperature": (70.0, 3.0),
    "pressure": (4.2, 0.30),
    "flow": (120.0, 8.0),
    "conductivity": (12.0, 1.0),
    "vibration": (2.5, 0.40),
    "valve": (62.0, 5.0),
}
TRAIN_DAYS = 14          # days 1-14: clean reference window, no faults/drift
DRIFT_ONSET_DAY = 24     # gradual drift starts at day 24 00:00 (1-indexed)
DRIFT_CHANNELS = ["temperature", "conductivity"]
DRIFT_RATE_SIGMA_PER_DAY = 0.15


def _phase_schedule(n):
    """Operating phases: production, plus a 30-min cleaning phase every
    6 hours and a 2-hour changeover every 3 days. Returns phase index
    per timestamp: 0 production, 1 cleaning, 2 changeover."""
    phase = np.zeros(n, dtype=int)
    t = np.arange(n)
    minute_of_day = t % PER_DAY
    # cleaning: minutes 360-390 and 1080-1110 each day
    phase[((minute_of_day >= 360) & (minute_of_day < 390)) |
          ((minute_of_day >= 1080) & (minute_of_day < 1110))] = 1
    # changeover: first 2 hours of every 3rd day
    day = t // PER_DAY
    phase[(day % 3 == 2) & (minute_of_day < 120)] = 2
    return phase


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    phase = _phase_schedule(N)
    day = t // PER_DAY

    # Phase mean offsets in units of each channel's std (cleaning runs
    # hotter with higher flow; changeover runs lower flow, higher vibration).
    offsets = {
        "temperature": np.array([0.0, 1.2, 0.3]),
        "pressure": np.array([0.0, 0.5, -0.4]),
        "flow": np.array([0.0, 1.5, -1.0]),
        "conductivity": np.array([0.0, 0.8, 0.2]),
        "vibration": np.array([0.0, 0.4, 1.0]),
        "valve": np.array([0.0, 1.0, -0.5]),
    }

    data = {}
    for ch in CHANNELS:
        mean, std = BASE[ch]
        # Daily seasonality + slow weekly wander + AR(1) noise.
        daily = 0.4 * std * np.sin(2 * np.pi * t / PER_DAY + rng.uniform(0, 6.28))
        weekly = 0.3 * std * np.sin(2 * np.pi * t / (7 * PER_DAY) + rng.uniform(0, 6.28))
        ar = np.zeros(N)
        eps = rng.normal(0, std * 0.55, N)
        for i in range(1, N):
            ar[i] = 0.7 * ar[i - 1] + eps[i]
        series = mean + daily + weekly + ar + offsets[ch][phase] * std
        data[ch] = series
    # Mild cross-channel coupling: flow pushes pressure.
    data["pressure"] = data["pressure"] + 0.15 * (data["flow"] - 120.0) / 8.0 * 0.30

    df = pd.DataFrame(data)
    df.insert(0, "timestamp_min", t)
    df["phase"] = phase
    labels = np.zeros(N, dtype=int)  # 1 = injected fault at this timestamp
    events = []  # (start, end, kind, channel) for detection-delay scoring

    test_start = TRAIN_DAYS * PER_DAY

    def add_event(start, end, kind, channel):
        events.append((int(start), int(end), kind, channel))
        labels[start:end] = 1

    # --- Injected faults, test window only (days 15-30) ---
    # Spikes: single-point 6-sigma jumps.
    spike_times = rng.choice(np.arange(test_start, N), size=40, replace=False)
    for s in spike_times:
        ch = CHANNELS[rng.integers(0, len(CHANNELS))]
        df.loc[s, ch] += 6 * BASE[ch][1] * rng.choice([-1, 1])
        add_event(s, s + 1, "spike", ch)
    # Stuck sensors: flatline 30-120 min.
    for _ in range(8):
        ch = CHANNELS[rng.integers(0, len(CHANNELS))]
        start = int(rng.integers(test_start, N - 120))
        length = int(rng.integers(30, 121))
        df.loc[start:start + length - 1, ch] = df.loc[start, ch]
        add_event(start, start + length, "stuck", ch)
    # Level shifts: +3 sigma for 2-6 hours.
    for _ in range(8):
        ch = CHANNELS[rng.integers(0, len(CHANNELS))]
        start = int(rng.integers(test_start, N - 360))
        length = int(rng.integers(120, 361))
        df.loc[start:start + length - 1, ch] += 3 * BASE[ch][1]
        add_event(start, start + length, "shift", ch)
    # Noise bursts: std x4 for 1-3 hours.
    for _ in range(6):
        ch = CHANNELS[rng.integers(0, len(CHANNELS))]
        start = int(rng.integers(test_start, N - 180))
        length = int(rng.integers(60, 181))
        df.loc[start:start + length - 1, ch] += rng.normal(0, 3 * BASE[ch][1], length)
        add_event(start, start + length, "noise", ch)

    # --- Gradual sensor drift from day 24 (labeled separately) ---
    drift_start = (DRIFT_ONSET_DAY - 1) * PER_DAY
    drift_idx = np.arange(drift_start, N)
    drift_days = (drift_idx - drift_start) / PER_DAY
    for ch in DRIFT_CHANNELS:
        df.loc[drift_start:, ch] += DRIFT_RATE_SIGMA_PER_DAY * drift_days * BASE[ch][1]
    drift_labels = np.zeros(N, dtype=int)
    drift_labels[drift_start:] = 1

    df["is_fault"] = labels
    df["is_drift_period"] = drift_labels
    meta = {
        "seed": seed,
        "n_points": N,
        "channels": CHANNELS,
        "train_days": TRAIN_DAYS,
        "test_start_min": test_start,
        "drift_onset_min": int(drift_start),
        "drift_channels": DRIFT_CHANNELS,
        "n_fault_points": int(labels.sum()),
        "n_events": len(events),
        "events": [{"start": s, "end": e, "kind": k, "channel": c} for s, e, k, c in events],
        "synthetic": True,
    }
    return df, meta


if __name__ == "__main__":
    df, meta = generate()
    df.to_csv("sensor_data.csv", index=False)
    with open("data_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote sensor_data.csv: {len(df)} rows, {meta['n_fault_points']} fault points, "
          f"{meta['n_events']} events, drift onset minute {meta['drift_onset_min']}")
