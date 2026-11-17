"""Anomaly detection: feature building, model comparison, evaluation.

Models (all unsupervised, fit on the clean 14-day reference window):
  - rolling z-score baseline (per-channel, OR-combined)
  - Isolation Forest
  - One-Class SVM (RBF)
  - Local Outlier Factor (novelty mode)

Features per timestamp: standardized channel values + rolling mean/std
(15-minute window) per channel, computed causally (past values only).

Evaluation on the held-out test window (days 15-30), reported both
overall and on the pre-drift slice (days 15-23) so drift does not
contaminate the fault-detection numbers. Metrics: precision, recall,
F1, ROC-AUC (score-based methods), % of injected events detected,
median detection delay in minutes, false alarms per clean day.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from generate_data import CHANNELS, TRAIN_DAYS, PER_DAY, DRIFT_ONSET_DAY

ROLL = 15


def build_features(df):
    """Causal rolling features. Returns feature matrix (numpy) aligned
    with df rows; first ROLL rows use expanding windows (min_periods=1)."""
    feats = [df[CHANNELS].to_numpy(dtype=float)]
    roll = df[CHANNELS].rolling(ROLL, min_periods=1)
    feats.append(roll.mean().to_numpy(dtype=float))
    feats.append(roll.std().fillna(0.0).to_numpy(dtype=float))
    return np.hstack(feats)


def fit_models(X_train):
    scaler = StandardScaler().fit(X_train)
    Z = scaler.transform(X_train)
    models = {
        "isolation_forest": IsolationForest(n_estimators=200, contamination="auto",
                                            random_state=42, n_jobs=-1).fit(Z),
        "one_class_svm": OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Z),
        "lof": LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(Z),
    }
    return scaler, models


def score_models(scaler, models, X):
    """Returns dict name -> (flags, scores) where flags are 0/1 and
    higher scores mean MORE anomalous."""
    Z = scaler.transform(X)
    out = {}
    for name, m in models.items():
        s = -m.score_samples(Z)  # higher = more anomalous
        out[name] = (None, s)
    return out


def zscore_flags(df, train_end, thresh=4.0):
    """Rolling z-score baseline: per-channel rolling mean/std over the
    trailing 60 minutes (causal); flag if |z| > thresh on any channel.
    Reference statistics come from the train window only for the
    initial fill; rolling stats use past values only."""
    flags = np.zeros(len(df), dtype=int)
    for ch in CHANNELS:
        roll = df[ch].rolling(60, min_periods=30)
        mu = roll.mean()
        sd = roll.std().replace(0, np.nan).bfill().fillna(1.0)
        z = (df[ch] - mu) / sd
        flags |= (z.abs() > thresh).fillna(False).to_numpy(dtype=int)
    return flags


def threshold_flags(scores, quantile):
    thr = np.quantile(scores, quantile)
    return (scores > thr).astype(int), thr


def detection_delay(flags, events):
    """For each injected event, minutes from event start to first flag
    inside the event (None if never flagged)."""
    delays = []
    for ev in events:
        seg = flags[ev["start"]:ev["end"]]
        hit = np.argmax(seg) if seg.any() else None
        delays.append(int(hit) if seg.any() else None)
    detected = [d for d in delays if d is not None]
    return {
        "events_detected": len(detected),
        "events_total": len(events),
        "median_delay_min": float(np.median(detected)) if detected else None,
    }


def evaluate(df, meta, flags_dict, score_dict):
    test_start = meta["test_start_min"]
    drift_start = meta["drift_onset_min"]
    y = df["is_fault"].to_numpy()
    results = {}
    slices = {
        "test_all": slice(test_start, None),
        "test_pre_drift": slice(test_start, drift_start),
    }
    for name, flags in flags_dict.items():
        entry = {}
        for sname, sl in slices.items():
            yt, yp = y[sl], flags[sl]
            entry[sname] = {
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "false_alarms_per_day": float(yp[(yt == 0)].sum() / ((sl.stop or len(y)) - sl.start) * PER_DAY),
            }
            if name in score_dict:
                try:
                    entry[sname]["roc_auc"] = float(roc_auc_score(yt, score_dict[name][sl]))
                except ValueError:
                    entry[sname]["roc_auc"] = None
        # Event-level: only events fully inside the test window.
        test_events = [e for e in meta["events"] if e["start"] >= test_start]
        entry["events"] = detection_delay(flags, test_events)
        results[name] = entry
    return results


def run(df, meta):
    X = build_features(df)
    train_end = TRAIN_DAYS * PER_DAY
    scaler, models = fit_models(X[:train_end])
    scored = score_models(scaler, models, X)
    flags, scores = {}, {}
    thresholds = {}
    # Threshold each score method at the train-window 99.5th percentile
    # (flag rate calibrated on clean data only — no test labels used).
    for name, (_, s) in scored.items():
        thr = np.quantile(s[:train_end], 0.995)
        thresholds[name] = float(thr)
        flags[name] = (s > thr).astype(int)
        scores[name] = s
    flags["zscore_baseline"] = zscore_flags(df, train_end)
    return evaluate(df, meta, flags, scores), scaler, models, thresholds
