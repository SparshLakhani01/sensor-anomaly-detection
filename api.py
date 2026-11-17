"""FastAPI service: score sensor readings with the fitted One-Class SVM
(best model in the offline evaluation) and check a window for drift vs
the production-phase reference distribution.

Endpoints:
  GET  /health
  POST /score  {"readings": [[temp, pressure, flow, conductivity,
                              vibration, valve], ...]} (most recent last;
                              rolling features use the supplied window)
  POST /drift  {"readings": [[...], ...]} -> PSI + KS p-value per channel,
                              vs the production-phase reference; send a
                              same-regime window (see README).
Artifacts (scaler + models + thresholds) are loaded from artifacts.pkl,
produced by run_evaluation.py. Reference distributions for drift come
from sensor_data.csv train rows (first 14 days), production phase only.
"""
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from drift import psi
from generate_data import CHANNELS, TRAIN_DAYS, PER_DAY
from scipy import stats

app = FastAPI(title="Sensor anomaly + drift monitor (synthetic-data prototype)")

with open("artifacts.pkl", "rb") as f:
    ART = pickle.load(f)
REF = pd.read_csv("sensor_data.csv").iloc[: TRAIN_DAYS * PER_DAY]
REF_PROD = REF[REF["phase"] == 0]


class Readings(BaseModel):
    readings: list[list[float]]


def _features(readings):
    df = pd.DataFrame(readings, columns=CHANNELS)
    roll = df.rolling(15, min_periods=1)
    feats = np.hstack([df.to_numpy(dtype=float),
                       roll.mean().to_numpy(dtype=float),
                       roll.std().fillna(0.0).to_numpy(dtype=float)])
    return feats


@app.get("/health")
def health():
    return {"status": "ok", "channels": CHANNELS}


@app.post("/score")
def score(body: Readings):
    feats = _features(body.readings)
    z = ART["scaler"].transform(feats)
    model = ART["models"]["one_class_svm"]
    s = -model.score_samples(z)
    thr = ART["thresholds"]["one_class_svm"]
    return {"scores": [round(float(v), 4) for v in s],
            "flags": [bool(v) for v in (s > thr)],
            "threshold": thr}


@app.post("/drift")
def drift(body: Readings):
    win = pd.DataFrame(body.readings, columns=CHANNELS)
    out = {}
    for ch in CHANNELS:
        p = psi(REF_PROD[ch].to_numpy(), win[ch].to_numpy())
        ks = stats.ks_2samp(REF_PROD[ch], win[ch]).pvalue
        out[ch] = {"psi": round(p, 4), "ks_p": float(f"{ks:.2e}"),
                   "alert": bool(p > 0.2 and ks < 0.01)}
    return {"channels": out}
