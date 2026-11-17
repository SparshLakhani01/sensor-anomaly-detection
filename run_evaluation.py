"""End-to-end run: generate synthetic data, fit models, evaluate
anomaly detection, run drift monitoring, write report.json, and print
a human-readable summary. All metrics are measured from this run.
"""
import json
import pickle

from detect import run as run_detection
from drift import run_drift
from generate_data import generate


def main():
    df, meta = generate()
    df.to_csv("sensor_data.csv", index=False)
    results, scaler, models, thresholds = run_detection(df, meta)
    with open("artifacts.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "models": models, "thresholds": thresholds,
                     "train_end": meta["test_start_min"]}, f)
    drift = run_drift(df, meta)
    report = {
        "synthetic": True,
        "data": {k: v for k, v in meta.items() if k != "events"},
        "detection": results,
        "drift": {k: v for k, v in drift.items() if k != "windows"},
        "drift_windows": drift["windows"],
    }
    with open("report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=== Synthetic sensor data ===")
    print(f"points: {meta['n_points']}, channels: {len(meta['channels'])}, "
          f"fault points: {meta['n_fault_points']}, events: {meta['n_events']}, "
          f"drift onset minute: {meta['drift_onset_min']}")
    print("\n=== Anomaly detection (test window, days 15-30) ===")
    for name, r in results.items():
        a, p = r["test_all"], r["test_pre_drift"]
        ev = r["events"]
        print(f"{name}: F1={a['f1']:.3f} P={a['precision']:.3f} R={a['recall']:.3f} "
              f"AUC={a.get('roc_auc') if a.get('roc_auc') is None else round(a['roc_auc'], 3)} "
              f"| pre-drift F1={p['f1']:.3f} | events {ev['events_detected']}/{ev['events_total']} "
              f"median delay {ev['median_delay_min']} min "
              f"| false alarms/day {a['false_alarms_per_day']:.2f}")
    print("\n=== Drift monitoring (PSI>0.2 and KS p<0.01, 6h windows, 3-window persistence) ===")
    print(f"alerts: {drift['n_alerts']}/{drift['n_windows']} windows, "
          f"false confirmed alerts before onset: {drift['n_false_alerts_before_onset']}, "
          f"first drift-channel alert lag: {drift['first_drift_channel_lag_hours']} hours, "
          f"both drift channels lag: {drift['both_drift_channels_lag_hours']} hours")
    print("\nwrote report.json, sensor_data.csv, artifacts.pkl")


if __name__ == "__main__":
    main()
