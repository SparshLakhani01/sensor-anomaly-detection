"""Unit tests (stdlib unittest). Run: .venv/bin/python -m unittest discover -s tests -v"""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift import psi
from generate_data import CHANNELS, generate


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        df1, _ = generate(seed=42)
        df2, _ = generate(seed=42)
        self.assertTrue(df1.equals(df2))

    def test_shape_and_channels(self):
        df, meta = generate()
        self.assertEqual(len(df), 30 * 1440)
        for ch in CHANNELS:
            self.assertIn(ch, df.columns)

    def test_train_window_clean(self):
        df, meta = generate()
        self.assertEqual(df.iloc[: meta["test_start_min"]]["is_fault"].sum(), 0)
        self.assertEqual(df.iloc[: meta["test_start_min"]]["is_drift_period"].sum(), 0)

    def test_faults_and_events_agree(self):
        df, meta = generate()
        self.assertEqual(int(df["is_fault"].sum()), meta["n_fault_points"])
        self.assertEqual(len(meta["events"]), meta["n_events"])
        self.assertGreater(meta["n_events"], 50)

    def test_drift_period_labeling(self):
        df, meta = generate()
        self.assertEqual(df["is_drift_period"].sum(), len(df) - meta["drift_onset_min"])


class TestDrift(unittest.TestCase):
    def test_psi_identical_is_zero(self):
        x = np.random.default_rng(0).normal(0, 1, 5000)
        self.assertAlmostEqual(psi(x, x), 0.0, places=6)

    def test_psi_shifted_is_positive(self):
        rng = np.random.default_rng(0)
        self.assertGreater(psi(rng.normal(0, 1, 5000), rng.normal(2, 1, 5000)), 0.5)

    def test_psi_handles_constant(self):
        self.assertEqual(psi(np.ones(100), np.ones(100)), 0.0)


class TestReport(unittest.TestCase):
    def test_report_metrics_in_range(self):
        with open(os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "report.json")) as f:
            rep = json.load(f)
        self.assertTrue(rep["synthetic"])
        for name, r in rep["detection"].items():
            for sl in ("test_all", "test_pre_drift"):
                for m in ("precision", "recall", "f1"):
                    self.assertGreaterEqual(r[sl][m], 0.0)
                    self.assertLessEqual(r[sl][m], 1.0)
        self.assertGreaterEqual(rep["drift"]["n_false_alerts_before_onset"], 0)


if __name__ == "__main__":
    unittest.main()
