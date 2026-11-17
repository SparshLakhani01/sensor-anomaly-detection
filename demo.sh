#!/bin/bash
# End-to-end demo: generate data, evaluate models, run drift monitoring,
# run unit tests, smoke-test the API. All metrics printed are measured
# from this run. Requires the project venv at .venv (see README).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "== 1. Generate + evaluate =="
$PY run_evaluation.py
echo ""
echo "== 2. Unit tests =="
$PY -m unittest discover -s tests 2>&1 | tail -3
echo ""
echo "== 3. API smoke test =="
.venv/bin/uvicorn api:app --port 8123 --log-level error &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 4
curl -s http://127.0.0.1:8123/health; echo
$PY - <<'EOF'
import json, urllib.request
import pandas as pd
df = pd.read_csv("sensor_data.csv")
prod = df[df["phase"] == 0]
cols = ["temperature","pressure","flow","conductivity","vibration","valve"]
# production-phase-only windows (the /drift endpoint's documented usage):
# a clean train-period window and a drift-period window (day 28+)
for label, frame in [("clean", prod.iloc[1000:1300]),
                     ("drift-period", prod[prod["timestamp_min"] >= 40000].iloc[:300])]:
    payload = json.dumps({"readings": frame[cols].values.tolist()}).encode()
    req = urllib.request.Request("http://127.0.0.1:8123/drift", data=payload,
                                 headers={"Content-Type": "application/json"})
    print(label, json.load(urllib.request.urlopen(req)))
req = urllib.request.Request("http://127.0.0.1:8123/score",
    data=json.dumps({"readings": df.iloc[20000:20030][
        ["temperature","pressure","flow","conductivity","vibration","valve"]
    ].values.tolist()}).encode(), headers={"Content-Type": "application/json"})
print("score sample:", json.load(urllib.request.urlopen(req)))
EOF
kill $SRV 2>/dev/null || true
echo "demo complete."
