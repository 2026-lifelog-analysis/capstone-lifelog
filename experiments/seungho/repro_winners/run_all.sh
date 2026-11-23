#!/usr/bin/env bash
# Run the whole reproduction + follow-up pipeline.
# usage: bash run_all.sh <DATA_DIR> <OUT_DIR>
#   DATA_DIR must contain pmdata/ (p01..p16), HumanRhythms/ (Cleaned_Raw_Data, Additional_Data)
#   and optionally ch2025_metrics_train.csv (ETRI labels; only aggregate numbers are written).
# Takes about 10 minutes. Outputs: <OUT_DIR>/{pm,hr,etri}/...
set -euo pipefail
DATA="$1"; OUT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=python3
mkdir -p "$OUT/pm" "$OUT/hr" "$OUT/etri"

echo "== 1/4 build person-day tables"
$PY "$HERE/pm_build.py" "$DATA/pmdata" "$OUT/pm/daily.pkl"
$PY "$HERE/hr_build.py" "$DATA/HumanRhythms" "$OUT/hr/daily.pkl"

for ds in pm hr; do
  echo "== 2/4 [$ds] reproduction: 춘자 4-stage comparison, LOV3 label structure"
  $PY "$HERE/repro_chunja.py" "$OUT/$ds/daily.pkl" "$OUT/$ds" > "$OUT/$ds/repro_chunja.log"
  $PY "$HERE/repro_lov3.py"   "$OUT/$ds/daily.pkl" "$OUT/$ds" > "$OUT/$ds/repro_lov3.log"
  echo "== 3/4 [$ds] RQ3 sparse surveys"
  $PY "$HERE/sparse_survey.py" "$OUT/$ds/daily.pkl" "$OUT/$ds" > "$OUT/$ds/rq3.log"
  echo "== 4/4 [$ds] RQ4 sensor-independent events / scales"
  $PY "$HERE/external_check.py" "$OUT/$ds/daily.pkl" "$OUT/$ds/rq3_preds.pkl" "$OUT/$ds" > "$OUT/$ds/rq4.log"
done

if [ -f "$DATA/ch2025_metrics_train.csv" ]; then
  echo "== ETRI implementation check (labels only)"
  $PY "$HERE/etri_icc_check.py" "$DATA/ch2025_metrics_train.csv" "$OUT/etri" | tee "$OUT/etri/etri_icc_check.log"
fi
echo "done -> $OUT"
