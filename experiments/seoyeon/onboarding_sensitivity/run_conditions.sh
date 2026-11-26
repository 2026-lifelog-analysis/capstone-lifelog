#!/usr/bin/env bash
# 출발점 민감도: 승호님 RQ3(설문이 드물 때 추적 방식 비교)를 온보딩 길이·라벨 기준·출발점 계산을 바꿔 다시 돌린다.
# usage: PY=<파이썬> bash run_conditions.sh <HumanRhythms 원자료 폴더> <결과 폴더>
#   1) 승호님 hr_build.py로 사람-일 표를 만들고 (원본 그대로)
#   2) 조건마다 sparse_onboarding.py를 돌린다 (기본 조건은 원본과 같은 계산)
# 조건 근거
#   ONBOARD half : 승호님 원래 설계 (라벨 있는 날의 앞 50%)
#   ONBOARD 28   : 팀 공통 기준선 28일 (capstone-lifelog src/baseline.py, MIRATAR 실험)
#   ONBOARD 14   : lifelog-bench 계약 6절의 처음 14줄
#   SKIP 7       : 기준 구간 민감도 결과에서 수면의 질 첫 7줄이 평균보다 0.13점 낮음
#   SHRINK 1     : 춘자 논문 식 (5), ETRI 확장실험 방향 1의 섞는 방식
set -euo pipefail
HR="$1"; OUT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
# macOS에서 LightGBM이 찾는 libomp: 시스템에 없으면 scikit-learn에 들어 있는 것을 쓴다
OMP="$("$PY" -c 'import os, sklearn; print(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs"))' 2>/dev/null || true)"
if [ -d "$OMP" ]; then export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:+$DYLD_FALLBACK_LIBRARY_PATH:}$OMP"; fi
mkdir -p "$OUT"

if [ ! -f "$OUT/daily.pkl" ]; then
  echo "== 사람-일 표 만들기 (승호님 hr_build.py)"
  "$PY" -I "$HERE/seungho_original/hr_build.py" "$HR" "$OUT/daily.pkl" > "$OUT/hr_build.log"
fi

run() {  # name ONBOARD LABEL SKIP SHRINK
  local name="$1"
  if [ -f "$OUT/$name/rq3_summary.json" ]; then echo "== $name (이미 있음)"; return; fi
  echo "== $name  ONBOARD=$2 LABEL=$3 SKIP=$4 SHRINK=$5  $(date +%H:%M:%S)"
  ONBOARD="$2" LABEL="$3" SKIP="$4" SHRINK="$5" "$PY" -I "$HERE/sparse_onboarding.py" "$OUT/daily.pkl" "$OUT/$name" > "$OUT/$name.log" 2>&1
}

CONDS="${CONDS:-main}"
# 주 조건: 온보딩 길이 × 라벨 기준
run half_onboard half onboard 0 0
if [ "$CONDS" != "repro" ]; then
  run d28_fixed 28 half 0 0
  run d28_onboard 28 onboard 0 0
  run d14_fixed 14 half 0 0
  run d14_onboard 14 onboard 0 0
fi
# 보조 조건: 현실적인 라벨(짧은 온보딩 기준)에서 출발점 계산만 바꿈
if [ "$CONDS" = "all" ]; then
  run d14_onboard_skip7 14 onboard 7 0
  run d14_onboard_shrink 14 onboard 0 1
  run d28_onboard_skip7 28 onboard 7 0
  run d28_onboard_shrink 28 onboard 0 1
fi
echo "done -> $OUT"
