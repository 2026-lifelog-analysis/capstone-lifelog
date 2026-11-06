#!/usr/bin/env bash
# 작성 2026-11-06 07:45 (KST)
# ① 전체 실행: 1차(시드 0, 3개 프로세스) → 2차(시드 1~4, 4개 프로세스) → 분석
# 사용: bash experiments/seungho/run_all.sh [분석 출력 폴더]
set -euo pipefail
cd "$(dirname "$0")/../.."
RUNS=results/forecast/v3/runs
OUT=${1:-results/forecast/v3/seungho-delta-analysis}
TAG=$(date +%Y%m%d-%H%M%S)
LOG=results/forecast/v3/seungho-delta-logs-$TAG
mkdir -p "$LOG"
echo "[$(date +%T)] 1차 시작"
for c in 0 1 2; do
  python3 -m experiments.seungho.run_delta phase1 --chunk $c --nchunks 3 > "$LOG/p1c$c.log" 2>&1 &
done
wait
P1=$(for c in 0 1 2; do grep -o "저장: .*" "$LOG/p1c$c.log" | tail -1 | cut -d' ' -f2; done)
echo "[$(date +%T)] 1차 끝: $P1"
for s in 1 2 3 4; do
  python3 -m experiments.seungho.run_delta phase2 --seed $s --phase1 $P1 > "$LOG/p2s$s.log" 2>&1 &
done
wait
P2=$(grep -ho "저장: .*" "$LOG"/p2s*.log | cut -d' ' -f2)
echo "[$(date +%T)] 2차 끝"
python3 -m experiments.seungho.run_delta analyze --runs $P1 $P2 --out "$OUT" > "$LOG/analyze.log" 2>&1
echo "[$(date +%T)] 분석 끝: $OUT (로그 $LOG)"
