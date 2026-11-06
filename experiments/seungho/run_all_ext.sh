#!/usr/bin/env bash
# 작성 2026-11-06 08:40 (KST)
# ① 확장 전체: (run_all.sh 끝날 때까지 대기) → 3차 계열(시드 0) → 목표별 전진 선택 → 시드 1~4 확인 → 전체 분석
set -euo pipefail
cd "$(dirname "$0")/../.."
V=results/forecast/v3
while pgrep -f "experiments/seungho/run_all.sh" > /dev/null; do sleep 30; done
TAG=$(date +%Y%m%d-%H%M%S)
LOG=$V/seungho-ext-logs-$TAG
mkdir -p "$LOG"
echo "[$(date +%T)] 3차 시작"
for c in 0 1 2; do
  python3 -m experiments.seungho.run_extended phase3 --chunk $c --nchunks 3 > "$LOG/p3c$c.log" 2>&1 &
done
wait
P3=$(for c in 0 1 2; do grep -o "저장: .*" "$LOG/p3c$c.log" | tail -1 | cut -d' ' -f2; done)
echo "[$(date +%T)] 3차 끝. 전진 선택 시작"
for t in sleep bed wake; do
  python3 -m experiments.seungho.run_extended greedy --target $t > "$LOG/gr_$t.log" 2>&1 &
done
wait
G=$V/seungho-ext-greedy
echo "[$(date +%T)] 전진 선택 끝. 확인 시작"
python3 -m experiments.seungho.run_extended grfinal --greedy $G > "$LOG/grfinal.log" 2>&1 &
for s in 1 2 3 4; do
  python3 -m experiments.seungho.run_extended phase5 --seed $s --phase3 $P3 --greedy $G > "$LOG/p5s$s.log" 2>&1 &
done
wait
echo "[$(date +%T)] 확인 끝. 분석 시작"
RUNS=$(ls -d $V/runs/*seungho-delta-p1c* $V/runs/*seungho-delta-p2s* $V/runs/*seungho-ext-p3c* \
              $V/runs/*seungho-ext-grfinal-* $V/runs/*seungho-ext-p5s* | grep -v json)
python3 -m experiments.seungho.run_delta analyze --runs $RUNS --out $V/seungho-final-analysis > "$LOG/analyze.log" 2>&1
echo "[$(date +%T)] 전체 끝: $V/seungho-final-analysis (로그 $LOG)"
