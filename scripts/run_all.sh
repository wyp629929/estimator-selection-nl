#!/usr/bin/env bash
# Final rerun (2026-10-16): E4b, E4, E2 and the three E3 variants in sequence, with E1 (one
# process) in the background. All outputs go to results/. Resumable: re-running the same
# command skips replications that are already finished (E1 restarts from the beginning, ~30 min).
# Usage: bash scripts/run_all.sh <workers>
set -euo pipefail
W=${1:?usage: bash scripts/run_all.sh <workers>}
cd "$(dirname "$0")/.."
mkdir -p logs
python3 scripts/e1_mirror.py > logs/e1.log 2>&1 &
E1=$!
python3 scripts/e4b_main.py --workers "$W" 2>&1 | tee -a logs/e4b.log
python3 scripts/e4_main.py --workers "$W" 2>&1 | tee -a logs/e4.log
python3 scripts/e2_main.py --workers "$W" 2>&1 | tee -a logs/e2.log
for v in corr tuned highp; do
  python3 scripts/e2_main.py --variant "$v" --workers "$W" 2>&1 | tee -a "logs/e3_${v}.log"
done
wait "$E1" || { echo "E1 FAILED, see logs/e1.log" | tee -a logs/e2.log; exit 1; }
echo "ALL DONE" | tee -a logs/e2.log
