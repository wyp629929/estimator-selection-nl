#!/usr/bin/env bash
# E6 (Appendix D of the manuscript): the three noise variants of E2, one after the other.
# Resumable: re-running the same command skips replications that are already finished.
# Usage: bash scripts/run_e6.sh <workers>
set -euo pipefail
W=${1:?usage: bash scripts/run_e6.sh <workers>}
cd "$(dirname "$0")/.."
mkdir -p logs/e6
for v in normal t3 skew; do
  python3 scripts/e2_main.py --variant "noise_$v" --workers "$W" 2>&1 | tee -a "logs/e6/e6_$v.log"
done
echo "ALL DONE" | tee -a logs/e6/e6_skew.log
