#!/usr/bin/env bash
# E7 (Appendix D of the manuscript): E7b (what the CP width selects) then E7a (local selection) on the
# 20 OpenML-CTR23 regression datasets. Resumable: re-running skips finished splits.
# Usage: bash scripts/run_e7.sh <workers>
set -euo pipefail
W=${1:?usage: bash scripts/run_e7.sh <workers>}
cd "$(dirname "$0")/.."
mkdir -p logs/e7
python3 scripts/e4b_main.py --set e7 --workers "$W" 2>&1 | tee -a logs/e7/e7b.log
python3 scripts/e4_main.py --set e7 --workers "$W" 2>&1 | tee -a logs/e7/e7a.log
echo "ALL DONE" | tee -a logs/e7/e7a.log
