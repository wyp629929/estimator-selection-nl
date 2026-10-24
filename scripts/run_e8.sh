#!/usr/bin/env bash
# E8 (Section 4.5 and Appendix D of the manuscript): E8b (what the CP width selects) then E8a (local selection) on the
# 31 datasets of the Grinsztajn et al. (2022) benchmark. Resumable: re-running skips finished splits.
# Usage: bash scripts/run_e8.sh <workers>
set -euo pipefail
W=${1:?usage: bash scripts/run_e8.sh <workers>}
cd "$(dirname "$0")/.."
mkdir -p logs/e8
python3 scripts/e4b_main.py --set e8 --workers "$W" 2>&1 | tee -a logs/e8/e8b.log
python3 scripts/e4_main.py --set e8 --workers "$W" 2>&1 | tee -a logs/e8/e8a.log
echo "ALL DONE" | tee -a logs/e8/e8a.log
