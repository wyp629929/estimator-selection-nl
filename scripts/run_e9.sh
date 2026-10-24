#!/usr/bin/env bash
# E9 (Section 4.5 and Appendix D of the manuscript): E9a (binary results with predictions clipped to [0, 1]) on the 20
# binary datasets, then the post hoc diagnostics of topo_2_1 and sulfur on 20 splits each.
# Resumable for E9a (finished splits are skipped); the diagnostics rerun from scratch (a few minutes).
# Usage: bash scripts/run_e9.sh <workers>
set -euo pipefail
W=${1:?usage: bash scripts/run_e9.sh <workers>}
cd "$(dirname "$0")/.."
mkdir -p logs/e9
python3 scripts/diag_e9a_clipped.py --workers "$W" 2>&1 | tee -a logs/e9/e9a.log
python3 scripts/diag_e8_topo_sulfur.py --workers "$W" 2>&1 | tee -a logs/e9/topo_sulfur.log
python3 scripts/e9a_report.py > results/e9a/report.txt 2>&1; echo "exit code: $?" >> results/e9a/report.txt
echo "ALL DONE" | tee -a logs/e9/e9a.log
