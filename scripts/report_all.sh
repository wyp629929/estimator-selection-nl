#!/usr/bin/env bash
# Reports for E1-E5, each with its exit code (0 = all acceptance checks passed).
# Usage: bash scripts/report_all.sh
cd "$(dirname "$0")/.."
run() { out=$1; shift; python3 "$@" > "$out" 2>&1; echo "exit code: $?" >> "$out"; }
run results/e1/report.txt scripts/e1_report.py
run results/e2/report.txt scripts/e2_report.py
for v in corr tuned highp; do
  run "results/e3/$v/report.txt" scripts/e2_report.py --dir "results/e3/$v"
done
run results/e4/report.txt scripts/e4_report.py
run results/e4b/report.txt scripts/e4b_report.py --e4 results/e4
run results/e5/report.txt scripts/e5_report.py
grep -H "exit code" results/e1/report.txt results/e2/report.txt results/e3/*/report.txt results/e4/report.txt results/e4b/report.txt results/e5/report.txt
