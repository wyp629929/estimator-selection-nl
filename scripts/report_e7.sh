#!/usr/bin/env bash
# E7 reports: acceptance checks for the CP width part (e4b) and the local selection part (e4),
# exit code 0 = all passed, and the pre-registered predictions (informational only).
# Usage: bash scripts/report_e7.sh
cd "$(dirname "$0")/.."
F=results/e7
python3 scripts/e4b_report.py --dir $F/e4b --e4 $F/e4 > $F/e4b/report.txt 2>&1; echo "exit code: $?" >> $F/e4b/report.txt
python3 scripts/e4_report.py --dir $F/e4 > $F/e4/report.txt 2>&1; echo "exit code: $?" >> $F/e4/report.txt
python3 scripts/e7_report.py > $F/predictions.txt 2>&1; echo "exit code: $?" >> $F/predictions.txt
grep -H "exit code" $F/*/report.txt $F/predictions.txt
