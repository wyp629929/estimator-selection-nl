#!/usr/bin/env bash
# E8 reports: acceptance checks for E8b and E8a (exit code 0 = all passed), and the pre-registered
# predictions (e8_report.py, informational only).
# Usage: bash scripts/report_e8.sh
cd "$(dirname "$0")/.."
F=results/e8
python3 scripts/e4b_report.py --dir $F/e4b --e4 $F/e4 > $F/e8b_report.txt 2>&1; echo "exit code: $?" >> $F/e8b_report.txt
python3 scripts/e4_report.py --dir $F/e4 > $F/e8a_report.txt 2>&1; echo "exit code: $?" >> $F/e8a_report.txt
python3 scripts/e8_report.py > $F/e8_predictions.txt 2>&1; echo "exit code: $?" >> $F/e8_predictions.txt
grep -H "exit code" $F/*.txt
