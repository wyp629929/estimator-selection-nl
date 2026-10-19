#!/usr/bin/env bash
# E6 reports: acceptance checks per noise variant (exit code 0 = all passed), and the
# pre-registered predictions across variants (informational only).
# Usage: bash scripts/report_e6.sh
cd "$(dirname "$0")/.."
F=results/e6
for v in normal t3 skew; do
  python3 scripts/e2_report.py --dir "$F/noise_$v" > "$F/noise_$v/report.txt" 2>&1
  echo "exit code: $?" >> "$F/noise_$v/report.txt"
done
python3 scripts/e6_report.py --dir "$F" > "$F/predictions.txt" 2>&1
echo "exit code: $?" >> "$F/predictions.txt"
grep -H "exit code" $F/*/report.txt $F/predictions.txt
