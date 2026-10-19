#!/usr/bin/env bash
# Major Revision: run all experiments sequentially
echo "=============================================="
echo "Major Revision Experiments"
echo "=============================================="
echo "Machine: $(hostname) | $(date)"
echo ""

echo "=== [1/5] CP Coverage Analysis (500 reps, 4 scenarios) ==="
python3 scripts/conformal_prediction.py
echo ""

echo "=== [2/5] Main simulation (500 reps, 4 scenarios + tuned DNN) ==="
python3 scripts/simulate_with_tuned_dnn.py
echo ""

echo "=== [3/5] Paired t-test + effect sizes (500 reps, nonlinear) ==="
python3 scripts/paired_ttest.py
echo ""

echo "=== [4/5] Sample size sensitivity (500 reps, 3 sample sizes) ==="
python3 scripts/sample_size_sensitivity.py
echo ""

echo "=== [5/5] Real-data experiments ==="
python3 scripts/run_real_data.py
echo ""

echo "Done: $(date)"
