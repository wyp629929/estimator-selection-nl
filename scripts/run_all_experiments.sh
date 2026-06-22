#!/usr/bin/env bash
# Run SC regression first, then CI coverage analysis
echo "=== Phase 1: Superconductivity Regression ==="
python3 scripts/run_superconductivity.py
echo ""
echo "=== Phase 2: CI Coverage Analysis ==="
python3 scripts/coverage_analysis.py
echo ""
echo "=== All experiments complete ==="