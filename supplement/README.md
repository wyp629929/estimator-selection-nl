# Supplement: On the Alignment Between Marginal Uncertainty Measures and Conditional Decision Risk in Estimator Selection

## Overview
This supplement contains all code and instructions to reproduce the simulation experiments, figures, and tables in the paper.

## Dependencies
- Python 3.13
- numpy, scipy, matplotlib
- scikit-learn 1.6, xgboost 2.1, lightgbm 4.5
- PyTorch 2.5 (for DNN in main experiment only)
- tectonic (for LaTeX compilation)

## File Structure
```
scripts/
  run_regime.py              # CV-MSE validation experiment (6 regimes × 200 reps)
  run_cqr.py                 # CQR experiment (6 regimes × 200 reps)
  run_cvplus.py              # CV+ experiment (6 regimes × 200 reps)
  ranking_decision_gap.py    # Main ranking-decision gap experiment
  real_data_gap.py           # Real-data validation
  cv_mse_final.py            # CV-MSE baseline
  figure1.py                 # Figure 1: bar chart
  figure_bias_crossing.py    # Figure 2: bias crossing schematic
  figure_heatmap.py          # Figure 3: surrogate comparison heatmap
  figure_theory_bound.py     # Figure 4: theory bound validation
results/                     # JSON result files
figures/                     # Figure outputs
paper_rewrite.tex            # Paper source
refs.bib                     # Bibliography
```

## Reproducing Results
1. `cd scripts`
2. Run `python3 ranking_decision_gap.py` (main experiment, ~2 hours)
3. Run `python3 run_regime.py linear` etc. for each regime (CV-MSE, ~10 min each)
4. Run `python3 run_cqr.py` (CQR, ~30 min)
5. Run `python3 run_cvplus.py` (CV+, ~60 min)
6. Run `python3 real_data_gap.py` (real data, ~15 min)
7. Generate figures: `python3 figure1.py`, `python3 figure_bias_crossing.py`, etc.

## Random Seeds
All experiments use `BASE_SEED = 2024`, with per-replication seed `2024 + rep`.

## Data
- Simulation data is generated on-the-fly using numpy random state
- Real datasets: PIMA Diabetes, Home Credit Default Risk, Superconductivity
  - PIMA: included in data/pima_diabetes.csv
  - Home Credit: https://www.kaggle.com/c/home-credit-default-risk
  - Superconductivity: UCI Repository
