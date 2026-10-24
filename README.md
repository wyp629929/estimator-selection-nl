# Estimator selection with conformal interval width

Code and results for *On the Alignment Between Marginal Uncertainty Measures and Conditional Decision Risk in Estimator Selection* (Statistics and Computing, revised manuscript).

The paper asks when a residual-based uncertainty summary, the split-conformal interval width, can guide the choice among fitted estimators. For the global prediction risk, the width agrees with the risk when one estimator's errors are uniformly smaller and can prefer the riskier estimator otherwise; for the conditional risk at each covariate value, no rule computed from the residuals alone can identify the best estimator, while a selector that uses the covariates can.

## Repository layout

```
theory/
  theory_checks.py          numerical checks of every result of Section 3, including equation (4)
scripts/                    experiments, diagnostics, reports (one flat directory; scripts import each other)
results/                    final results, one directory per experiment, each with its report.txt
  e1/                       mirror construction of Theorem 4                         (Figure 2)
  e2/                       six simulated regimes, global and local targets          (Tables 1, 2)
  e3/{corr,tuned,highp}/    correlated covariates, tuned estimators, 50/400 covariates (Table B1)
  e4/                       local selection on 8 real data sets                      (Tables 4, B3)
  e4b/                      what the conformal width selects on 8 real data sets     (Tables 3, B2)
  e5/                       mechanism of Proposition 2(b), dose-response             (Figure 1)
  e6/noise_{normal,t3,skew}/  noise variants                                         (Table B1)
  e7/{e4,e4b}/              the 20 OpenML-CTR23 regression data sets: local selection and width
  e8/{e4,e4b}/              replication on 31 data sets of Grinsztajn et al. (2022)  (Section 4.5, Tables B4, B5)
  e9a/                      binary data sets with predictions clipped to [0, 1]      (Section 4.5, Table B6)
  diag/                     diagnostics of the shrinkage pattern (Section 4.4) and of topo_2_1 and sulfur (Section 4.5)
checksums/                  SHA-256 of the benchmark CSVs used by E8
figures/                    Figure 2
requirements.txt
```

The original-submission code is preserved under the tag [`v1-submission`](https://github.com/wyp629929/estimator-selection-nl/tree/v1-submission). The revised code corrects the split-conformal quantile, uses one definition of the global oracle, evaluates selectors and oracles on the same fitted models, and seeds all random number generation.

## Scripts

| Experiment | Run | Report |
|---|---|---|
| E1, mirror construction | `e1_mirror.py` | `e1_report.py` (also draws Figure 2) |
| E2, simulations; E3 variants (`--variant corr/tuned/highp`) | `e2_main.py` | `e2_report.py` |
| E4, local selection on real data (`--set e4`, `e7` or `e8`) | `e4_main.py` | `e4_report.py` |
| E4b, what the width selects on real data (`--set e4`, `e7` or `e8`) | `e4b_main.py` | `e4b_report.py` |
| E5, mechanism of Proposition 2(b) | `e5_mechanism.py` | `e5_report.py` |
| E6, noise variants | `run_e6.sh` (calls `e2_main.py --variant noise_*`) | `report_e6.sh`, `e6_report.py` |
| E7, CTR23 data sets | `run_e7.sh` | `report_e7.sh`, `e7_report.py` |
| E8, replication on the Grinsztajn et al. (2022) benchmark | `run_e8.sh` | `report_e8.sh`, `e8_report.py` |
| E9a, binary data sets with clipped predictions; diagnostics of topo_2_1 and sulfur | `run_e9.sh` (calls `diag_e9a_clipped.py`, `diag_e8_topo_sulfur.py`) | `e9a_report.py` |
| Data preparation | `e4_prepare.py` | |
| Diagnostics | `diag_shrinkage_all_splits.py`, `diag_e7_health_insurance.py`, `diag_e6_*.py`, `diag_creditcard_coverage_*.py` | |
| Tables and Figure 1 of the manuscript | `make_paper_assets.py` (writes `paper_assets/`) | |

Each run writes one JSON line per replication or split and resumes where it stopped when rerun. Each report prints the tables and a list of acceptance checks (completeness, absence of NaN, conformal coverage, and the deterministic identities of the theory) and exits with a non-zero code if a check fails. Pre-registered predictions are printed separately and never used as checks.

## Reproducing

Python 3.12:

```bash
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Data.** The data sets are public but not redistributed here. `scripts/e4_prepare.py` writes `data/e4/*.npz`; it downloads the OpenML data sets and expects three files from their sources: `data/pima_diabetes.csv` (PIMA Indians Diabetes, UCI, response column `outcome`), `data/superconductivity/train.csv` (UCI Superconductivity) and `data/application_train.csv` (Kaggle, Home Credit Default Risk). For E8 it reads the CSVs of the tabular benchmark of Grinsztajn et al. (2022) from `data/e8_raw/`, downloaded from [inria-soda/tabular-benchmark](https://huggingface.co/datasets/inria-soda/tabular-benchmark) with the folder layout of that repository (`clf_cat/`, `clf_num/`, `reg_cat/`, `reg_num/`); the files used are listed in `GRINSZTAJN` in `e4_prepare.py`, and their checksums are in `checksums/e8_raw.sha256`.

**Runs** (CPU only):

```bash
python3 theory/theory_checks.py
python3 scripts/e5_mechanism.py
bash scripts/run_all.sh <workers>       # E1, E2, E3, E4, E4b
bash scripts/run_e6.sh <workers>
bash scripts/run_e7.sh <workers>
bash scripts/run_e8.sh <workers>
bash scripts/run_e9.sh <workers>       # E9a and the diagnostics of Section 4.5
bash scripts/report_all.sh && bash scripts/report_e6.sh && bash scripts/report_e7.sh && bash scripts/report_e8.sh
python3 scripts/make_paper_assets.py
```

On a 20-core machine, `run_all.sh` takes about 17 hours, `run_e6.sh` and `run_e7.sh` about 1.5 hours each, `run_e8.sh` about 2.5 hours and `run_e9.sh` about half an hour; the other steps take minutes. The experiment scripts (`e1_mirror.py`, `e2_main.py`, `e4_main.py`, `e4b_main.py`, `e5_mechanism.py`) also have a `--quick` mode for a smoke test. Results are reproducible bit for bit on the same machine and software versions.
