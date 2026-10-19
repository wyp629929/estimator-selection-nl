#!/usr/bin/env python3
"""E4: estimator selection on real data, evaluated directly by held-out risk.

For any selector S fitted on calibration data, the test MSE of the composite predictor
x -> fhat_{S(x)}(x) is an unbiased estimate of its risk E[D(S(X)|X)]. Differences between
selectors are therefore differences in local regret, and no knowledge of f0 is needed.

Per random split: 50% train / 25% calibration / 25% test. Missing values are imputed with
training medians and covariates are standardized with training statistics. The 7 estimators
and all selectors are exactly those of E2 (imported from e2_main.py).

Data: data/e4/<name>.npz, produced by e4_prepare.py.
Output: one JSON line per (dataset, split) in <out-dir>/<dataset>.jsonl; resumable.

Usage:
  python3 scripts/e4_main.py --quick              # smoke test
  python3 scripts/e4_main.py --workers 16         # full run
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from e2_main import (CV_FOLDS, GLOBAL_SEL, LOCAL_SEL, MODELS, cp_quantile, done_reps, fit_models,  # noqa: E402
                     local_selectors, n_at_cut, predict_all)

DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'e4')
OUT_DIR = os.path.join(BASE_DIR, '..', 'results', 'e4')
BASE_SEED = 2024
# Seed ids are fixed: a dataset keeps its id (and hence its splits and fitted models) when others are added.
DATASET_IDS = {'pima': 0, 'supercond': 1, 'homecredit': 2,
               'creditcard': 3, 'mammography': 4, 'bank': 5, 'default': 6, 'adult': 7}   # last five added 2026-10-15
E4_DATASETS = list(DATASET_IDS)
# E7 (added 2026-10-18, Appendix D of the manuscript): the 20 OpenML-CTR23 regression datasets with at
# least 4,000 rows except superconductivity. Ids 8-27, so the E4 datasets keep theirs.
E7_DATASETS = ['abalone', 'white_wine', 'cpu_activity', 'kin8nm', 'pumadyn32nh', 'grid_stability',
               'brazilian_houses', 'naval_propulsion_plant', 'miami_housing', 'fifa', 'california_housing',
               'kings_county', 'health_insurance', 'fps_benchmark', 'cps88wages', 'physiochemical_protein',
               'sarcos', 'diamonds', 'video_transcoding', 'wave_energy']
DATASET_IDS.update({k: 8 + i for i, k in enumerate(E7_DATASETS)})
DATASETS = list(DATASET_IDS)
SPLITS = {'pima': 50, 'supercond': 20, 'homecredit': 20,
          'creditcard': 20, 'mammography': 20, 'bank': 20, 'default': 20, 'adult': 20}
SPLITS.update({k: 20 for k in E7_DATASETS})
SETS = {'e4': (E4_DATASETS, OUT_DIR), 'e7': (E7_DATASETS, os.path.join(BASE_DIR, '..', 'results', 'e7', 'e4'))}
FRAC_TRAIN, FRAC_CAL = 0.5, 0.25
CP_ALPHA = 0.10


def fit_split(name, split, device, max_rows=None):
    """Random 50/25/25 split, median imputation and standardization with training statistics,
    the 7 estimators fitted on the training part, and their predictions on calibration and test."""
    seed = BASE_SEED + split
    d = np.load(os.path.join(DATA_DIR, f'{name}.npz'))
    X, y = d['X'].astype(float), d['y']
    rng = np.random.default_rng([BASE_SEED, DATASET_IDS[name], split])
    perm = rng.permutation(len(y))
    if max_rows:
        perm = perm[:max_rows]
    n = len(perm)
    n_tr, n_ca = int(FRAC_TRAIN * n), int(FRAC_CAL * n)
    tr, ca, te = perm[:n_tr], perm[n_tr:n_tr + n_ca], perm[n_tr + n_ca:]

    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)                    # columns missing in all training rows
    X = np.where(np.isnan(X), med, X)
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd                                          # training statistics only

    models, dnn = fit_models(Z[tr], y[tr], seed, device)
    return dict(seed=seed, Zc=Z[ca], Zt=Z[te], yc=y[ca], yt=y[te], n_train=len(tr), n_cal=len(ca), n_test=len(te),
                Pc=predict_all(models, dnn, Z[ca], device), Pt=predict_all(models, dnn, Z[te], device))


def one_split(name, split, device_name, max_rows=None):
    import torch
    torch.set_num_threads(1)
    device = torch.device(device_name)
    t0 = time.time()
    f = fit_split(name, split, device, max_rows)
    seed, Pc, Pt, yc, yt = f['seed'], f['Pc'], f['Pt'], f['yc'], f['yt']
    Lt = (yt[:, None] - Pt) ** 2                               # test loss of every model at every point
    model_mse = Lt.mean(axis=0)

    Rc = yc[:, None] - Pc
    w = np.array([2 * cp_quantile(np.abs(Rc[:, j]), CP_ALPHA) for j in range(len(MODELS))])
    out = {}
    for sname, score in [('cp', w), ('val', np.mean(Rc ** 2, axis=0))]:
        s = int(np.argmin(score))
        out[sname] = dict(choice=MODELS[s], mse=float(model_mse[s]))
    sel, U_t, level = local_selectors(f['Zc'], yc, Pc, f['Zt'], Pt, seed)
    for sname in LOCAL_SEL:
        out[sname] = dict(mse=float(Lt[np.arange(len(yt)), sel[sname]].mean()))
    for sname, lev in level.items():
        out[sname]['level'] = str(lev)
    binary = bool(np.isin(np.unique(np.concatenate([yc, yt])), [0.0, 1.0]).all())

    return dict(dataset=name, split=split, seed=seed, n_train=f['n_train'], n_cal=f['n_cal'], n_test=f['n_test'],
                model_mse=model_mse.tolist(), selectors=out,
                cp_coverage=np.mean(np.abs(yt[:, None] - Pt) <= w / 2, axis=0).tolist(),
                cp_n_at_cut=[n_at_cut(np.abs(Rc[:, j]), CP_ALPHA) for j in range(len(MODELS))],
                cp_distinct_share=[float(len(np.unique(np.abs(Rc[:, j]))) / len(Rc)) for j in range(len(MODELS))],
                binary=binary, e_cal=np.mean(np.abs(Rc) >= 0.5, axis=0).tolist() if binary else None,
                U_joint_coverage=float(np.mean(np.all(Lt - Lt.min(axis=1, keepdims=True) <= U_t, axis=1))),
                seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', default='e4', choices=list(SETS), help='e4 = the 8 E4 datasets; e7 = the 20 CTR23 datasets')
    ap.add_argument('--datasets', nargs='+', default=None, choices=DATASETS, help='default: all datasets of --set')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    ap.add_argument('--out-dir', default=None, help='default: results/e4 (set e4) or results/e7/e4 (set e7)')
    ap.add_argument('--quick', action='store_true', help='smoke test: 2 splits per dataset, at most 4000 rows')
    args = ap.parse_args()
    args.datasets = args.datasets or SETS[args.set][0]
    if args.out_dir is None:
        args.out_dir = SETS[args.set][1] + ('_quick' if args.quick else '')
    splits = {k: 2 for k in DATASETS} if args.quick else SPLITS
    max_rows = 4000 if args.quick else None
    os.makedirs(args.out_dir, exist_ok=True)

    import lightgbm, sklearn, torch, xgboost
    meta = dict(settings=dict(datasets=args.datasets, splits={k: splits[k] for k in args.datasets},
                              frac_train=FRAC_TRAIN, frac_cal=FRAC_CAL, cp_alpha=CP_ALPHA, cv_folds=CV_FOLDS, models=MODELS,
                              global_selectors=GLOBAL_SEL, local_selectors=LOCAL_SEL, base_seed=BASE_SEED,
                              max_rows=max_rows, device=args.device, quick=args.quick),
                environment=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                                 xgboost=xgboost.__version__, lightgbm=lightgbm.__version__, torch=torch.__version__,
                                 machine=platform.platform(), cpu_count=os.cpu_count()))
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    done = {k: done_reps(os.path.join(args.out_dir, f'{k}.jsonl'), key='split') for k in args.datasets}
    # largest datasets first so that the long tasks do not end up in the tail
    tasks = [(k, s) for k in reversed(args.datasets) for s in range(splits[k]) if s not in done[k]]
    print(f'[E4] {len(tasks)} splits to run ({args.workers} workers) -> {args.out_dir}', flush=True)
    t0, finished = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_split, k, s, args.device, max_rows): (k, s) for k, s in tasks}
        for fut in as_completed(futs):
            row = fut.result()
            with open(os.path.join(args.out_dir, f'{row["dataset"]}.jsonl'), 'a') as f:
                f.write(json.dumps(row) + '\n')
            finished += 1
            el = time.time() - t0
            print(f'  {finished}/{len(tasks)} done ({row["dataset"]} split {row["split"]}, {row["seconds"]:.0f}s), '
                  f'{el / 60:.1f} min elapsed', flush=True)
    print('All splits finished.', flush=True)


if __name__ == '__main__':
    main()
