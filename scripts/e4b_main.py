#!/usr/bin/env python3
"""E4b: what the split-CP width measures on real data, across CP levels and calibration sizes.

Uses exactly the splits and fitted models of E4 (e4_main.fit_split). For every split, every
alpha in ALPHAS and every calibration size n (nested subsamples of the calibration part, plus
the full part), it records each model's split-CP width, the model the width selects, the model
validation MSE selects, and their test risks. For binary outcomes it also records, per model,
the share of positives (y = 1) whose residual exceeds the cutoff q: those positives do not
affect the width at all, whatever their predictions are.

What the theory says (Lemma 3 and Proposition 3 of the manuscript): the width 2q does
not depend on the residuals above q, and at most n - k residuals lie above q. For binary outcomes
the regime is set by e = P(|r| >= 1/2), a property of each model: e < alpha puts every point with
|r| >= 1/2 above the cutoff (width set by the well-predicted points), e > alpha puts the cutoff at
or above 1/2 (shrinking predictions towards 1/2 never widens the interval). The final rerun
(2026-10-16) records e per model and the number of calibration scores tied at the cutoff.

Output: one JSON line per (dataset, split) in <out-dir>/<dataset>.jsonl; resumable.

Usage:
  python3 scripts/e4b_main.py --quick
  python3 scripts/e4b_main.py --workers 16
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
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from e2_main import MODELS, cp_quantile, done_reps, n_at_cut  # noqa: E402
from e4_main import BASE_SEED, DATA_DIR, DATASET_IDS, E4_DATASETS, E7_DATASETS, SPLITS, fit_split  # noqa: E402

OUT_DIR = os.path.join(BASE_DIR, '..', 'results', 'e4b')
SETS = {'e4': (E4_DATASETS, OUT_DIR), 'e7': (E7_DATASETS, os.path.join(BASE_DIR, '..', 'results', 'e7', 'e4b'))}
ALPHAS = [0.05, 0.1, 0.2]
N_CAL = [250, 1000, 4000]          # plus the full calibration part


def cp_entry(r_abs, ycal, alpha, binary):
    """Width per model and, for binary outcomes, the share of positives whose residual exceeds the
    cutoff q = width / 2 (these positives have no influence on the width)."""
    w = np.array([2 * cp_quantile(r_abs[:, j], alpha) for j in range(r_abs.shape[1])])
    invisible = None
    if binary and (ycal == 1).any():
        pos = ycal == 1
        invisible = [float(np.mean(r_abs[pos, j] > w[j] / 2)) for j in range(r_abs.shape[1])]
    return w, invisible


def one_split(name, split, device_name, max_rows=None):
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    f = fit_split(name, split, torch.device(device_name), max_rows)
    Pc, Pt, yc, yt = f['Pc'], f['Pt'], f['yc'], f['yt']
    model_mse = ((yt[:, None] - Pt) ** 2).mean(axis=0)
    binary = bool(np.isin(np.unique(np.concatenate([yc, yt])), [0.0, 1.0]).all())
    sizes = [n for n in N_CAL if n < len(yc)] + [len(yc)]
    # M1' (Appendix D of the manuscript): a fixed random half A of the test set selects costly cells,
    # the other half B is used for the bulk/tail decomposition.
    half_a = np.zeros(len(yt), bool)
    half_a[np.random.default_rng([BASE_SEED, DATASET_IDS[name], split, 1]).permutation(len(yt))[:len(yt) // 2]] = True

    def m1prime(c, v, a):
        lc, lv = (yt - Pt[:, c]) ** 2, (yt - Pt[:, v]) ** 2
        d, gap = (lc - lv)[~half_a], np.abs(Pt[:, c] - Pt[:, v])[~half_a]
        tail = np.zeros(len(d), bool)
        tail[np.argsort(-gap, kind='stable')[:int(np.ceil(a * len(d)))]] = True     # largest prediction gaps
        return dict(mse_ratio_A=float(lc[half_a].mean() / lv[half_a].mean()),
                    bulk_d=float(d[~tail].mean()), tail_d=float(d[tail].mean()))

    by_n = {}
    for n in sizes:
        r_abs = np.abs(yc[:n, None] - Pc[:n])
        v = int(np.argmin(np.mean(r_abs ** 2, axis=0)))
        entry = {'val': dict(choice=MODELS[v], mse=float(model_mse[v])), 'cp': {},
                 'n_pos': int(yc[:n].sum()) if binary else None,
                 'e': np.mean(r_abs >= 0.5, axis=0).tolist() if binary else None,   # e of Proposition 3, per model
                 # share of distinct |r| values: the coverage upper bound needs continuous scores (E7 rule)
                 'distinct_share': [float(len(np.unique(r_abs[:, j])) / n) for j in range(len(MODELS))]}
        for a in ALPHAS:
            w, invisible = cp_entry(r_abs, yc[:n], a, binary)
            c = int(np.argmin(w))
            entry['cp'][str(a)] = dict(
                widths=w.tolist(), choice=MODELS[c], mse=float(model_mse[c]),
                kendall_width_vs_risk=float(stats.kendalltau(w, model_mse)[0]),
                pos_invisible=invisible,
                n_at_cut=[n_at_cut(r_abs[:, j], a) for j in range(len(MODELS))],
                coverage=np.mean(np.abs(yt[:, None] - Pt) <= w / 2, axis=0).tolist(),
                # share of test points where the CP pick has a smaller absolute error than the val pick
                win_share=(None if c == v else
                           float(np.mean(np.abs(yt - Pt[:, c]) < np.abs(yt - Pt[:, v])))),
                m1prime=None if c == v else m1prime(c, v, a))
        by_n[str(n)] = entry

    return dict(dataset=name, split=split, seed=f['seed'], binary=binary,
                prevalence_cal=float(yc.mean()) if binary else None,
                n_train=f['n_train'], n_cal=f['n_cal'], n_test=f['n_test'],
                model_mse=model_mse.tolist(), by_n=by_n, seconds=time.time() - t0)


def main():
    available = [k for k in DATASET_IDS if os.path.exists(os.path.join(DATA_DIR, f'{k}.npz'))]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', default='e4', choices=list(SETS), help='e4 = the 8 E4 datasets; e7 = the 20 CTR23 datasets')
    ap.add_argument('--datasets', nargs='+', default=None, choices=available, help='default: all datasets of --set')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    ap.add_argument('--out-dir', default=None, help='default: results/e4b (set e4) or results/e7/e4b (set e7)')
    ap.add_argument('--quick', action='store_true', help='smoke test: 2 splits per dataset, at most 4000 rows')
    args = ap.parse_args()
    args.datasets = args.datasets or SETS[args.set][0]
    if args.out_dir is None:
        args.out_dir = SETS[args.set][1] + ('_quick' if args.quick else '')
    splits = {k: (2 if args.quick else SPLITS[k]) for k in args.datasets}
    max_rows = 4000 if args.quick else None
    os.makedirs(args.out_dir, exist_ok=True)

    import lightgbm, sklearn, torch, xgboost
    meta = dict(settings=dict(datasets=args.datasets, splits=splits, alphas=ALPHAS, n_cal=N_CAL, models=MODELS,
                              base_seed=BASE_SEED, dataset_ids={k: DATASET_IDS[k] for k in args.datasets},
                              max_rows=max_rows, device=args.device, quick=args.quick),
                environment=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                                 xgboost=xgboost.__version__, lightgbm=lightgbm.__version__, torch=torch.__version__,
                                 machine=platform.platform(), cpu_count=os.cpu_count()))
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    done = {k: done_reps(os.path.join(args.out_dir, f'{k}.jsonl'), key='split') for k in args.datasets}
    cost = {k: int(np.prod(np.load(os.path.join(DATA_DIR, f'{k}.npz'))['X'].shape)) for k in args.datasets}
    tasks = sorted(((k, s) for k in args.datasets for s in range(splits[k]) if s not in done[k]),
                   key=lambda t: -cost[t[0]])          # largest rows x features first, to shorten the tail
    print(f'[E4b] {len(tasks)} splits to run ({args.workers} workers) -> {args.out_dir}', flush=True)
    t0, finished = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_split, k, s, args.device, max_rows): (k, s) for k, s in tasks}
        for fut in as_completed(futs):
            row = fut.result()
            with open(os.path.join(args.out_dir, f'{row["dataset"]}.jsonl'), 'a') as f:
                f.write(json.dumps(row) + '\n')
            finished += 1
            print(f'  {finished}/{len(tasks)} done ({row["dataset"]} split {row["split"]}, {row["seconds"]:.0f}s), '
                  f'{(time.time() - t0) / 60:.1f} min elapsed', flush=True)
    print('All splits finished.', flush=True)


if __name__ == '__main__':
    main()
