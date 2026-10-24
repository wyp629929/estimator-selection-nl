#!/usr/bin/env python3
"""E9a (Section 4.5 and Appendix D of the manuscript): the binary-outcome results with predictions clipped to [0, 1].

With predictions in [0, 1], |Y - phat| = 1 - phat_Y, the score of the least ambiguous set-valued
classifier (Remark 6), so ranking by the width is ranking by that classifier's threshold. For every
binary dataset of E4 and E8 and every split (same splits and fits as E4b/E8b, refitted here), this
records, both without and with clipping, at the full calibration part and alpha in {0.05, 0.1, 0.2}:
the CP pick, the val pick (calibration Brier), each model's test Brier, e per model, the share of
positives above each model's cutoff, the sd of test predictions and the mean |r| within each
model's cutoff (pattern S). Both versions come from the same refit, so machine differences cancel.

Output: results/e9a/<dataset>.jsonl, resumable.
Usage: python3 scripts/diag_e9a_clipped.py --workers 19
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from e2_main import MODELS, cp_quantile, done_reps  # noqa: E402
from e4_main import E8_BINARY, SPLITS, fit_split  # noqa: E402

OUT = os.path.join(BASE_DIR, '..', 'results', 'e9a')
BINARY = ['pima', 'homecredit', 'creditcard', 'mammography', 'bank', 'default', 'adult'] + E8_BINARY
ALPHAS = [0.05, 0.1, 0.2]


def summarize(yc, yt, Pc, Pt):
    rc = np.abs(yc[:, None] - Pc)
    mse = ((yt[:, None] - Pt) ** 2).mean(axis=0)
    v = int(np.argmin(np.mean(rc ** 2, axis=0)))
    out = dict(model_mse=mse.tolist(), val=MODELS[v], e=np.mean(rc >= 0.5, axis=0).tolist(),
               sd_test=Pt.std(axis=0).tolist(), cp={})
    pos = yc == 1
    for a in ALPHAS:
        q = np.array([cp_quantile(rc[:, j], a) for j in range(len(MODELS))])
        c = int(np.argmin(q))
        out['cp'][str(a)] = dict(
            choice=MODELS[c], widths=(2 * q).tolist(),
            pos_above=[float(np.mean(rc[pos, j] > q[j])) for j in range(len(MODELS))] if pos.any() else None,
            body=[float(rc[rc[:, j] <= q[j], j].mean()) for j in range(len(MODELS))])
    return out


def one(name, split):
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    f = fit_split(name, split, torch.device('cpu'))
    yc, yt, Pc, Pt = f['yc'], f['yt'], f['Pc'], f['Pt']
    return dict(dataset=name, split=split, n_cal=len(yc),
                outside=float(np.mean((Pt < 0) | (Pt > 1))),
                raw=summarize(yc, yt, Pc, Pt),
                clipped=summarize(yc, yt, np.clip(Pc, 0, 1), np.clip(Pt, 0, 1)),
                seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--datasets', nargs='+', default=BINARY)
    ap.add_argument('--splits', type=int, default=None, help='smoke test: only the first N splits of each dataset')
    ap.add_argument('--out-dir', default=OUT)
    args = ap.parse_args()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    done = {k: done_reps(os.path.join(out, f'{k}.jsonl'), key='split') for k in args.datasets}
    tasks = [(k, s) for k in args.datasets for s in range(min(SPLITS[k], args.splits or SPLITS[k])) if s not in done[k]]
    print(f'[E9a] {len(tasks)} splits to run ({args.workers} workers) -> {out}', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, k, s) for k, s in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            with open(os.path.join(out, f"{row['dataset']}.jsonl"), 'a') as f:
                f.write(json.dumps(row) + '\n')
            print(f"  {i}/{len(tasks)} ({row['dataset']} split {row['split']}, {row['seconds']:.0f}s), "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print('All splits finished.', flush=True)


if __name__ == '__main__':
    main()
