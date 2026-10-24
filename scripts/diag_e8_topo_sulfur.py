#!/usr/bin/env python3
"""E8 post hoc diagnostics (Section 4.5 and Appendix D of the manuscript), on all 20 splits, refitted with fit_split.

topo_2_1: for each split and level where the CP and val picks differ, coverage at the val pick's
  radius on the test points with the highest 5% of outcomes, and the ratio of squared errors on the
  middle half of the outcomes (crossing along the range of the outcome, Proposition 2(d)).
sulfur: for each model, the share of its calibration MSE contributed by its largest 1% of absolute
  residuals; whether the val pick changes when those residuals are removed (a diagnostic only).

Output: results/diag/e8_topo_sulfur.json
Usage: python3 scripts/diag_e8_topo_sulfur.py --workers 19
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from e2_main import MODELS, cp_quantile  # noqa: E402


def one(args):
    name, split = args
    import torch
    torch.set_num_threads(1)
    import e4_main as E4
    f = E4.fit_split(name, split, torch.device('cpu'))
    yc, yt, Pc, Pt = f['yc'], f['yt'], f['Pc'], f['Pt']
    rc = np.abs(yc[:, None] - Pc)
    v = int(np.argmin((rc ** 2).mean(axis=0)))
    out = dict(dataset=name, split=split, val=MODELS[v], test_mse=((yt[:, None] - Pt) ** 2).mean(axis=0).tolist())
    if name == 'topo_2_1':
        hi = yt >= np.quantile(yt, 0.95)
        mid = (yt >= np.quantile(yt, 0.25)) & (yt <= np.quantile(yt, 0.75))
        out['cells'] = {}
        for a in (0.05, 0.1, 0.2):
            q = np.array([cp_quantile(rc[:, j], a) for j in range(len(MODELS))])
            c = int(np.argmin(q))
            if c == v:
                continue
            rt_c, rt_v = np.abs(yt - Pt[:, c]), np.abs(yt - Pt[:, v])
            out['cells'][str(a)] = dict(cp=MODELS[c], cover_hi_c=float(np.mean(rt_c[hi] <= q[v])),
                                        cover_hi_v=float(np.mean(rt_v[hi] <= q[v])),
                                        mid_mse_ratio=float(np.mean(rt_c[mid] ** 2) / np.mean(rt_v[mid] ** 2)),
                                        cover_mid_c=float(np.mean(rt_c[mid] <= q[v])), cover_mid_v=float(np.mean(rt_v[mid] <= q[v])))
    else:
        sq = rc ** 2
        top = np.quantile(rc, 0.99, axis=0)
        out['top1_share'] = [float(sq[rc[:, j] > top[j], j].sum() / sq[:, j].sum()) for j in range(len(MODELS))]
        trimmed = [float(sq[rc[:, j] <= top[j], j].mean()) for j in range(len(MODELS))]
        out['val_trimmed'] = MODELS[int(np.argmin(trimmed))]
        out['cp'] = {str(a): MODELS[int(np.argmin([cp_quantile(rc[:, j], a) for j in range(len(MODELS))]))] for a in (0.05, 0.1, 0.2)}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--splits', type=int, default=20, help='smoke test: fewer splits')
    ap.add_argument('--out', default=os.path.join(BASE_DIR, '..', 'results', 'diag', 'e8_topo_sulfur.json'))
    args = ap.parse_args()
    tasks = [(k, s) for k in ('topo_2_1', 'sulfur') for s in range(args.splits)]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(one, tasks))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=1)
    print('wrote', args.out)
