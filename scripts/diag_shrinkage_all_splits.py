#!/usr/bin/env python3
"""Post hoc diagnostic over all splits: when CP width and validation MSE disagree on an (near-)binary
outcome, is it the shrinkage pattern (Proposition 3(c) and its analogue) rather than a crossing?

For each split (full calibration set, given alpha), refit the 7 models as in E4/E7 and record the
CP pick c and the val pick v. In splits with c != v, the shrinkage pattern means:
  (S1) c's test predictions are more concentrated: sd(f_c) < sd(f_v);
  (S2) c is nevertheless worse in test MSE;
  (S3) c is not better on the bulk: the mean |r| of c over calibration points within c's own cutoff
       is at least that of v within v's cutoff (no "better on most points" as in a crossing).
Also reported: whether the local refit picks the same models as the cloud run (float differences).

Usage: python3 scripts/diag_shrinkage_all_splits.py <dataset> <alpha> <results jsonl of the cloud E4b/E7b run>
"""
import os, sys, json
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ProcessPoolExecutor
import numpy as np

NAME, ALPHA, CLOUD = sys.argv[1], float(sys.argv[2]), sys.argv[3]


def one(split):
    import torch
    torch.set_num_threads(1)
    import e4_main as E4
    from e2_main import MODELS, cp_quantile
    f = E4.fit_split(NAME, split, torch.device('cpu'))
    yc, yt, Pc, Pt = f['yc'], f['yt'], f['Pc'], f['Pt']
    rc = np.abs(yc[:, None] - Pc)
    q = np.array([cp_quantile(rc[:, j], ALPHA) for j in range(len(MODELS))])
    c, v = int(np.argmin(q)), int(np.argmin(np.mean(rc ** 2, axis=0)))
    mse = ((yt[:, None] - Pt) ** 2).mean(axis=0)
    body = np.array([rc[rc[:, j] <= q[j], j].mean() for j in range(len(MODELS))])
    return dict(split=split, cp=MODELS[c], val=MODELS[v], sd_c=float(Pt[:, c].std()), sd_v=float(Pt[:, v].std()),
                mse_c=float(mse[c]), mse_v=float(mse[v]), body_c=float(body[c]), body_v=float(body[v]),
                width_c=float(2 * q[c]), width_v=float(2 * q[v]))


if __name__ == '__main__':
    cloud = {}
    for l in open(CLOUD):
        x = json.loads(l)
        e = x['by_n'][str(max(int(n) for n in x['by_n']))]
        cloud[x['split']] = (e['cp'][str(ALPHA)]['choice'], e['val']['choice'])
    with ProcessPoolExecutor(6) as ex:
        res = sorted(ex.map(one, sorted(cloud)), key=lambda r: r['split'])
    same = sum((r['cp'], r['val']) == cloud[r['split']] for r in res)
    dis = [r for r in res if r['cp'] != r['val']]
    s1 = sum(r['sd_c'] < r['sd_v'] for r in dis); s2 = sum(r['mse_c'] > r['mse_v'] for r in dis)
    s3 = sum(r['body_c'] >= r['body_v'] for r in dis); allp = sum(r['sd_c'] < r['sd_v'] and r['mse_c'] > r['mse_v']
                                                                   and r['body_c'] >= r['body_v'] for r in dis)
    print(f'{NAME}, alpha={ALPHA}: local refit picks the same (CP, val) as the cloud run in {same}/{len(res)} splits')
    print(f'  splits with CP != val: {len(dis)}/{len(res)}; picks: ' +
          str(sorted({(r["cp"], r["val"]) for r in dis}, key=str)))
    print(f'  S1 sd(f_c) < sd(f_v): {s1}/{len(dis)}   S2 MSE_c > MSE_v: {s2}/{len(dis)}   '
          f'S3 bulk mean |r| of c >= of v: {s3}/{len(dis)}   all three: {allp}/{len(dis)}')
    for r in dis:
        print(f"   split {r['split']:2d}: {r['cp']:>8s} vs {r['val']:<8s} sd {r['sd_c']:.3f}/{r['sd_v']:.3f}  "
              f"width {r['width_c']:.3f}/{r['width_v']:.3f}  MSE {r['mse_c']:.4g}/{r['mse_v']:.4g}  bulk |r| {r['body_c']:.3f}/{r['body_v']:.3f}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'diag',
                       f'shrinkage_{NAME}_a{ALPHA}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1)
