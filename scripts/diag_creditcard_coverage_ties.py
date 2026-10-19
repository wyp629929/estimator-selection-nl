import os, sys, json
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import e4_main as E4
from e2_main import MODELS, cp_quantile, n_at_cut
rows = {json.loads(l)['split']: json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'e4b', 'creditcard.jsonl'))}
j = MODELS.index('rf')
for split in map(int, sys.argv[1:]):
    f = E4.fit_split('creditcard', split, torch.device('cpu'))
    rc = np.abs(f['yc'][:250] - f['Pc'][:250, j]); rt = np.abs(f['yt'] - f['Pt'][:, j])
    q = cp_quantile(rc, 0.05)
    cloud = rows[split]['by_n']['250']['cp']['0.05']
    print(f"split {split}: cutoff {q:.6f} (cloud width/2 {cloud['widths'][j]/2:.6f}), calib ties at cutoff {n_at_cut(rc, 0.05)} "
          f"(cloud {cloud['n_at_cut'][j]}), test share == cutoff {np.mean(rt == q):.4f}, "
          f"coverage <= cutoff {np.mean(rt <= q):.4f} (cloud {cloud['coverage'][j]:.4f}), coverage < cutoff {np.mean(rt < q):.4f}, "
          f"distinct test residuals {len(np.unique(rt))}")

# Randomized tie-breaking: rank of each test score among the n calibration scores plus itself,
# ties broken uniformly at random; covered iff rank <= k. Exchangeability gives P = k/(n+1) exactly.
def randomized_coverage(rc, rt, alpha, rng):
    n = len(rc); k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    s = np.sort(rc)
    below = np.searchsorted(s, rt, side='left'); equal = np.searchsorted(s, rt, side='right') - below
    rank = below + 1 + np.floor(rng.random(len(rt)) * (equal + 1))      # position of the test score among n+1
    return float(np.mean(rank <= k)), k / (n + 1)

rng = np.random.default_rng(0)
for split in map(int, sys.argv[1:]):
    f = E4.fit_split('creditcard', split, torch.device('cpu'))
    rc = np.abs(f['yc'][:250] - f['Pc'][:250, j]); rt = np.abs(f['yt'] - f['Pt'][:, j])
    cov, target = randomized_coverage(rc, rt, 0.05, rng)
    print(f"split {split}: distinct calib residuals {len(np.unique(rc))}/250, randomized-tie coverage {cov:.4f} vs k/(n+1) = {target:.4f}")
