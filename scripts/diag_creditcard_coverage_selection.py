"""Population = the 12,500 test residuals of creditcard RF (split 9). Draw calibration sets of 250,
compute split-CP coverage on the remaining points, and compare all draws with the draws whose
cutoff is tie-free (the filter used by the acceptance check)."""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import e4_main as E4
from e2_main import MODELS, cp_quantile
j = MODELS.index('rf')
f = E4.fit_split('creditcard', 9, torch.device('cpu'))
pool = np.abs(f['yt'] - f['Pt'][:, j])
rng = np.random.default_rng(1)
cov, free = [], []
for _ in range(20000):
    idx = rng.permutation(len(pool)); c, t = pool[idx[:250]], pool[idx[250:]]
    q = cp_quantile(c, 0.05)
    cov.append(np.mean(t <= q)); free.append(np.sum(c == q) == 1)
cov, free = np.array(cov), np.array(free)
print(f"all draws: mean coverage {cov.mean():.4f} (bound 1-a+1/(n+1) = {0.95 + 1/251:.4f}); "
      f"tie-free-at-cutoff draws ({free.mean():.2f} of all): mean {cov[free].mean():.4f}; others: {cov[~free].mean():.4f}")
print(f"distinct residual values in the pool: {len(np.unique(pool))}; share of the most common value: {np.max(np.unique(pool, return_counts=True)[1]) / len(pool):.3f}")
