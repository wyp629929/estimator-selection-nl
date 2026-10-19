"""E6 diagnostic: the one t3/threshold replication where CP (alpha 0.05, 0.1) picks xgboost at n=4000.
Is it the crossing of Proposition 2(b): large xgboost errors on a set of probability below alpha?"""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import e2_main as E

rep, regime = int(sys.argv[1]), 'threshold'
cfg = E.VARIANTS['noise_t3']
rng = np.random.default_rng([E.BASE_SEED, E.REGIMES.index(regime), rep])
X, y, _ = E.draw(regime, E.N_TRAIN, rng, cfg)
models, dnn = E.fit_models(X, y, E.BASE_SEED + rep, torch.device('cpu'))
Xi, _, fi = E.draw(regime, E.N_INT, rng, cfg)
P = E.predict_all(models, dnn, Xi, 'cpu')
g = fi[:, None] - P
j, k = E.MODELS.index('xgboost'), E.MODELS.index('rf')
print('G - sigma^2:', {m: round(float(v), 3) for m, v in zip(E.MODELS, (g ** 2).mean(axis=0))})
for thr in (1, 2, 3, 5):
    big = np.abs(g[:, j]) > thr
    print(f'P(|g_xgb| > {thr}) = {big.mean():.4f}; share of xgb excess squared error from these points: '
          f'{(g[big, j] ** 2).sum() / (g[:, j] ** 2).sum():.2f}; max |g_xgb| = {np.abs(g[:, j]).max():.1f}')
print(f'P(|g_xgb| < |g_rf|) = {np.mean(np.abs(g[:, j]) < np.abs(g[:, k])):.3f}')
