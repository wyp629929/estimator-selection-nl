#!/usr/bin/env python3
"""E1: mirror-pair experiment for Theorem 4 (information barrier, local target).

Two configurations ("worlds") differ only by the reflection T(x) = (-x1, x2, ..., xd):
world P has regression function f0, world PT has f0∘T. In each replication:

  * world P trains OLS and RF on (X, y), y = f0(X) + eps;
  * world PT trains the same estimators, with the same seeds, on (T X, y), which is a
    valid training sample from PT. OLS and RF are equivariant under coordinate
    reflection, so the PT fits equal fhat∘T (checked numerically: `equivariance_maxdiff`);
  * each world draws its own calibration sample; both worlds share the test covariates.

By Theorem 4(i), the calibration residual matrix then has the same law in both worlds
(checked by two-sample KS tests: `ks_reject_rate` should be close to 0.05).

Selectors (local target):
  cp       split-CP width, alpha = 0.1                         residual-only, global
  val      validation MSE                                      residual-only, global
  bnd      boundary rule: val-MSE winner on {x1 > 0},          residual-only, x-dependent
           the other model on {x1 <= 0}
  knn      k-NN regression of r_M^2 on X, k = ceil(sqrt(n)),   conditional,   x-dependent
           argmin (Theorem 4(iv))
  bnd_orc  boundary rule with the better orientation           infeasible reference
           (chosen using the oracle)

Metrics per (world, n_cal, selector): misalignment Delta_eps for each eps in EPS
(eps = 0 is exact argmin; Remark 6 gives the bound kappa_eps / 2), and local regret
E[D(S(X)|X) - min_M D(M|X)]. Also per world: kappa_eps and the Lemma 1 floor.

Usage:
  python3 scripts/e1_mirror.py --quick            # smoke test, < 1 min
  python3 scripts/e1_mirror.py                    # full run (see README.md)
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import scipy
import sklearn
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'results', 'e1', 'e1_mirror.json')

BASE_SEED = 2024
SIGMA = 1.0
CP_ALPHA = 0.10
EPS = [0.0, 0.1, 0.5]
MODELS = ['ols', 'rf']
SELECTORS = ['cp', 'val', 'bnd', 'knn', 'bnd_orc']
KS_N = 5000                     # calibration size at which the KS checks are run

# Each config: covariate dimension d and linear coefficients (x1 only gates the regime).
CONFIGS = {
    'd3': dict(d=3, beta=[0.0, 2.0, -2.0]),
    'd5': dict(d=5, beta=[0.0, 2.0, -2.0, 1.6, 1.2]),
}


def f0(X, beta):
    """Linear everywhere, plus an interaction step on {x1 > 0} that OLS cannot fit."""
    step = 3.0 * ((X[:, 1] > 0) & (X[:, 2] > 0)) - 0.75
    return X @ beta + (X[:, 0] > 0) * step


def reflect(X):
    X = X.copy()
    X[:, 0] *= -1
    return X


def fit_models(X, y, seed):
    ols = LinearRegression().fit(X, y)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5,
                               random_state=seed, n_jobs=-1).fit(X, y)
    return [ols, rf]


def cp_quantile(abs_resid, alpha=CP_ALPHA):
    """Split-conformal quantile: the k-th smallest score, k = ceil((n + 1)(1 - alpha)) (capped at n).
    Fixed 2026-10-15: the earlier version (inherited from ranking_decision_gap.py) used
    np.quantile(abs_resid, k / n, method='higher'), which returns the (k+1)-th smallest score."""
    n = len(abs_resid)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.partition(np.asarray(abs_resid), k - 1)[k - 1]


def oracle_sets(G, eps):
    """Boolean (n_points, m): model is within eps of the best local risk."""
    return G <= G.min(axis=1, keepdims=True) + eps


def select(name, R, Xc, Xt, G):
    """Return selected model index at each test point. R: calibration residuals (n, m)."""
    n_test = len(Xt)
    if name == 'cp':
        return np.full(n_test, np.argmin([cp_quantile(np.abs(R[:, j])) for j in range(R.shape[1])]))
    if name == 'val':
        return np.full(n_test, np.argmin(np.mean(R ** 2, axis=0)))
    if name == 'bnd':
        winner = int(np.argmin(np.mean(R ** 2, axis=0)))
        return np.where(Xt[:, 0] > 0, winner, 1 - winner)
    if name == 'knn':
        k = int(np.ceil(np.sqrt(len(Xc))))
        eta = np.column_stack([KNeighborsRegressor(n_neighbors=k).fit(Xc, R[:, j] ** 2).predict(Xt)
                               for j in range(R.shape[1])])
        return np.argmin(eta, axis=1)
    if name == 'bnd_orc':
        right = np.where(Xt[:, 0] > 0, 1, 0)
        a, b = right, 1 - right
        reg = lambda s: np.mean(G[np.arange(n_test), s] - G.min(axis=1))
        return a if reg(a) <= reg(b) else b
    raise ValueError(name)


def run_config(cfg_name, cfg, reps, n_train, n_cal_list, n_test):
    d, beta = cfg['d'], np.array(cfg['beta'])
    n_max = max(n_cal_list)
    shape = (reps, 2, len(n_cal_list), len(SELECTORS))
    delta = np.zeros(shape + (len(EPS),))
    regret = np.zeros(shape)
    kappa = np.zeros((reps, 2, len(EPS)))
    floor = np.zeros((reps, 2, len(EPS)))
    ks_p = np.full((reps, 3), np.nan)          # r_ols, r_rf, r_ols - r_rf, world P vs PT
    equiv = np.zeros((reps, len(MODELS)))
    gmse = np.zeros((reps, 2, len(MODELS)))   # global risk (excluding sigma^2) per world/model

    t0 = time.time()
    for rep in range(reps):
        seed = BASE_SEED + rep
        rng = np.random.default_rng([seed, 0])
        X_tr = rng.standard_normal((n_train, d))
        y_tr = f0(X_tr, beta) + SIGMA * rng.standard_normal(n_train)
        Xt = rng.standard_normal((n_test, d))                      # shared test covariates

        worlds = [
            dict(f=lambda X: f0(X, beta), fits=fit_models(X_tr, y_tr, seed)),
            dict(f=lambda X: f0(reflect(X), beta), fits=fit_models(reflect(X_tr), y_tr, seed)),
        ]
        for j in range(len(MODELS)):
            equiv[rep, j] = np.max(np.abs(worlds[1]['fits'][j].predict(Xt)
                                          - worlds[0]['fits'][j].predict(reflect(Xt))))

        resid_ks = []
        for w, world in enumerate(worlds):
            fits, fw = world['fits'], world['f']
            # local risk minus sigma^2 (common to all models, so it does not affect any comparison)
            G = np.column_stack([(fw(Xt) - m.predict(Xt)) ** 2 for m in fits])
            G_T = np.column_stack([(fw(reflect(Xt)) - m.predict(reflect(Xt))) ** 2 for m in fits])
            gmse[rep, w] = G.mean(axis=0)
            for e, eps in enumerate(EPS):
                A, A_T = oracle_sets(G, eps), oracle_sets(G_T, eps)
                kappa[rep, w, e] = np.mean(~np.any(A & A_T, axis=1))
                floor[rep, w, e] = 1 - A.mean(axis=0).max()

            crng = np.random.default_rng([seed, 1 + w])
            Xc_all = crng.standard_normal((n_max, d))
            yc_all = fw(Xc_all) + SIGMA * crng.standard_normal(n_max)
            R_all = np.column_stack([yc_all - m.predict(Xc_all) for m in fits])
            resid_ks.append(R_all[:KS_N] if n_max >= KS_N else R_all)

            for i, n in enumerate(n_cal_list):
                R, Xc = R_all[:n], Xc_all[:n]              # nested calibration samples
                for s, name in enumerate(SELECTORS):
                    sel = select(name, R, Xc, Xt, G)
                    excess = G[np.arange(n_test), sel] - G.min(axis=1)
                    regret[rep, w, i, s] = excess.mean()
                    for e, eps in enumerate(EPS):
                        delta[rep, w, i, s, e] = np.mean(excess > eps)

        a, b = resid_ks
        ks_p[rep] = [stats.ks_2samp(a[:, 0], b[:, 0]).pvalue,
                     stats.ks_2samp(a[:, 1], b[:, 1]).pvalue,
                     stats.ks_2samp(a[:, 0] - a[:, 1], b[:, 0] - b[:, 1]).pvalue]
        if (rep + 1) % max(1, reps // 20) == 0 or rep == reps - 1:
            print(f'  [{cfg_name}] rep {rep + 1}/{reps}  {time.time() - t0:.0f}s', flush=True)

    mean_se = lambda a, ax=0: (a.mean(axis=ax).tolist(), (a.std(axis=ax, ddof=1) / np.sqrt(a.shape[ax])).tolist())
    avg_worlds = lambda a: a.mean(axis=1)    # average over the two worlds, per replication
    summary = {
        'delta_mean': mean_se(delta)[0], 'delta_se': mean_se(delta)[1],                 # [world][n][selector][eps]
        'regret_mean': mean_se(regret)[0], 'regret_se': mean_se(regret)[1],             # [world][n][selector]
        'delta_avg_worlds_mean': mean_se(avg_worlds(delta))[0],                         # [n][selector][eps]
        'delta_avg_worlds_se': mean_se(avg_worlds(delta))[1],
        'regret_avg_worlds_mean': mean_se(avg_worlds(regret))[0],                       # [n][selector]
        'regret_avg_worlds_se': mean_se(avg_worlds(regret))[1],
        'kappa_mean': kappa.mean(axis=(0, 1)).tolist(),                                  # [eps]
        'theorem3_bound': (kappa.mean(axis=(0, 1)) / 2).tolist(),                        # [eps]
        'lemma1_floor_mean': floor.mean(axis=(0, 1)).tolist(),                           # [eps]
        'global_mse_mean': gmse.mean(axis=0).tolist(),                                   # [world][model]
        'ks_reject_rate_0.05': np.mean(ks_p < 0.05, axis=0).tolist(),                    # [r_ols, r_rf, diff]
        'equivariance_maxdiff': equiv.max(axis=0).tolist(),                              # [model]
    }
    raw = {'delta': delta.tolist(), 'regret': regret.tolist(), 'kappa': kappa.tolist(),
           'floor': floor.tolist(), 'ks_p': ks_p.tolist(), 'equivariance': equiv.tolist()}
    return summary, raw, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--n-train', type=int, default=5000)
    ap.add_argument('--n-cal', type=int, nargs='+', default=[250, 500, 1000, 2000, 5000, 10000, 20000, 50000])
    ap.add_argument('--n-test', type=int, default=5000)
    ap.add_argument('--out', default=OUTPUT_FILE)
    ap.add_argument('--quick', action='store_true', help='smoke test: 3 reps, small samples, d3 only')
    args = ap.parse_args()
    if args.quick:
        args.configs, args.reps, args.n_train, args.n_cal, args.n_test = ['d3'], 3, 1000, [250, 1000], 1000
        if args.out == OUTPUT_FILE:
            args.out = os.path.join(BASE_DIR, '..', 'results', 'e1', 'e1_mirror_quick.json')

    out = {
        'experiment': 'E1 mirror pair (Theorem 4; Theorem 3 before 2026-10-15)',
        'settings': dict(configs={c: CONFIGS[c] for c in args.configs}, reps=args.reps, n_train=args.n_train,
                         n_cal=sorted(args.n_cal), n_test=args.n_test, sigma=SIGMA, cp_alpha=CP_ALPHA,
                         eps=EPS, models=MODELS, selectors=SELECTORS, base_seed=BASE_SEED, ks_n=KS_N),
        'environment': dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                            sklearn=sklearn.__version__, machine=platform.platform()),
        'results': {}, 'raw': {},
    }
    for c in args.configs:
        print(f'Config {c}: {CONFIGS[c]}', flush=True)
        summary, raw, secs = run_config(c, CONFIGS[c], args.reps, args.n_train, sorted(args.n_cal), args.n_test)
        out['results'][c] = dict(summary, seconds=secs)
        out['raw'][c] = raw
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:                      # checkpoint after every config
            json.dump(out, f)
        print(f'Config {c} done in {secs:.0f}s -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
