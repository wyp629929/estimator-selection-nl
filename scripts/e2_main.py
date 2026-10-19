#!/usr/bin/env python3
"""E2: main experiment rerun under one evaluation protocol (6 regimes x 7 estimators).

Fixes relative to ranking_decision_gap.py / run_regime.py:
  * one global oracle: argmin_M G(M), with G(M) = E[(f0(X) - fhat_M(X))^2] + E[sigma^2(X)]
    computed on N_INT fresh covariate draws (f0 is known), instead of a majority vote over
    75 test points or the MSE of differently fitted models;
  * the selectors and the oracles use the same fitted models;
  * explicit seeding (the old script drew data from the unseeded global numpy RNG);
  * misalignment with tolerance eps, regret, near-tie statistics, and the Lemma 1 floor.

Per replication: fit the 7 models once on N_TRAIN points, then evaluate every selector on
nested calibration samples of size n_cal in N_CAL. Global selectors (cp, val) use all n_cal
points. Selectors that need a fitting step (lw, U, rhat) split the n_cal points 60/40 into
fit/calibrate parts; knn uses all n_cal points.

Selectors
  cp    split-CP width (alpha = 0.1)                          global
  val   validation MSE                                        global
  lw    normalized (locally adaptive) CP width                per point
  U     conformalized regret bound U_M (paper Section 3.3)    per point
  rhat  argmin of the estimated conditional mean regret       per point
  knn   k-NN regression of r_M^2 on X, all n_cal points       per point  (CRS, k-NN version)
  crs   forest regression of the regret on X, all n_cal points per point  (CRS, forest version)
For knn and crs the smoothing level (k, or the forest's minimum leaf size) is chosen by held-out
decision loss over a path whose smoothest level is a constant, i.e. the global validation-MSE rule:
5-fold CV for knn, out-of-bag loss for the forest (final rerun, 2026-10-16; replaces k = ceil(sqrt(n)),
the fixed forest and the validation switch of the E2 v2 run).

Variants (--variant): base = E2; corr = Toeplitz-correlated covariates (rho = 0.7); highp = p = 50
(p = 400 for highdim); tuned = per-replication 5-fold CV tuning of ridge/lasso/RF/XGBoost/LightGBM;
noise_normal / noise_t3 / noise_skew = E6, unit-variance noise of three shapes, with split-CP also
recorded at alpha = 0.05 and 0.2 (selectors cp_a0.05, cp_a0.2).

Output: one JSON line per (regime, replication) in <out-dir>/<regime>.jsonl. Reruns skip
replications that are already present, so an interrupted run can be resumed with the same
command.

Usage:
  python3 scripts/e2_main.py --quick                     # smoke test
  python3 scripts/e2_main.py --workers 16                # full E2 run
  python3 scripts/e2_main.py --variant corr --workers 16 # E3 variant
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')            # one thread per worker; parallelism is across replications

import argparse
import json
import platform
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, '..', 'results', 'e2')     

BASE_SEED = 2024
N_TRAIN = 250                 # as in the paper (50% of n = 500)
N_CAL = [250, 1000, 4000]     # 250 = the other half of the paper's n = 500
N_TEST = 5000                 # points for local metrics
N_INT = 50000                 # points for the global risk G(M)
P_LINEAR, P_HIGHDIM = 10, 100
# E3 robustness variants. 'base' is E2 and must reproduce it exactly.
VARIANTS = {
    'base':  dict(p_lin=10, p_high=100, rho=0.0, tuned=False, regimes=None),
    'corr':  dict(p_lin=10, p_high=100, rho=0.7, tuned=False, regimes=['linear', 'nonlinear', 'threshold']),
    'highp': dict(p_lin=50, p_high=400, rho=0.0, tuned=False, regimes=['linear', 'nonlinear', 'highdim']),
    'tuned': dict(p_lin=10, p_high=100, rho=0.0, tuned=True, regimes=['linear', 'nonlinear', 'threshold']),
}
# E6 noise variants (Appendix D of the manuscript): unit-variance noise of three shapes, and split-CP
# also recorded at the extra levels EXTRA_CP_ALPHAS. Other variants use Gaussian noise.
EXTRA_CP_ALPHAS = [0.05, 0.2]
for _noise in ('normal', 't3', 'skew'):
    VARIANTS['noise_' + _noise] = dict(p_lin=10, p_high=100, rho=0.0, tuned=False,
                                       regimes=['linear', 'nonlinear', 'threshold'], noise=_noise,
                                       extra_cp_alphas=EXTRA_CP_ALPHAS)
SIGMA = 1.0
CP_ALPHA = 0.10
FIT_FRAC = 0.6                # fit/calibrate split for lw, U, rhat
CV_FOLDS = 5                  # knn: k chosen by 5-fold CV of the decision loss
EPS_GLOBAL = [0.0, 0.01, 0.05]
EPS_LOCAL = [0.0, 0.1, 0.5]
MODELS = ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn']
GLOBAL_SEL = ['cp', 'val']
LOCAL_SEL = ['lw', 'U', 'rhat', 'knn', 'crs']
REGIMES = ['linear', 'semiparametric', 'nonlinear', 'threshold', 'highdim', 'heteroscedastic']
BETA = np.array([2, -1.5, 0.8, 0.5, -0.3, 0, 0, 0, 0, 0])


# ── Data-generating processes (identical to ranking_decision_gap.py) ─────────
def signal(regime, X):
    if regime in ('linear', 'heteroscedastic'):
        return X[:, :P_LINEAR] @ BETA
    if regime == 'semiparametric':
        return X[:, :P_LINEAR] @ BETA + 0.3 * np.sin(X[:, 0])
    if regime == 'nonlinear':
        return np.sin(X[:, 0]) + np.log(1 + np.abs(X[:, 1])) + X[:, 2] * X[:, 3]
    if regime == 'threshold':
        return 2.0 * (X[:, 0] > 0) + np.log(1 + np.abs(X[:, 1])) * (X[:, 2] > 0) - 1.0
    if regime == 'highdim':
        beta = np.zeros(X.shape[1]); beta[:5] = [2, -1.5, 0.8, 0.5, -0.3]
        return X @ beta
    raise ValueError(regime)


def noise_var(regime, X):
    if regime == 'heteroscedastic':
        return (0.5 + 0.5 * np.abs(X[:, 0])) ** 2
    return np.full(len(X), SIGMA ** 2)


def draw_noise(kind, n, rng):
    """Unit-variance noise: 'normal' (the original call, so Gaussian variants are unchanged),
    't3' (Student t with 3 df divided by sqrt(3)), 'skew' (standardized Gamma(2), skewness sqrt(2))."""
    if kind == 'normal':
        return rng.standard_normal(n)
    if kind == 't3':
        return rng.standard_t(3, n) / np.sqrt(3)
    if kind == 'skew':
        return (rng.gamma(2.0, 1.0, n) - 2.0) / np.sqrt(2.0)
    raise ValueError(kind)


def draw(regime, n, rng, cfg=None):
    cfg = cfg or VARIANTS['base']
    p = cfg['p_high'] if regime == 'highdim' else cfg['p_lin']
    X = rng.standard_normal((n, p))
    if cfg['rho'] > 0:                       # Toeplitz correlation rho^|i-j|
        idx = np.arange(p)
        X = X @ np.linalg.cholesky(cfg['rho'] ** np.abs(idx[:, None] - idx[None, :])).T
    f = signal(regime, X)
    y = f + np.sqrt(noise_var(regime, X)) * draw_noise(cfg.get('noise', 'normal'), n, rng)
    return X, y, f


# ── Estimators (hyperparameters identical to ranking_decision_gap.py) ────────
TUNE_GRIDS = {                 # E3 'tuned' variant: 5-fold CV over small grids, per replication
    'ridge': {'alpha': [0.1, 1.0, 10.0]},
    'lasso': {'alpha': [0.001, 0.01, 0.1]},
    'rf': {'min_samples_leaf': [1, 5, 20]},
    'xgboost': {'max_depth': [2, 4, 6], 'learning_rate': [0.05, 0.1]},
    'lightgbm': {'max_depth': [2, 4, 6], 'learning_rate': [0.05, 0.1]},
}


def fit_models(X, y, seed, device, tuned=False):
    import lightgbm as lgb
    import torch
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    torch.manual_seed(seed)
    models = {
        'ols': LinearRegression().fit(X, y),
        'ridge': Ridge(alpha=1.0).fit(X, y),
        'lasso': Lasso(alpha=0.01, max_iter=5000).fit(X, y),
        'rf': RandomForestRegressor(200, max_depth=10, min_samples_leaf=5, random_state=seed, n_jobs=1).fit(X, y),
        'xgboost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=seed,
                                    verbosity=0, n_jobs=1).fit(X, y),
        'lightgbm': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, verbose=-1,
                                      random_state=seed, n_jobs=1).fit(X, y),
    }
    if tuned:
        for name, grid in TUNE_GRIDS.items():
            models[name] = GridSearchCV(models[name], grid, cv=5, scoring='neg_mean_squared_error',
                                        n_jobs=1).fit(X, y).best_estimator_
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    X_tr, X_va, y_tr, y_va = train_test_split(Xs, y, test_size=0.2, random_state=seed)
    net = train_dnn(X_tr, y_tr, X_va, y_va, device)
    return models, (scaler, net)


def train_dnn(X_tr, y_tr, X_va, y_va, device, epochs=200, lr=1e-3, batch_size=64, patience=20):
    import torch
    import torch.nn as nn
    net = nn.Sequential(nn.Linear(X_tr.shape[1], 128), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
        batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xv, yv = torch.FloatTensor(X_va).to(device), torch.FloatTensor(y_va).to(device)
    best, best_state, wait = float('inf'), None, 0
    for _ in range(epochs):
        net.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            nn.functional.mse_loss(net(Xb).squeeze(-1), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(net(Xv).squeeze(-1), yv).item()
        if v < best - 1e-5:
            best, best_state, wait = v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net


def predict_all(models, dnn, X, device):
    import torch
    scaler, net = dnn
    out = [models[m].predict(X) for m in MODELS[:-1]]
    net.eval()
    with torch.no_grad():
        out.append(net(torch.FloatTensor(scaler.transform(X)).to(device)).squeeze(-1).cpu().numpy())
    return np.column_stack(out).astype(float)             # (n, 7)


# ── Selectors ────────────────────────────────────────────────────────────────
def n_at_cut(scores, alpha=CP_ALPHA):
    """Number of scores equal to the split-CP cutoff. The coverage upper bound 1 - alpha + 1/(n+1)
    holds only for tie-free scores; reports apply it only where this count is 1."""
    return int(np.sum(np.asarray(scores) == cp_quantile(scores, alpha)))


def cp_quantile(scores, alpha=CP_ALPHA):
    """Split-conformal quantile: the k-th smallest score, k = ceil((n + 1)(1 - alpha)) (capped at n).
    Fixed 2026-10-15: the earlier version (inherited from ranking_decision_gap.py) used
    np.quantile(scores, k / n, method='higher'), which returns the (k+1)-th smallest score."""
    n = len(scores)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.partition(np.asarray(scores), k - 1)[k - 1]


def cv_level(Xc, Yc, Pc, levels, score_fn):
    """Choose a smoothing level by CV_FOLDS-fold cross-validation of the decision loss.
    score_fn(X_fit, Y_fit, P_fit, X_new, level) returns estimated risk scores (n_new, m); the rule
    picks the argmin at each point and is scored by the held-out squared loss of the picked model.
    levels[0] is the constant level, i.e. the global validation-MSE rule; ties go to the earlier,
    smoother level."""
    n = len(Yc)
    fold = np.arange(n) % CV_FOLDS
    loss = []
    for lev in levels:
        tot = 0.0
        for k in range(CV_FOLDS):
            tr, te = fold != k, fold == k
            s = score_fn(Xc[tr], Yc[tr], Pc[tr], Xc[te], lev).argmin(axis=1)
            tot += np.sum((Yc[te] - Pc[te][np.arange(te.sum()), s]) ** 2)
        loss.append(tot)
    return levels[int(np.argmin(loss))]


def crs_select(Xc, Yc, Pc, Xt, seed):
    """Conditional risk selection (forest version). For each model, a random forest regresses the
    regret (squared loss minus the smallest loss among the models) on the covariates; at each new
    point the model with the smallest estimated regret is picked. Same argmin as regressing the
    loss itself, with the noise common to all models removed. The minimum leaf size is chosen
    from crs_levels(n) by the out-of-bag decision loss (each calibration point is scored with the
    model picked by the trees that did not see it); the 'const' level, scored by the exact
    leave-one-out mean regret, is the global validation-MSE rule. Ties go to the smoother level.
    Returns the selections on Xt and the chosen level."""
    n, m = Pc.shape
    L = (Yc[:, None] - Pc) ** 2
    Reg = L - L.min(axis=1, keepdims=True)
    held_out_loss = lambda est: float(np.mean(L[np.arange(n), est.argmin(axis=1)]))
    best_loss, best_level, best_forests = held_out_loss((Reg.sum(axis=0) - Reg) / (n - 1)), 'const', None
    for leaf in crs_levels(n)[1:]:
        forests = [RandomForestRegressor(100, min_samples_leaf=leaf, oob_score=True, random_state=seed, n_jobs=1)
                   .fit(Xc, Reg[:, j]) for j in range(m)]
        loss = held_out_loss(np.column_stack([f.oob_prediction_ for f in forests]))
        if loss < best_loss:
            best_loss, best_level, best_forests = loss, leaf, forests
    if best_forests is None:
        return np.full(len(Xt), int(np.argmin(Reg.mean(axis=0)))), best_level
    return np.column_stack([f.predict(Xt) for f in best_forests]).argmin(axis=1), best_level


def knn_scores(X_fit, Y_fit, P_fit, X_new, k, chunk=2000):
    """k-NN regression of r_M^2 on X (Theorem 4(iv)); k = 'const' gives the calibration MSE."""
    R2 = (Y_fit[:, None] - P_fit) ** 2
    if k == 'const':
        return np.tile(R2.mean(axis=0), (len(X_new), 1))
    nn = NearestNeighbors(n_neighbors=k).fit(X_fit)
    return np.vstack([R2[nn.kneighbors(X_new[i:i + chunk], return_distance=False)].mean(axis=1)
                      for i in range(0, len(X_new), chunk)])


def crs_levels(n):
    """Minimum leaf sizes from coarse to fine, as in the P3 pilot (scripts/pilot_p3_crs_cv.py)."""
    return ['const'] + sorted({max(10, int(f * n)) for f in (0.2, 0.05, 0.02)} | {10}, reverse=True)


def knn_levels(n):
    """Neighbourhood sizes from coarse to fine; each must fit in a CV training fold."""
    return ['const'] + [k for k in (1000, 500, 200, 100, 50, 20, 10, 5) if k <= n // 4]


def local_selectors(Xc, Yc, Pc, Xt, Pt, seed):
    """Per-point selections on the test points. Xc/Yc/Pc: calibration covariates, responses,
    predictions (n, m). Xt/Pt: test covariates and predictions. Returns dict name -> (n_test,),
    the U_M bounds on the test points, and the CV-chosen levels of knn and crs."""
    n = len(Yc)
    k_fit = int(FIT_FRAC * n)
    Xf, Xq = Xc[:k_fit], Xc[k_fit:]
    Rf, Rq = Yc[:k_fit, None] - Pc[:k_fit], Yc[k_fit:, None] - Pc[k_fit:]     # residuals (., m)
    m = Pc.shape[1]
    sel = {}

    # lw: normalized CP. sigma_hat_M(x) from |r_M| on the fit part; conformal quantile on the rest.
    lw = np.empty((len(Xt), m))
    for j in range(m):
        rf = RandomForestRegressor(100, max_depth=5, min_samples_leaf=10, random_state=seed, n_jobs=1)
        rf.fit(Xf, np.abs(Rf[:, j]))
        s_q, s_t = np.maximum(rf.predict(Xq), 0.05), np.maximum(rf.predict(Xt), 0.05)
        lw[:, j] = 2.0 * cp_quantile(np.abs(Rq[:, j]) / s_q) * s_t
    sel['lw'] = lw.argmin(axis=1)

    # U and rhat: regret R_M = loss_M - min loss, mean model rhat_M and dispersion a_M on the fit part.
    Lf, Lq = Rf ** 2, Rq ** 2
    Regf, Regq = Lf - Lf.min(axis=1, keepdims=True), Lq - Lq.min(axis=1, keepdims=True)
    r_q, a_q, r_t, a_t = (np.empty((len(Xq), m)), np.empty((len(Xq), m)),
                          np.empty((len(Xt), m)), np.empty((len(Xt), m)))
    for j in range(m):
        rr = RandomForestRegressor(100, max_depth=5, min_samples_leaf=10, oob_score=True,
                                   random_state=seed, n_jobs=1).fit(Xf, Regf[:, j])
        ra = RandomForestRegressor(100, max_depth=3, min_samples_leaf=10, random_state=seed + 1,
                                   n_jobs=1).fit(Xf, np.abs(Regf[:, j] - rr.oob_prediction_))
        r_q[:, j], r_t[:, j] = rr.predict(Xq), rr.predict(Xt)
        a_q[:, j], a_t[:, j] = np.maximum(ra.predict(Xq), 1e-6), np.maximum(ra.predict(Xt), 1e-6)
    q = cp_quantile(np.max((Regq - r_q) / a_q, axis=1))           # joint conformal quantile
    U_t = np.maximum(0.0, r_t + q * a_t)
    sel['U'] = U_t.argmin(axis=1)
    sel['rhat'] = r_t.argmin(axis=1)

    # knn and crs on all n calibration points; knn's k by 5-fold CV, crs's leaf size by out-of-bag loss.
    level = {}
    level['knn'] = cv_level(Xc, Yc, Pc, knn_levels(n), knn_scores)
    sel['knn'] = knn_scores(Xc, Yc, Pc, Xt, level['knn']).argmin(axis=1)
    sel['crs'], level['crs'] = crs_select(Xc, Yc, Pc, Xt, seed)
    return sel, U_t, level


# ── One replication ──────────────────────────────────────────────────────────
def one_rep(regime, rep, device_name, n_cal_list, n_test, n_int, variant='base'):
    import torch
    torch.set_num_threads(1)
    device = torch.device(device_name)
    t0 = time.time()
    seed = BASE_SEED + rep
    rng = np.random.default_rng([BASE_SEED, REGIMES.index(regime), rep])

    cfg = VARIANTS[variant]
    X_tr, y_tr, _ = draw(regime, N_TRAIN, rng, cfg)
    models, dnn = fit_models(X_tr, y_tr, seed, device, tuned=cfg['tuned'])

    # global risk G(M) (includes the noise variance, as in the paper's Table 1)
    Xi, _, fi = draw(regime, n_int, rng, cfg)
    G = np.mean((fi[:, None] - predict_all(models, dnn, Xi, device)) ** 2, axis=0) + noise_var(regime, Xi).mean()
    Gmin = G.min()
    Gs = np.sort(G)

    # local risk D(M|x) on test points (sigma^2(x) is common to all M and cancels in every comparison)
    Xt, yt, ft = draw(regime, n_test, rng, cfg)
    Pt = predict_all(models, dnn, Xt, device)
    D = (ft[:, None] - Pt) ** 2
    Dmin = D.min(axis=1)
    Dgap = np.sort(D, axis=1)[:, 1] - Dmin
    A = {e: D <= Dmin[:, None] + e for e in EPS_LOCAL}

    row = dict(regime=regime, rep=rep, seed=seed, variant=variant, G=G.tolist(),
               G_gap12=float(Gs[1] - Gs[0]),
               local_gap_lt=[float(np.mean(Dgap < e)) for e in EPS_LOCAL[1:]],
               floor=[float(1 - A[e].mean(axis=0).max()) for e in EPS_LOCAL],
               oracle_share=A[0.0].mean(axis=0).tolist(),
               by_ncal={})

    Xc_all, yc_all, _ = draw(regime, max(n_cal_list), rng, cfg)
    Pc_all = predict_all(models, dnn, Xc_all, device)

    def local_metrics(s):
        exc = D[np.arange(n_test), s] - Dmin
        return dict(regret=float(exc.mean()), delta=[float(np.mean(exc > e)) for e in EPS_LOCAL])

    for n in n_cal_list:
        Xc, yc, Pc = Xc_all[:n], yc_all[:n], Pc_all[:n]
        Rc = yc[:, None] - Pc
        out = {}
        # global selectors
        w = np.array([2 * cp_quantile(np.abs(Rc[:, j])) for j in range(len(MODELS))])
        v = np.mean(Rc ** 2, axis=0)
        cover = np.mean(np.abs(yt[:, None] - Pt) <= w / 2, axis=0)            # split-CP coverage per model
        extra = [(f'cp_a{a}', np.array([2 * cp_quantile(np.abs(Rc[:, j]), a) for j in range(len(MODELS))]))
                 for a in cfg.get('extra_cp_alphas', [])]
        for name, score in [('cp', w), ('val', v)] + extra:
            s = int(np.argmin(score))
            pairs = [(i, j) for i in range(len(MODELS)) for j in range(i + 1, len(MODELS))]
            out[name] = dict(choice=MODELS[s], G_regret=float(G[s] - Gmin),
                             G_delta=[float(G[s] > Gmin + e) for e in EPS_GLOBAL],
                             pairwise=float(np.mean([(score[i] < score[j]) == (G[i] < G[j]) for i, j in pairs])),
                             **local_metrics(np.full(n_test, s)))
        # per-point selectors
        sel, U_t, level = local_selectors(Xc, yc, Pc, Xt, Pt, seed)
        for name in LOCAL_SEL:
            out[name] = local_metrics(sel[name])
        for name, lev in level.items():
            out[name]['level'] = str(lev)
        L_t = (yt[:, None] - Pt) ** 2
        U_cover = float(np.mean(np.all(L_t - L_t.min(axis=1, keepdims=True) <= U_t, axis=1)))
        row['by_ncal'][str(n)] = dict(selectors=out, cp_coverage=cover.tolist(), U_joint_coverage=U_cover)
    row['seconds'] = time.time() - t0
    return row


def done_reps(path, key='rep'):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)[key] for line in f if line.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', default='base', choices=list(VARIANTS),
                    help='base = E2; corr / highp / tuned = E3 robustness variants')
    ap.add_argument('--regimes', nargs='+', default=None, choices=REGIMES,
                    help='default: all six for base, the variant subset otherwise')
    ap.add_argument('--reps', type=int, default=None, help='default: 200 for base, 100 for E3 variants')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    ap.add_argument('--out-dir', default=None, help='default: results/e2 (base) or results/e3/<variant>')
    ap.add_argument('--quick', action='store_true', help='smoke test: 2 reps, small test sets')
    args = ap.parse_args()
    cfg = VARIANTS[args.variant]
    args.regimes = args.regimes or cfg['regimes'] or REGIMES
    args.reps = args.reps or (200 if args.variant == 'base' else 100)
    if args.out_dir is None:
        sub = 'e6' if args.variant.startswith('noise_') else 'e3'
        args.out_dir = OUT_DIR if args.variant == 'base' else os.path.join(BASE_DIR, '..', 'results', sub, args.variant)
        if args.quick:
            args.out_dir += '_quick'
    if args.quick:
        args.reps, n_test, n_int, n_cal = 2, 1000, 5000, [250, 1000]
    else:
        n_test, n_int, n_cal = N_TEST, N_INT, N_CAL
    os.makedirs(args.out_dir, exist_ok=True)

    import lightgbm, sklearn, torch, xgboost
    meta = dict(settings=dict(n_train=N_TRAIN, n_cal=n_cal, n_test=n_test, n_int=n_int, sigma=SIGMA, cp_alpha=CP_ALPHA,
                              fit_frac=FIT_FRAC, cv_folds=CV_FOLDS, eps_global=EPS_GLOBAL, eps_local=EPS_LOCAL, models=MODELS,
                              global_selectors=GLOBAL_SEL, local_selectors=LOCAL_SEL, reps=args.reps,
                              extra_cp_selectors=[f'cp_a{a}' for a in cfg.get('extra_cp_alphas', [])],
                              base_seed=BASE_SEED, device=args.device, quick=args.quick,
                              variant=args.variant, variant_cfg=cfg, regimes=args.regimes,
                              tune_grids=TUNE_GRIDS if cfg['tuned'] else None),
                environment=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
                                 xgboost=xgboost.__version__, lightgbm=lightgbm.__version__, torch=torch.__version__,
                                 machine=platform.platform(), cpu_count=os.cpu_count()))
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    done = {r: done_reps(os.path.join(args.out_dir, f'{r}.jsonl')) for r in args.regimes}
    tasks = [(r, k) for k in range(args.reps) for r in args.regimes if k not in done[r]]   # interleave regimes
    print(f'[{args.variant}] {len(tasks)} replications to run ({args.workers} workers, device={args.device}) '
          f'-> {args.out_dir}', flush=True)
    t0, finished = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_rep, r, k, args.device, n_cal, n_test, n_int, args.variant): (r, k) for r, k in tasks}
        for fut in as_completed(futs):
            row = fut.result()                       # an exception here stops the run; completed rows are kept
            with open(os.path.join(args.out_dir, f'{row["regime"]}.jsonl'), 'a') as f:
                f.write(json.dumps(row) + '\n')
            finished += 1
            if finished % max(1, len(tasks) // 50) == 0 or finished == len(tasks):
                el = time.time() - t0
                print(f'  {finished}/{len(tasks)} done, {el / 60:.1f} min elapsed, '
                      f'~{el / finished * (len(tasks) - finished) / 60:.1f} min left', flush=True)
    print('All replications finished.', flush=True)


if __name__ == '__main__':
    main()
