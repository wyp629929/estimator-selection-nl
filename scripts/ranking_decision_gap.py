#!/usr/bin/env python3
"""
Ranking–Decision Gap: Comprehensive Analysis of Conformal Signals for Model Selection.

Extends Section 5 of "Estimator Selection Across Nonlinearity Regimes" to
  7 models × 5 scenarios × B replications,
systematically evaluating whether conformal prediction signals can guide
estimator selection and aggregation.

Three CP signals:
  1. Raw CP interval width  (split conformal, constant per model per rep)
  2. Locally adaptive CP width  (normalized conformal, per-point)
  3. Conformalized regret bound U_m  (per-point)

Diagnostics:
  - Argmin match rate: does the best-signal model match the conditionally optimal one?
  - Pairwise ranking accuracy: does the signal correctly rank model pairs?
  - Top-k coverage: does the top-k by signal contain the truly best model?
  - Aggregation: CRGA vs stacking vs equal-weight fusion

Usage:
  python ranking_decision_gap.py [--reps 200] [--scenarios linear semiparametric ...]

Output:
  ../results/ranking_decision_gap.json
"""

import os, sys, time, json, warnings, itertools, argparse
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
BASE_SEED = 2024
N_REPS_DEFAULT = 200
N = 500
P_LINEAR = 10
P_HIGHDIM = 100
S_HIGHDIM = 5
SIGMA = 1.0
CP_ALPHA = 0.10        # 90% nominal coverage
EPS = 1e-6

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'results', 'ranking_decision_gap.json')

# ── Models list ────────────────────────────────────────────────────────────
ALL_MODELS = ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn']
M = len(ALL_MODELS)  # 7

# ── PyTorch DNN ────────────────────────────────────────────────────────────
class DNNOriginal(nn.Module):
    """3 hidden layers (128-64-32), ReLU, dropout 0.2."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_dnn(model, X_train, y_train, X_val, y_val,
              epochs=200, lr=0.001, batch_size=64, patience=20):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=True)
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_state = None
    wait = 0
    Xvt = torch.FloatTensor(X_val).to(DEVICE)
    yvt = torch.FloatTensor(y_val).to(DEVICE)
    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = nn.MSELoss()(model(Xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.MSELoss()(model(Xvt), yvt).item()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_dnn(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy().flatten()

# ── Data-Generating Processes ──────────────────────────────────────────────
# Signal functions (noiseless, for conditional MSE computation)
def signal_linear(X):
    beta = np.array([2, -1.5, 0.8, 0.5, -0.3, 0, 0, 0, 0, 0])
    return X[:, :P_LINEAR] @ beta

def signal_semiparametric(X):
    beta = np.array([2, -1.5, 0.8, 0.5, -0.3, 0, 0, 0, 0, 0])
    return X[:, :P_LINEAR] @ beta + 0.3 * np.sin(X[:, 0])

def signal_nonlinear(X):
    return (np.sin(X[:, 0]) + np.log(1 + np.abs(X[:, 1]))
            + X[:, 2] * X[:, 3])

def signal_threshold(X):
    return (2.0 * (X[:, 0] > 0)
            + np.log(1 + np.abs(X[:, 1])) * (X[:, 2] > 0) - 1.0)

def signal_highdim(X):
    beta = np.zeros(P_HIGHDIM)
    beta[:S_HIGHDIM] = [2, -1.5, 0.8, 0.5, -0.3]
    return X @ beta

def signal_heteroscedastic(X):
    """Same linear mean function as dgp_linear, used with heteroscedastic noise."""
    beta = np.array([2, -1.5, 0.8, 0.5, -0.3, 0, 0, 0, 0, 0])
    return X[:, :P_LINEAR] @ beta

SIGNAL_FN = {
    'linear': signal_linear,
    'semiparametric': signal_semiparametric,
    'nonlinear': signal_nonlinear,
    'threshold': signal_threshold,
    'highdim': signal_highdim,
    'heteroscedastic': signal_heteroscedastic,
}

def make_dgp(name, signal_fn):
    """Return a DGP function that generates (X, y, f0) for the given scenario."""
    def dgp():
        if name == 'highdim':
            X = np.random.randn(N, P_HIGHDIM)
        else:
            X = np.random.randn(N, P_LINEAR)
        f0 = signal_fn(X)
        if name == 'heteroscedastic':
            sigma_pt = 0.5 + 0.5 * np.abs(X[:, 0])  # variance 0.25-1.0, sigma 0.5-1.12
            y = f0 + np.random.randn(N) * sigma_pt
        else:
            y = f0 + np.random.randn(N) * SIGMA
        return X, y, f0
    return dgp

# ── Conformal Helpers ──────────────────────────────────────────────────────
def cp_quantile(residuals, alpha=CP_ALPHA):
    """Finite-sample corrected conformal quantile."""
    n = len(residuals)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(residuals, level, method='higher')

# ── One Replication ────────────────────────────────────────────────────────
def one_rep(seed, signal_fn, dgp_fn, scenario_name):
    """
    Run one replication.

    Data split:
        Train 50% | MetaFit 15% | MetaCal 10% | FinalCal 10% | Test 15%
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    X, y, f0 = dgp_fn()
    n = len(y)

    # ── Block split ──
    n_base = int(n * 0.50)
    n_mf   = int(n * 0.15)
    n_mc   = int(n * 0.10)
    n_fc   = int(n * 0.10)
    n_test = n - n_base - n_mf - n_mc - n_fc

    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    X_base, y_base = X[:n_base], y[:n_base]
    X_mf,   y_mf   = X[n_base:n_base+n_mf], y[n_base:n_base+n_mf]
    X_mc,   y_mc   = X[n_base+n_mf:n_base+n_mf+n_mc], y[n_base+n_mf:n_base+n_mf+n_mc]
    X_fc,   y_fc   = X[n_base+n_mf+n_mc:n_base+n_mf+n_mc+n_fc], y[n_base+n_mf+n_mc:n_base+n_mf+n_mc+n_fc]
    X_test, y_test = X[n_base+n_mf+n_mc+n_fc:], y[n_base+n_mf+n_mc+n_fc:]
    f0_test = f0[idx][n_base+n_mf+n_mc+n_fc:]

    # ── StandardScaler (for DNN) ──
    scaler = StandardScaler()
    X_base_s = scaler.fit_transform(X_base)
    X_mf_s   = scaler.transform(X_mf)
    X_mc_s   = scaler.transform(X_mc)
    X_fc_s   = scaler.transform(X_fc)
    X_test_s = scaler.transform(X_test)

    # ── Train 7 models ──
    models = {}

    # OLS
    models['ols'] = LinearRegression().fit(X_base, y_base)

    # Ridge
    models['ridge'] = Ridge(alpha=1.0).fit(X_base, y_base)

    # Lasso
    models['lasso'] = Lasso(alpha=0.01, max_iter=5000).fit(X_base, y_base)

    # Random Forest
    models['rf'] = RandomForestRegressor(
        200, max_depth=10, min_samples_leaf=5, random_state=seed).fit(X_base, y_base)

    # XGBoost
    models['xgboost'] = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=seed, verbosity=0).fit(X_base, y_base)

    # LightGBM
    models['lightgbm'] = lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        verbose=-1, random_state=seed).fit(X_base, y_base)

    # DNN
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_base_s, y_base, test_size=0.2, random_state=seed)
    dnn_model = DNNOriginal(X_tr.shape[1])
    dnn_model = train_dnn(dnn_model, X_tr, y_tr, X_va, y_va)
    models['dnn'] = dnn_model

    # ── Helper: predict with any model ──
    def predict(model_obj, X_raw, X_scaled):
        if isinstance(model_obj, DNNOriginal):
            return predict_dnn(model_obj, X_scaled)
        return model_obj.predict(X_raw)

    # ── Predictions on all sets ──
    preds = {}
    for sname, Xr, Xs in [
        ('base', X_base, X_base_s), ('mf', X_mf, X_mf_s),
        ('mc', X_mc, X_mc_s),       ('fc', X_fc, X_fc_s),
        ('test', X_test, X_test_s)]:
        preds[sname] = {}
        for m in ALL_MODELS:
            preds[sname][m] = predict(models[m], Xr, Xs)

    # ── Conditional MSE (oracle benchmark, uses true f0) ──
    # Compute per-point conditional variance (heteroscedastic or constant)
    if scenario_name == 'heteroscedastic':
        sigma2_pt = 0.5 + 0.5 * np.abs(X[:, 0])
        sigma2_pt = sigma2_pt ** 2
        sigma2_test = sigma2_pt[idx][n_base+n_mf+n_mc+n_fc:]
    else:
        sigma2_test = SIGMA ** 2

    mu_test = {}
    for m in ALL_MODELS:
        mu_test[m] = (f0_test - preds['test'][m]) ** 2 + sigma2_test

    sq_err = {}
    for m in ALL_MODELS:
        sq_err[m] = (y_test - preds['test'][m]) ** 2

    # ================================================================
    # Signal 1: Raw CP Width (standard split conformal on FinalCal)
    # ================================================================
    cp_widths = {}       # {model: constant_width}
    for m in ALL_MODELS:
        q = cp_quantile(np.abs(y_fc - preds['fc'][m]))
        cp_widths[m] = 2.0 * q

    # ================================================================
    # Signal 2: Locally Adaptive CP Width (normalized conformal)
    #   - Fit |residual| ~ X with RF on Base → σ̂(x)
    #   - Normalized scores on MetaCal → q
    #   - Width(x) = 2 × q × σ̂(x)
    # ================================================================
    local_widths = {}    # {model: array of per-point widths on Test}
    for m in ALL_MODELS:
        abs_resid = np.abs(y_base - preds['base'][m])
        resid_rf = RandomForestRegressor(
            100, max_depth=5, min_samples_leaf=10, random_state=seed)
        resid_rf.fit(X_base, abs_resid)

        sigma_mc   = np.maximum(resid_rf.predict(X_mc), 0.05)
        sigma_test = np.maximum(resid_rf.predict(X_test), 0.05)

        scores = np.abs(y_mc - preds['mc'][m]) / sigma_mc
        q = cp_quantile(scores)
        local_widths[m] = 2.0 * q * sigma_test

    # ================================================================
    # Signal 3: Conformalized Regret Bound U_m
    #   Pipeline: MetaFit → fit r̂_m, â_m | MetaCal → joint q | FinalCal → CRGA
    # ================================================================
    # Loss and regret on MetaFit
    ell_mf = {}
    for m in ALL_MODELS:
        ell_mf[m] = (y_mf - preds['mf'][m]) ** 2
    ell_min_mf = np.min([ell_mf[m] for m in ALL_MODELS], axis=0)

    R_mf = {}
    for m in ALL_MODELS:
        R_mf[m] = ell_mf[m] - ell_min_mf

    # Fit r̂_m(x) and â_m(x) with RF
    r_hat, a_hat = {}, {}
    for m in ALL_MODELS:
        rf_r = RandomForestRegressor(
            100, max_depth=5, min_samples_leaf=10, oob_score=True, random_state=seed)
        rf_r.fit(X_mf, R_mf[m])
        r_hat[m] = rf_r

        r_oob = rf_r.oob_prediction_
        e_m = np.abs(R_mf[m] - r_oob)
        rf_a = RandomForestRegressor(
            100, max_depth=3, min_samples_leaf=10, random_state=seed + 1)
        rf_a.fit(X_mf, e_m)
        a_hat[m] = rf_a

    # Joint conformal quantile on MetaCal
    scores_list = []
    for i in range(n_mc):
        xi = X_mc[i:i+1]
        ell_i = {}
        for m in ALL_MODELS:
            ell_i[m] = (y_mc[i] - preds['mc'][m][i]) ** 2
        ell_min_i = min(ell_i.values())
        s_i = []
        for m in ALL_MODELS:
            R_val = ell_i[m] - ell_min_i
            r_val = r_hat[m].predict(xi)[0]
            a_val = max(a_hat[m].predict(xi)[0], EPS)
            s_i.append((R_val - r_val) / a_val)
        scores_list.append(max(s_i))

    q_regret = cp_quantile(np.array(scores_list))

    def compute_U(X_set):
        U = {}
        for m in ALL_MODELS:
            r_val = r_hat[m].predict(X_set)
            a_val = np.maximum(a_hat[m].predict(X_set), EPS)
            U[m] = np.maximum(0, r_val + q_regret * a_val)
        return U

    U_test = compute_U(X_test)

    # ================================================================
    # Diagnostics
    # ================================================================
    diag = {}

    # ── A. Correlation of signal with negative conditional MSE ──
    # (Only for per-point signals: locally adaptive width and U_m)
    for sig_name, sig_dict in [('local_width', local_widths), ('U', U_test)]:
        for m in ALL_MODELS:
            corr = np.corrcoef(-sig_dict[m], -mu_test[m])[0, 1]
            diag[f'corr_{sig_name}_{m}'] = float(corr if not np.isnan(corr) else 0.0)

    # ── B. Argmin match rate ──
    # For CP width (constant): the argmin is a single model per rep.
    #   Compare against the most-frequently-conditionally-optimal model.
    mu_best_per_point = np.argmin(
        np.column_stack([mu_test[m] for m in ALL_MODELS]), axis=1)
    # Majority-vote "best model" under conditional MSE
    global_mu_best_idx = np.bincount(mu_best_per_point).argmax()
    global_mu_best = ALL_MODELS[global_mu_best_idx]

    # CP width argmin (global, constant per rep)
    cp_best = min(ALL_MODELS, key=lambda m: cp_widths[m])
    diag['argmin_match_cp_width'] = float(1.0 if cp_best == global_mu_best else 0.0)

    # Locally adaptive width argmin (per-point)
    lw_best = np.argmin(
        np.column_stack([local_widths[m] for m in ALL_MODELS]), axis=1)
    diag['argmin_match_local_width'] = float(np.mean(lw_best == mu_best_per_point))

    # Regret bound argmin (per-point)
    U_best = np.argmin(
        np.column_stack([U_test[m] for m in ALL_MODELS]), axis=1)
    diag['argmin_match_U'] = float(np.mean(U_best == mu_best_per_point))

    # ── C. Pairwise ranking accuracy ──
    pairs = list(itertools.combinations(range(M), 2))

    for sig_name, sig_dict in [('cp_width', cp_widths), ('local_width', local_widths), ('U', U_test)]:
        pair_correct = []
        for i, j in pairs:
            mi, mj = ALL_MODELS[i], ALL_MODELS[j]
            if sig_name == 'cp_width':
                # Global comparison: one value per model per rep
                mu_i_avg = np.mean(mu_test[mi])
                mu_j_avg = np.mean(mu_test[mj])
                correct = 1.0 if (cp_widths[mi] < cp_widths[mj]) == (mu_i_avg < mu_j_avg) else 0.0
                pair_correct.append(correct)
            else:
                mu_order = mu_test[mi] < mu_test[mj]
                sig_order = sig_dict[mi] < sig_dict[mj]
                pair_correct.append(np.mean(mu_order == sig_order))
        diag[f'pairwise_ranking_{sig_name}'] = float(np.mean(pair_correct))

    # ── D. Top-k coverage ──
    #   Does the set of k models with smallest signal contain the conditionally best model?
    for sig_name, sig_dict in [('cp_width', cp_widths), ('local_width', local_widths), ('U', U_test)]:
        for k in [1, 2, 3]:
            if sig_name == 'cp_width':
                top_k = set(sorted(ALL_MODELS, key=lambda m: cp_widths[m])[:k])
                # Check if per-point optimal model is in top_k for most points
                mu_best_models = [ALL_MODELS[idx] for idx in mu_best_per_point]
                covered = [m in top_k for m in mu_best_models]
                diag[f'top{k}_coverage_cp_width'] = float(np.mean(covered))
            else:
                top_k_sets = []
                for pt in range(n_test):
                    ordered = sorted(ALL_MODELS, key=lambda m: sig_dict[m][pt])
                    top_k_sets.append(set(ordered[:k]))
                covered = [ALL_MODELS[mu_best_per_point[pt]] in top_k_sets[pt]
                           for pt in range(n_test)]
                diag[f'top{k}_coverage_{sig_name}'] = float(np.mean(covered))

    # ── E. U_m regret coverage ──
    ell_test = {}
    for m in ALL_MODELS:
        ell_test[m] = (y_test - preds['test'][m]) ** 2
    ell_min_test = np.min([ell_test[m] for m in ALL_MODELS], axis=0)
    R_test = {}
    for m in ALL_MODELS:
        R_test[m] = ell_test[m] - ell_min_test
    R_covered = np.mean([R_test[m] <= U_test[m] for m in ALL_MODELS])
    diag['U_coverage'] = float(R_covered)

    # ================================================================
    # Aggregation: CRGA vs Stacking vs Equal-Weight
    # ================================================================
    agg = {}

    # 1-7: Individual models
    for m in ALL_MODELS:
        q = cp_quantile(np.abs(y_fc - preds['fc'][m]))
        agg[m] = {
            'mse': float(mean_squared_error(y_test, preds['test'][m])),
            'cp_width': float(2.0 * q),
            'coverage': float(np.mean(np.abs(y_test - preds['test'][m]) <= q)),
        }

    # 8: Equal-weight fusion
    ew_pred = np.mean([preds['test'][m] for m in ALL_MODELS], axis=0)
    ew_fc   = np.mean([preds['fc'][m]   for m in ALL_MODELS], axis=0)
    q_ew = cp_quantile(np.abs(y_fc - ew_fc))
    agg['equal_weight'] = {
        'mse': float(mean_squared_error(y_test, ew_pred)),
        'cp_width': float(2.0 * q_ew),
        'coverage': float(np.mean(np.abs(y_test - ew_pred) <= q_ew)),
    }

    # 9: Stacking (non-negative Ridge on MetaFit)
    stack_X_mf = np.column_stack([preds['mf'][m] for m in ALL_MODELS])
    stack_reg = Ridge(alpha=1.0, positive=True, fit_intercept=True).fit(stack_X_mf, y_mf)
    stack_coef = stack_reg.coef_
    if stack_coef.sum() > 0:
        stack_coef = stack_coef / stack_coef.sum()
    else:
        stack_coef = np.ones(M) / M

    st_fc   = sum(stack_coef[i] * preds['fc'][ALL_MODELS[i]]   for i in range(M))
    st_test = sum(stack_coef[i] * preds['test'][ALL_MODELS[i]] for i in range(M))
    q_st = cp_quantile(np.abs(y_fc - st_fc))
    agg['stacking'] = {
        'mse': float(mean_squared_error(y_test, st_test)),
        'cp_width': float(2.0 * q_st),
        'coverage': float(np.mean(np.abs(y_test - st_test) <= q_st)),
    }

    # 10: CRGA (Conformalized Regret-Guided Aggregation)
    all_U_fc = np.concatenate([U_test[m] for m in ALL_MODELS])
    tau = np.median(all_U_fc)

    def softmax_weights(U_dict, names, tau_val):
        L = np.column_stack([U_dict[m] for m in names])
        exp_L = np.exp(-L / tau_val)
        return exp_L / exp_L.sum(axis=1, keepdims=True)

    # For CRGA, re-calibrate on FinalCal
    U_fc = compute_U(X_fc)
    w_fc   = softmax_weights(U_fc,   ALL_MODELS, tau)
    w_test = softmax_weights(U_test, ALL_MODELS, tau)

    crga_fc   = np.sum(w_fc   * np.column_stack([preds['fc'][m]   for m in ALL_MODELS]), axis=1)
    crga_test = np.sum(w_test * np.column_stack([preds['test'][m] for m in ALL_MODELS]), axis=1)
    q_crga = cp_quantile(np.abs(y_fc - crga_fc))
    agg['crga'] = {
        'mse': float(mean_squared_error(y_test, crga_test)),
        'cp_width': float(2.0 * q_crga),
        'coverage': float(np.mean(np.abs(y_test - crga_test) <= q_crga)),
    }

    # 11: Hard CRGA (pick single model with smallest U_m per point)
    hard_best = np.argmin(np.column_stack([U_test[m] for m in ALL_MODELS]), axis=1)
    hard_pred = np.array([preds['test'][ALL_MODELS[idx]][i] for i, idx in enumerate(hard_best)])
    hard_fc_pred = np.array([preds['fc'][ALL_MODELS[idx]][i]
                              for i, idx in enumerate(
                                  np.argmin(np.column_stack([U_fc[m] for m in ALL_MODELS]), axis=1))])
    q_hard = cp_quantile(np.abs(y_fc - hard_fc_pred))
    agg['hard_crga'] = {
        'mse': float(mean_squared_error(y_test, hard_pred)),
        'cp_width': float(2.0 * q_hard),
        'coverage': float(np.mean(np.abs(y_test - hard_pred) <= q_hard)),
    }

    # ── Per-point squared errors for tail-risk ──
    sqerr_all = {}
    for m in ALL_MODELS:
        sqerr_all[m] = (y_test - preds['test'][m]) ** 2
    sqerr_all['equal_weight'] = (y_test - ew_pred) ** 2
    sqerr_all['stacking']     = (y_test - st_test) ** 2
    sqerr_all['crga']         = (y_test - crga_test) ** 2
    sqerr_all['hard_crga']    = (y_test - hard_pred) ** 2

    # Oracle: conditional oracle per point
    oracle_sq = np.min(
        np.column_stack([(f0_test - preds['test'][m]) ** 2 for m in ALL_MODELS]), axis=1)
    oracle_sq += sigma2_test
    sqerr_all['cond_oracle'] = oracle_sq

    return {
        'diagnostics': diag,
        'aggregation': agg,
        '_sqerr': {k: v.tolist() for k, v in sqerr_all.items()},
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Ranking–Decision Gap: Conformal Signals for Model Selection')
    parser.add_argument('--reps', type=int, default=N_REPS_DEFAULT,
                        help=f'Number of Monte Carlo replications (default: {N_REPS_DEFAULT})')
    parser.add_argument('--scenarios', nargs='+',
                        default=['linear', 'semiparametric', 'nonlinear', 'threshold', 'highdim'],
                        choices=['linear', 'semiparametric', 'nonlinear', 'threshold', 'highdim', 'heteroscedastic'],
                        help='Scenarios to run')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE,
                        help='Output JSON path')
    args = parser.parse_args()

    n_reps = args.reps
    scenarios_to_run = args.scenarios
    output_path = args.output

    print('=' * 64)
    print('  Ranking–Decision Gap Analysis')
    print(f'  Models:  {M} ({", ".join(ALL_MODELS)})')
    print(f'  Scenarios: {", ".join(scenarios_to_run)}')
    print(f'  Replications: {n_reps}, n={N}, sigma={SIGMA}')
    print(f'  Device: {DEVICE}')
    print(f'  Output: {output_path}')
    print('=' * 64)

    all_results = {}
    for sc_name in scenarios_to_run:
        signal_fn = SIGNAL_FN[sc_name]
        dgp_fn = make_dgp(sc_name, signal_fn)

        print(f'\n── {sc_name} ──')
        t_start = time.time()
        rep_list = []
        for rep in range(n_reps):
            res = one_rep(BASE_SEED + rep, signal_fn, dgp_fn, sc_name)
            rep_list.append(res)
            if rep % 25 == 0:
                print(f'  rep {rep}/{n_reps}  [{time.time()-t_start:.0f}s]', flush=True)

        # ── Aggregate aggregation results ──
        agg_methods = ALL_MODELS + ['equal_weight', 'stacking', 'crga', 'hard_crga']
        agg_summary = {}
        for method in agg_methods:
            metrics = [r['aggregation'][method] for r in rep_list]
            agg_summary[method] = {
                'mse_mean': float(np.mean([m['mse'] for m in metrics])),
                'mse_sd':   float(np.std([m['mse'] for m in metrics])),
                'width_mean': float(np.mean([m['cp_width'] for m in metrics])),
                'width_sd':   float(np.std([m['cp_width'] for m in metrics])),
                'coverage_mean': float(np.mean([m['coverage'] for m in metrics])),
                'coverage_sd':   float(np.std([m['coverage'] for m in metrics])),
            }

        # ── Aggregate diagnostics ──
        diag_keys = [k for k in rep_list[0]['diagnostics'].keys()]
        for dk in diag_keys:
            vals = [r['diagnostics'][dk] for r in rep_list]
            agg_summary[f'diag_{dk}'] = {
                'mean': float(np.mean(vals)),
                'sd':   float(np.std(vals)),
            }

        # ── Tail-risk (per-point across all reps) ──
        tail_methods = ALL_MODELS + ['equal_weight', 'stacking', 'crga', 'hard_crga']
        all_sqerr = {}
        for method in tail_methods + ['cond_oracle']:
            sq_list = np.concatenate([r['_sqerr'][method] for r in rep_list])
            sorted_sq = np.sort(sq_list)
            n_total = len(sq_list)
            n_w5  = max(1, int(n_total * 0.05))
            n_w10 = max(1, int(n_total * 0.10))
            all_sqerr[method] = {
                'q95': float(np.quantile(sq_list, 0.95)),
                'q99': float(np.quantile(sq_list, 0.99)),
                'worst5_mean':  float(np.mean(sorted_sq[-n_w5:])),
                'worst10_mean': float(np.mean(sorted_sq[-n_w10:])),
                'max': float(sorted_sq[-1]),
            }
        agg_summary['_tail_risk'] = all_sqerr

        all_results[sc_name] = agg_summary

        # ── Print summary ──
        print(f'\n  ── Diagnostics Summary ({sc_name}) ──')
        print(f'  Argmin match — CP width:       {agg_summary["diag_argmin_match_cp_width"]["mean"]:.4f} '
              f'(baseline 1/{M}={1/M:.3f})')
        print(f'  Argmin match — Local width:    {agg_summary["diag_argmin_match_local_width"]["mean"]:.4f}')
        print(f'  Argmin match — Regret U:       {agg_summary["diag_argmin_match_U"]["mean"]:.4f}')
        print(f'  Pairwise ranking — CP width:   {agg_summary["diag_pairwise_ranking_cp_width"]["mean"]:.4f} '
              f'(baseline 0.500)')
        print(f'  Pairwise ranking — Local width: {agg_summary["diag_pairwise_ranking_local_width"]["mean"]:.4f}')
        print(f'  Pairwise ranking — U:           {agg_summary["diag_pairwise_ranking_U"]["mean"]:.4f}')
        print(f'  Top-1 coverage — CP width:     {agg_summary["diag_top1_coverage_cp_width"]["mean"]:.4f}')
        print(f'  Top-1 coverage — U:             {agg_summary["diag_top1_coverage_U"]["mean"]:.4f}')
        print(f'  Top-2 coverage — U:             {agg_summary["diag_top2_coverage_U"]["mean"]:.4f}')
        print(f'  U coverage of regret:           {agg_summary["diag_U_coverage"]["mean"]:.4f}')

        print(f'\n  ── Aggregation MSE ({sc_name}) ──')
        best_m = min(ALL_MODELS, key=lambda m: agg_summary[m]['mse_mean'])
        print(f'  Best single model: {best_m} = {agg_summary[best_m]["mse_mean"]:.4f}')
        print(f'  Equal-weight:      {agg_summary["equal_weight"]["mse_mean"]:.4f}')
        print(f'  Stacking:          {agg_summary["stacking"]["mse_mean"]:.4f}')
        print(f'  CRGA:              {agg_summary["crga"]["mse_mean"]:.4f}')
        print(f'  Hard CRGA:         {agg_summary["hard_crga"]["mse_mean"]:.4f}')
        print(f'  Elapsed: {time.time()-t_start:.0f}s')

    # ── Save ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {output_path}')


if __name__ == '__main__':
    main()
