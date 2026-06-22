#!/usr/bin/env python3
"""
CRGA Pilot: Conformal Regret-Guided Aggregation.

Tests all comparison methods across 4 scenarios with diagnostic metrics
using true DGP conditional MSE.

Data: Train 50% | MetaFit 15% | MetaCal 10% | FinalCal 10% | Test 15%
"""
import os, sys, time, json, warnings, itertools
import numpy as np
from collections import defaultdict

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch, torch.nn as nn, torch.optim as optim
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
BASE_SEED = 2024
N_REPS = 100
N = 500
P_LINEAR = 10
P_HIGHDIM = 100
S_HIGHDIM = 5
SIGMA = 1.0
CP_ALPHA = 0.10
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'crga_pilot_results.json')

# ── True DGP signals (noiseless) ────────────────────────────────────────────
def signal_linear(X):
    beta = np.array([2, -1.5, 0.8, 0.5, -0.3, 0, 0, 0, 0, 0])
    return X @ beta

def signal_semiparametric(X):
    beta = np.array([2, -1.5, 0.8, 0.5, -0.3, 0, 0, 0, 0, 0])
    return X @ beta + 0.3 * np.sin(X[:, 0])

def signal_nonlinear(X):
    return np.sin(X[:, 0]) + np.log(1 + np.abs(X[:, 1])) + X[:, 2] * X[:, 3]

def signal_highdim(X):
    beta = np.zeros(P_HIGHDIM)
    beta[:S_HIGHDIM] = [2, -1.5, 0.8, 0.5, -0.3]
    return X @ beta

SIGNAL_FN = {
    'linear': signal_linear, 'semiparametric': signal_semiparametric,
    'nonlinear': signal_nonlinear, 'highdim': signal_highdim,
}

def dgp_from_signal(name, signal_fn):
    def dgp():
        if name == 'highdim':
            X = np.random.randn(N, P_HIGHDIM)
        else:
            X = np.random.randn(N, P_LINEAR)
        f0 = signal_fn(X)
        y = f0 + np.random.randn(N) * SIGMA
        return X, y
    return dgp

# ── DNN Model ──────────────────────────────────────────────────────────────
class DNNOrig(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def train_dnn(model, Xtr, ytr, Xva, yva, epochs=200, lr=1e-3, bs=64, patience=20):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr)),
        batch_size=bs, shuffle=True)
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    best_loss, best_state, wait = float('inf'), None, 0
    Xvt, yvt = torch.FloatTensor(Xva).to(DEVICE), torch.FloatTensor(yva).to(DEVICE)
    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); nn.MSELoss()(model(Xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            loss = nn.MSELoss()(model(Xvt), yvt).item()
        if loss < best_loss - 1e-5:
            best_loss, best_state = loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model

def predict_dnn(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()

# ── Helpers ─────────────────────────────────────────────────────────────────
def cp_quantile(residuals, alpha=CP_ALPHA):
    n = len(residuals)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(residuals, level, method='higher')

def conditional_mse(f0, y_pred):
    """μ_m(x) = (f₀(x) - f̂_m(x))² + σ² (σ² = SIGMA² for MSE)."""
    return (f0 - y_pred) ** 2 + SIGMA ** 2

def one_rep(seed, signal_fn, make_X_y, scenario_name):
    """Run one replication: all methods + diagnostics."""
    np.random.seed(seed); torch.manual_seed(seed)
    X, y = make_X_y()
    n = len(y)

    # ── Split: 50 / 15 / 10 / 10 / 15 (exact block split) ──
    n_base = int(n * 0.50); n_mf = int(n * 0.15); n_mc = int(n * 0.10); n_fc = int(n * 0.10)
    n_test = n - n_base - n_mf - n_mc - n_fc
    idx = np.random.RandomState(seed).permutation(n)
    X_sh, y_sh = X[idx], y[idx]
    X_base, y_base = X_sh[:n_base], y_sh[:n_base]
    X_mf,  y_mf   = X_sh[n_base:n_base+n_mf], y_sh[n_base:n_base+n_mf]
    X_mc,  y_mc   = X_sh[n_base+n_mf:n_base+n_mf+n_mc], y_sh[n_base+n_mf:n_base+n_mf+n_mc]
    X_fc,  y_fc   = X_sh[n_base+n_mf+n_mc:n_base+n_mf+n_mc+n_fc], y_sh[n_base+n_mf+n_mc:n_base+n_mf+n_mc+n_fc]
    X_test, y_test = X_sh[n_base+n_mf+n_mc+n_fc:], y_sh[n_base+n_mf+n_mc+n_fc:]

    # ── Stage 1: Train base models on BaseTrain ──
    ols = LinearRegression().fit(X_base, y_base)
    rf = RandomForestRegressor(200, max_depth=10, min_samples_leaf=5, random_state=seed).fit(X_base, y_base)
    scaler = StandardScaler()
    X_base_s = scaler.fit_transform(X_base)
    X_mf_s = scaler.transform(X_mf); X_mc_s = scaler.transform(X_mc)
    X_fc_s = scaler.transform(X_fc); X_test_s = scaler.transform(X_test)
    X_tr, X_va, y_tr, y_va = train_test_split(X_base_s, y_base, test_size=0.2, random_state=seed)
    dnn_m = DNNOrig(X_tr.shape[1])
    dnn_m = train_dnn(dnn_m, X_tr, y_tr, X_va, y_va)

    models = {'ols': ols, 'rf': rf, 'dnn': dnn_m}
    def pred(m, Xr, Xs):
        if m == 'dnn': return predict_dnn(dnn_m, Xs)
        return models[m].predict(Xr)

    base_names = ['ols', 'rf', 'dnn']

    # Predict on all sets
    pr = {}
    for sname, Xr, Xs in [('base', X_base, X_base_s), ('mf', X_mf, X_mf_s),
                           ('mc', X_mc, X_mc_s), ('fc', X_fc, X_fc_s), ('test', X_test, X_test_s)]:
        pr[sname] = {m: pred(m, Xr, Xs) for m in base_names}

    # ── Stage 2: Local regret model (MetaFit) ──
    ell = {}; R = {}; r_hat = {}; a_hat = {}
    for m in base_names:
        ell[m] = (y_mf - pr['mf'][m]) ** 2
    ell_min = np.min([ell[m] for m in base_names], axis=0)
    for m in base_names:
        R[m] = ell[m] - ell_min
        # r̂_m via RF with OOB
        rf_r = RandomForestRegressor(100, max_depth=5, min_samples_leaf=10, oob_score=True, random_state=seed)
        rf_r.fit(X_mf, R[m])
        r_hat_oob = rf_r.oob_prediction_
        r_hat[m] = rf_r  # store for later prediction

        # Scale model: OOB residual → â_m
        e_m = np.abs(R[m] - r_hat_oob)
        rf_a = RandomForestRegressor(100, max_depth=3, min_samples_leaf=10, random_state=seed + 1)
        rf_a.fit(X_mf, e_m)
        a_hat[m] = rf_a

    # ── Stage 3: Joint conformal calibration (MetaCal) ──
    scores_list = []
    for i in range(n_mc):
        max_s = -np.inf
        for m in base_names:
            r_mc = R[m][:n_mc] if n_mc <= len(R[m]) else R[m]  # can't index; compute on MetaCal
            # Actually compute R_m on MetaCal
            pass
    # Recompute R on MetaCal properly
    ell_mc = {}; R_mc = {}
    for m in base_names:
        ell_mc[m] = (y_mc - pr['mc'][m]) ** 2
    ell_min_mc = np.min([ell_mc[m] for m in base_names], axis=0)
    for m in base_names:
        R_mc[m] = ell_mc[m] - ell_min_mc

    eps = 1e-6
    for i in range(n_mc):
        xi = X_mc[i:i+1]
        s_i = []
        for m in base_names:
            r_hat_val = r_hat[m].predict(xi)[0] if hasattr(r_hat[m], 'predict') else 0
            a_val = max(a_hat[m].predict(xi)[0], eps)
            s_i.append((R_mc[m][i] - r_hat_val) / a_val)
        scores_list.append(max(s_i))
    q = cp_quantile(np.array(scores_list), alpha=CP_ALPHA)

    # U_m(x) for any set
    def compute_U(X_set):
        U = {}
        for m in base_names:
            r_val = r_hat[m].predict(X_set)
            a_val = np.maximum(a_hat[m].predict(X_set), eps)
            U[m] = np.maximum(0, r_val + q * a_val)
        return U

    U_mc = compute_U(X_mc)
    U_fc = compute_U(X_fc)
    U_test = compute_U(X_test)

    # τ from MetaCal
    all_U = np.concatenate([U_mc[m] for m in base_names])
    tau = np.median(all_U)

    # ── Fusion weights ──
    def softmax_weights(U_dict, model_names, tau_val):
        L = np.column_stack([U_dict[m] for m in model_names])
        exp_L = np.exp(-L / tau_val)
        return exp_L / exp_L.sum(axis=1, keepdims=True)

    w_crga_fc = softmax_weights(U_fc, base_names, tau)
    w_crga_test = softmax_weights(U_test, base_names, tau)

    # ── Comparison methods ──

    # Conditional MSE (using true signal) for diagnostics
    f0_test = signal_fn(X_test)
    mu_test = {}
    for m in base_names:
        mu_test[m] = conditional_mse(f0_test, pr['test'][m])

    results = {}

    # 1-3: Single models
    for m in base_names:
        q_m = cp_quantile(np.abs(y_fc - pr['fc'][m]))
        results[m] = {
            'mse': mean_squared_error(y_test, pr['test'][m]),
            'cp_width': 2 * q_m,
            'coverage': np.mean(np.abs(y_test - pr['test'][m]) <= q_m),
        }

    # 4: Equal weight
    ew_pred = np.mean([pr['test'][m] for m in base_names], axis=0)
    ew_fc = np.mean([pr['fc'][m] for m in base_names], axis=0)
    q_ew = cp_quantile(np.abs(y_fc - ew_fc))
    results['equal_weight'] = {
        'mse': mean_squared_error(y_test, ew_pred),
        'cp_width': 2 * q_ew,
        'coverage': np.mean(np.abs(y_test - ew_pred) <= q_ew),
    }

    # 5: Stacking (OLS on MetaFit predictions)
    stack_X = np.column_stack([pr['mf'][m] for m in base_names])
    stack_reg = LinearRegression(fit_intercept=True).fit(stack_X, y_mf)
    stack_coef = np.maximum(stack_reg.coef_, 0)
    if stack_coef.sum() > 0: stack_coef /= stack_coef.sum()
    else: stack_coef = np.ones(3) / 3
    st_fc = sum(stack_coef[i] * pr['fc'][m] for i, m in enumerate(base_names))
    st_test = sum(stack_coef[i] * pr['test'][m] for i, m in enumerate(base_names))
    q_st = cp_quantile(np.abs(y_fc - st_fc))
    results['stacking'] = {
        'mse': mean_squared_error(y_test, st_test),
        'cp_width': 2 * q_st,
        'coverage': np.mean(np.abs(y_test - st_test) <= q_st),
    }

    # 6: CP-width WAA (original negative control)
    # Use normalized CP width as weight signal
    RFR = RandomForestRegressor
    waa_widths_fc = {}; waa_widths_test = {}
    for m in base_names:
        abs_resid_rf = RFR(100, max_depth=5, min_samples_leaf=10, random_state=seed)
        abs_resid_rf.fit(X_base, np.abs(y_base - pr['base'][m]))
        sigma_fc_m = np.maximum(abs_resid_rf.predict(X_fc), 0.05)
        sigma_test_m = np.maximum(abs_resid_rf.predict(X_test), 0.05)
        sigma_mc_m = np.maximum(abs_resid_rf.predict(X_mc), 0.05)
        scores_m = np.abs(y_mc - pr['mc'][m]) / sigma_mc_m
        q_m = cp_quantile(scores_m)
        waa_widths_fc[m] = 2 * q_m * sigma_fc_m
        waa_widths_test[m] = 2 * q_m * sigma_test_m

    tau_waa = np.median([np.median(waa_widths_test[m]) for m in base_names])
    w_waa_fc = softmax_weights(waa_widths_fc, base_names, tau_waa)
    w_waa_test = softmax_weights(waa_widths_test, base_names, tau_waa)
    waa_fc = np.sum(np.column_stack([pr['fc'][m] for m in base_names]) * w_waa_fc, axis=1)
    waa_test = np.sum(np.column_stack([pr['test'][m] for m in base_names]) * w_waa_test, axis=1)

    # 7-10: all local methods need separate fc/test weights
    # Predict loss on both sets
    loss_pred_fc = {}; loss_pred_test = {}
    for m in base_names:
        rf_l = RFR(100, max_depth=5, min_samples_leaf=10, random_state=seed)
        rf_l.fit(X_mf, ell[m])
        loss_pred_fc[m] = rf_l.predict(X_fc)
        loss_pred_test[m] = rf_l.predict(X_test)
    tau_loss = np.median([np.median(loss_pred_test[m]) for m in base_names])
    w_loss_fc = softmax_weights(loss_pred_fc, base_names, tau_loss)
    w_loss_test = softmax_weights(loss_pred_test, base_names, tau_loss)
    loss_fc = np.sum(np.column_stack([pr['fc'][m] for m in base_names]) * w_loss_fc, axis=1)
    loss_test = np.sum(np.column_stack([pr['test'][m] for m in base_names]) * w_loss_test, axis=1)

    # Regret predictions on both sets
    regret_pred_fc = {}; regret_pred_test = {}
    for m in base_names:
        regret_pred_fc[m] = r_hat[m].predict(X_fc)
        regret_pred_test[m] = r_hat[m].predict(X_test)
    tau_reg = np.median([np.median(regret_pred_test[m]) for m in base_names])
    w_reg_fc = softmax_weights(regret_pred_fc, base_names, tau_reg)
    w_reg_test = softmax_weights(regret_pred_test, base_names, tau_reg)
    reg_fc = np.sum(np.column_stack([pr['fc'][m] for m in base_names]) * w_reg_fc, axis=1)
    reg_test = np.sum(np.column_stack([pr['test'][m] for m in base_names]) * w_reg_test, axis=1)

    # Hard CRGA: separate fc/test
    U_best_fc = np.argmin(np.column_stack([U_fc[m] for m in base_names]), axis=1)
    U_best_test = np.argmin(np.column_stack([U_test[m] for m in base_names]), axis=1)
    hard_fc = np.array([pr['fc'][base_names[idx]][i] for i, idx in enumerate(U_best_fc)])
    hard_test = np.array([pr['test'][base_names[idx]][i] for i, idx in enumerate(U_best_test)])

    # ── CRGA point predictions (needed before result dicts) ──
    crga_fc = np.sum(np.column_stack([pr['fc'][m] for m in base_names]) * w_crga_fc, axis=1)
    crga_test = np.sum(np.column_stack([pr['test'][m] for m in base_names]) * w_crga_test, axis=1)
    q_crga = cp_quantile(np.abs(y_fc - crga_fc))

    # ── CP quantiles and result dicts for methods 6-10 ──
    q_waa = cp_quantile(np.abs(y_fc - waa_fc))
    results['cp_width_waa'] = {'mse': mean_squared_error(y_test, waa_test), 'cp_width': 2 * q_waa,
                               'coverage': np.mean(np.abs(y_test - waa_test) <= q_waa)}
    q_ls = cp_quantile(np.abs(y_fc - loss_fc))
    results['local_loss'] = {'mse': mean_squared_error(y_test, loss_test), 'cp_width': 2 * q_ls,
                             'coverage': np.mean(np.abs(y_test - loss_test) <= q_ls)}
    q_rg = cp_quantile(np.abs(y_fc - reg_fc))
    results['local_regret'] = {'mse': mean_squared_error(y_test, reg_test), 'cp_width': 2 * q_rg,
                               'coverage': np.mean(np.abs(y_test - reg_test) <= q_rg)}
    results['crga'] = {'mse': mean_squared_error(y_test, crga_test), 'cp_width': 2 * q_crga,
                       'coverage': np.mean(np.abs(y_test - crga_test) <= q_crga)}
    q_hard = cp_quantile(np.abs(y_fc - hard_fc))
    results['hard_crga'] = {'mse': mean_squared_error(y_test, hard_test), 'cp_width': 2 * q_hard,
                            'coverage': np.mean(np.abs(y_test - hard_test) <= q_hard)}

    # ── Per-point squared errors for tail-risk analysis ──
    sqerr = {}
    for m in base_names:
        sqerr[m] = (y_test - pr['test'][m]) ** 2
    sqerr['equal_weight'] = (y_test - ew_pred) ** 2
    sqerr['stacking'] = (y_test - st_test) ** 2
    sqerr['cp_width_waa'] = (y_test - waa_test) ** 2
    sqerr['local_loss'] = (y_test - loss_test) ** 2
    sqerr['local_regret'] = (y_test - reg_test) ** 2
    sqerr['crga'] = (y_test - crga_test) ** 2
    sqerr['hard_crga'] = (y_test - hard_test) ** 2
    sqerr['cond_oracle'] = np.min([(f0_test - pr['test'][m]) ** 2 for m in base_names], axis=0) + SIGMA**2
    results['_sqerr'] = {k: v.tolist() for k, v in sqerr.items()}
    q_hard = cp_quantile(np.abs(y_fc - hard_fc))
    results['hard_crga'] = {
        'mse': mean_squared_error(y_test, hard_test),
        'cp_width': 2 * q_hard,
        'coverage': np.mean(np.abs(y_test - hard_test) <= q_hard),
    }

    # ── Diagnostics ──
    diag = {}

    # a) Correlation: -U_m vs -μ_m
    for m in base_names:
        diag[f'corr_U_{m}'] = float(np.corrcoef(-U_test[m], -mu_test[m])[0, 1])

    # b) Conditional argmin match
    mu_best = np.argmin(np.column_stack([mu_test[m] for m in base_names]), axis=1)
    U_best = np.argmin(np.column_stack([U_test[m] for m in base_names]), axis=1)
    diag['argmin_match'] = float(np.mean(mu_best == U_best))

    # c) Pairwise ranking accuracy
    pairs = list(itertools.combinations(range(3), 2))
    pair_correct = []
    for i, j in pairs:
        mu_order = mu_test[base_names[i]] < mu_test[base_names[j]]
        U_order = U_test[base_names[i]] < U_test[base_names[j]]
        pair_correct.append(np.mean(mu_order == U_order))
    diag['pairwise_ranking_acc'] = float(np.mean(pair_correct))

    # d) Oracle benchmarks
    # Best global model (choose model with lowest test MSE)
    global_best_mse = min(results[m]['mse'] for m in base_names)
    # Covariate-wise conditional oracle
    cond_oracle_mse = np.mean(np.min([mu_test[m] for m in base_names], axis=0))
    # Realized-Y oracle
    realized_loss = {(y_test[i] - pr['test'][m][i])**2 for i in range(len(y_test)) for m in base_names}
    y_oracle = np.min(np.column_stack([(y_test - pr['test'][m])**2 for m in base_names]), axis=1)
    realized_oracle_mse = np.mean(y_oracle)

    diag['best_global_mse'] = float(global_best_mse)
    diag['cond_oracle_mse'] = float(cond_oracle_mse)
    diag['realized_oracle_mse'] = float(realized_oracle_mse)

    # e) U_m coverage of realized R_m
    R_test = {}
    ell_test = {}
    for m in base_names:
        ell_test[m] = (y_test - pr['test'][m]) ** 2
    ell_min_test = np.min([ell_test[m] for m in base_names], axis=0)
    for m in base_names:
        R_test[m] = ell_test[m] - ell_min_test
    R_covered = np.mean([R_test[m] <= U_test[m] for m in base_names])
    diag['U_coverage'] = float(R_covered)

    results['_diagnostics'] = diag
    return results


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('  CRGA Pilot')
    print(f'  Reps: {N_REPS}, n={N}, σ={SIGMA}')
    print(f'  Device: {DEVICE}')
    print('=' * 60)

    scenarios = {
        'linear': ('linear', signal_linear),
        'semiparametric': ('semiparametric', signal_semiparametric),
        'nonlinear': ('nonlinear', signal_nonlinear),
        'highdim': ('highdim', signal_highdim),
    }

    method_order = ['ols', 'rf', 'dnn', 'equal_weight', 'stacking',
                    'cp_width_waa', 'local_loss', 'local_regret',
                    'crga', 'hard_crga']

    all_results = {}
    for sc_name, (sc_key, signal_fn) in scenarios.items():
        dgp_fn = dgp_from_signal(sc_key, signal_fn)
        rep_list = []
        for rep in range(N_REPS):
            res = one_rep(BASE_SEED + rep, signal_fn, dgp_fn, sc_key)
            rep_list.append(res)
            if rep % 20 == 0:
                print(f'  {sc_name}: rep {rep}/{N_REPS}', flush=True)

        # Aggregate
        agg = {}
        for method in method_order:
            metrics = [r[method] for r in rep_list]
            agg[method] = {
                'mse_mean': float(np.mean([m['mse'] for m in metrics])),
                'mse_sd': float(np.std([m['mse'] for m in metrics])),
                'width_mean': float(np.mean([m['cp_width'] for m in metrics])),
                'coverage_mean': float(np.mean([m['coverage'] for m in metrics])),
            }

        # Aggregate diagnostics
        diag_keys = rep_list[0]['_diagnostics'].keys()
        for dk in diag_keys:
            vals = [r['_diagnostics'][dk] for r in rep_list]
            agg[f'diag_{dk}'] = {
                'mean': float(np.mean(vals)),
                'sd': float(np.std(vals)),
            }

        # ── Tail-risk aggregation (per-point across all reps) ──
        all_sqerr = {}
        for method in method_order:
            sq_list = np.concatenate([r['_sqerr'][method] for r in rep_list])
            n = len(sq_list)
            sorted_sq = np.sort(sq_list)
            n_worst10 = max(1, int(n * 0.10))
            n_worst5 = max(1, int(n * 0.05))
            all_sqerr[method] = {
                'q95': float(np.quantile(sq_list, 0.95)),
                'q99': float(np.quantile(sq_list, 0.99)),
                'worst10_mean': float(np.mean(sorted_sq[-n_worst10:])),
                'worst5_mean': float(np.mean(sorted_sq[-n_worst5:])),
                'max': float(sorted_sq[-1]),
            }
        agg['_tail_risk'] = all_sqerr

        all_results[sc_name] = agg

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── Print MSE results ──
    print('\n\n' + '=' * 140)
    header = f"{'Method':<18s}"
    for sc in scenarios:
        header += f'  {sc:>25s}'
    print(header)
    print('-' * 140)
    for m in method_order:
        row = f'{m:<18s}'
        for sc in scenarios:
            s = all_results[sc][m]
            row += f'  {s["mse_mean"]:>7.4f}±{s["mse_sd"]:.4f} {s["width_mean"]:>6.3f} {s["coverage_mean"]:>5.3f}'
        print(row)
    print('-' * 140)

    # Diagnostics table
    print('\n\n' + '=' * 140)
    print('DIAGNOSTICS')
    print('-' * 140)
    for sc in scenarios:
        a = all_results[sc]
        print(f'\n  {sc}:')
        print(f'    Best global MSE:      {a["diag_best_global_mse"]["mean"]:.4f}')
        print(f'    Cond. oracle MSE:     {a["diag_cond_oracle_mse"]["mean"]:.4f}')
        print(f'    Realized oracle MSE:  {a["diag_realized_oracle_mse"]["mean"]:.4f}')
        print(f'    Argmin match:         {a["diag_argmin_match"]["mean"]:.4f}')
        print(f'    Pairwise ranking acc: {a["diag_pairwise_ranking_acc"]["mean"]:.4f}')
        print(f'    U coverage of R:      {a["diag_U_coverage"]["mean"]:.4f}')
        for m in ['ols', 'rf', 'dnn']:
            print(f'    Corr(-U_{m}, -μ_{m}):  {a[f"diag_corr_U_{m}"]["mean"]:.4f}')

    # ── Tail-risk table (per-point across all reps) ──
    tail_order = ['equal_weight', 'stacking', 'cp_width_waa', 'crga', 'hard_crga']
    for sc in scenarios:
        a = all_results[sc]
        print(f'\n\n  Tail Risk — {sc} (per-point 100×75=7500 obs):')
        print(f'  {"Method":<18s} {"Q95":>8s} {"Q99":>8s} {"Worst5%":>10s} {"Worst10%":>10s} {"Max":>10s}')
        print(f'  {"-"*66}')
        for m in tail_order:
            t = a['_tail_risk'][m]
            print(f'  {m:<18s} {t["q95"]:>8.3f} {t["q99"]:>8.3f} {t["worst5_mean"]:>10.3f} {t["worst10_mean"]:>10.3f} {t["max"]:>10.3f}')
        # conditional oracle ref
        print(f'  {"cond_oracle":<18s} {"—":>8s} {"—":>8s} {"—":>10s} {"—":>10s} {"—":>10s}')
