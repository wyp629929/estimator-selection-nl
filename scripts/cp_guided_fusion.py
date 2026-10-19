#!/usr/bin/env python3
"""
CP-Width-Guided Adaptive Aggregation for Point Prediction.

Two-stage procedure:
  Stage 1 (weight construction):
    - Each base model (OLS, RF, DNN) gets locally adaptive CP intervals on Cal1 via
      normalized conformal prediction (RF-fitted absolute residuals).
    - Per-sample interval widths L_m(x) are used as an inverse reliability signal.
    - Fusion weights: w_m(x) = exp(-L_m(x)/τ) / Σ_k exp(-L_k(x)/τ).

  Stage 2 (coverage recovery):
    - The aggregated point estimator ŷ_ens = Σ w_m(x) ŷ_m(x) is treated as a new
      estimator and re-conformalized on Cal2 to restore finite-sample marginal coverage.

Key distinction from existing conformal aggregation literature:
  - Existing work fuses prediction sets/scores or does online model selection.
  - We use CP interval width as a local reliability signal for point estimator fusion.

Base models: OLS, Random Forest, DNN (3 heterogeneous classes)
Comparisons: Equal weight, Stacking (MSE-optimized), CP-width-guided (ours)
"""

import os, sys, time, json, warnings
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
BASE_SEED = 2024
N_REPS = 500
N = 500
P_LINEAR = 10
P_HIGHDIM = 100
S_HIGHDIM = 5
SIGMA = 1.0
CP_ALPHA = 0.10   # 90% nominal coverage

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {DEVICE}')

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'results', 'cp_fusion_results.json')

# ── DNN Model ───────────────────────────────────────────────────────────────
class DNNOriginal(nn.Module):
    """3 hidden layers (128-64-32), ReLU, dropout 0.2."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),    nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_dnn(model, X_train, y_train, X_val, y_val,
              epochs=200, lr=0.001, batch_size=64, patience=20):
    X_tr_t = torch.FloatTensor(X_train)
    y_tr_t = torch.FloatTensor(y_train)
    X_v_t  = torch.FloatTensor(X_val)
    y_v_t  = torch.FloatTensor(y_val)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True)
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_state = None
    wait = 0
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
            val_loss = nn.MSELoss()(model(X_v_t.to(DEVICE)), y_v_t.to(DEVICE)).item()
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
        return model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()


# ── Locally Adaptive CP with Normalized Conformal Prediction ────────────────
def fit_abs_residual_model(X_train, y_train, y_pred_train, seed=42):
    """Train an RF to predict absolute residuals |y - ŷ| from X.

    Returns a predictor with .predict(X) → σ̂(x) ≥ 0.
    This serves as a local difficulty/uncertainty estimate for normalized CP.
    """
    abs_resid = np.abs(y_train - y_pred_train)
    model = RandomForestRegressor(
        n_estimators=100, max_depth=5, min_samples_leaf=10,
        random_state=seed)
    model.fit(X_train, abs_resid)
    # Clip very small predictions to avoid division instability
    class ResidualModel:
        def __init__(self, inner):
            self.inner = inner
        def predict(self, X):
            return np.maximum(self.inner.predict(X), 0.05)  # lower bound σ̂ ≥ 0.05
    return ResidualModel(model)


def local_cp_width(y_cal, y_pred_cal, y_pred_new, sigma_cal, sigma_new, alpha=CP_ALPHA):
    """Compute locally adaptive CP interval width via normalized conformal.

    On calibration data: s_i = |y_i - ŷ_i| / σ̂_i  (normalized score)
    q = quantile(s)
    For new points: width(x) = 2 × q × σ̂(x)

    Returns: q (global quantile of normalized scores), widths (array per new point)
    """
    n = len(y_cal)
    scores = np.abs(y_cal - y_pred_cal) / sigma_cal
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q = np.quantile(scores, q_level, method='higher')
    widths = 2.0 * q * sigma_new
    return q, widths


def cp_coverage(y_true, y_pred, q, sigma):
    """Check |y - ŷ| / σ̂ ≤ q (normalized CP coverage)."""
    return np.mean(np.abs(y_true - y_pred) / sigma <= q)


# ── DGP functions ──────────────────────────────────────────────────────────
def dgp_linear():
    beta = np.array([2, -1.5, 0.8, 0.5, -0.3, 0, 0, 0, 0, 0])
    X = np.random.randn(N, P_LINEAR)
    y = X @ beta + np.random.randn(N) * SIGMA
    return X, y

def dgp_semiparametric():
    beta = np.array([2, -1.5, 0.8, 0.5, -0.3, 0, 0, 0, 0, 0])
    X = np.random.randn(N, P_LINEAR)
    y = X @ beta + 0.3 * np.sin(X[:, 0]) + np.random.randn(N) * SIGMA
    return X, y

def dgp_nonlinear():
    X = np.random.randn(N, P_LINEAR)
    y = (np.sin(X[:, 0]) + np.log(1 + np.abs(X[:, 1]))
         + X[:, 2] * X[:, 3] + np.random.randn(N) * SIGMA)
    return X, y

def dgp_highdim():
    beta = np.zeros(P_HIGHDIM)
    beta[:S_HIGHDIM] = [2, -1.5, 0.8, 0.5, -0.3]
    X = np.random.randn(N, P_HIGHDIM)
    y = X @ beta + np.random.randn(N) * SIGMA
    return X, y


# ── Main Experiment ────────────────────────────────────────────────────────
def run_aggregation_scenario(scenario_name, make_X_y):
    """
    Run CP-width-guided adaptive aggregation for one scenario.

    Data split: Train 50% | Cal1 10% | Cal2 10% | Test 30%
    Base models: OLS, RF, DNN
    Aggregation methods: equal-weight, stacking, cp-guided (softmax)
    """
    base_models = ['ols', 'rf', 'dnn']
    agg_methods = ['equal_weight', 'stacking', 'cp_guided']
    upper_bounds = ['best_val', 'best_oracle']
    all_methods = base_models + agg_methods + upper_bounds

    rep_results = {m: [] for m in all_methods}

    for rep in range(N_REPS):
        seed = BASE_SEED + rep
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ── Data split: Train 50% | Cal1 10% | Cal2 10% | Test 30% ──
        X, y = make_X_y()
        X_rest, X_test, y_rest, y_test = train_test_split(
            X, y, test_size=0.30, random_state=seed)
        cal_share = 2.0 / 7.0
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_rest, y_rest, test_size=cal_share, random_state=seed)
        X_cal1, X_cal2, y_cal1, y_cal2 = train_test_split(
            X_cal, y_cal, test_size=0.5, random_state=seed)

        # ── Train base models ──
        m_ols = LinearRegression().fit(X_train, y_train)
        m_rf = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=seed).fit(X_train, y_train)

        # DNN (scaled data + validation split)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_cal1_scaled  = scaler.transform(X_cal1)
        X_cal2_scaled  = scaler.transform(X_cal2)
        X_test_scaled  = scaler.transform(X_test)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=seed)
        m_dnn = DNNOriginal(X_tr.shape[1])
        m_dnn = train_dnn(m_dnn, X_tr, y_tr, X_val, y_val)

        models = {'ols': m_ols, 'rf': m_rf, 'dnn': m_dnn}

        def predict(name, X_raw, X_scaled):
            if name == 'dnn':
                return predict_dnn(models['dnn'], X_scaled)
            else:
                return models[name].predict(X_raw)

        # ── Predict on all sets ──
        preds = {}
        for m_name in base_models:
            preds[m_name] = {
                'train': predict(m_name, X_train, X_train_scaled),
                'cal1':  predict(m_name, X_cal1, X_cal1_scaled),
                'cal2':  predict(m_name, X_cal2, X_cal2_scaled),
                'test':  predict(m_name, X_test, X_test_scaled),
            }

        # ── Stage 1: locally adaptive CP widths on Cal1 ──
        # For each base model, fit an RF residual model → normalized CP → q + σ̂(x)
        width_info = {}  # per-model: {'q': float, 'sigma_*': array for each set}

        for m_name in base_models:
            resid_model = fit_abs_residual_model(
                X_train, y_train, preds[m_name]['train'], seed=seed)
            sigma_cal1 = resid_model.predict(X_cal1)
            sigma_test = resid_model.predict(X_test)

            q_m, _ = local_cp_width(
                y_cal1, preds[m_name]['cal1'], preds[m_name]['test'],
                sigma_cal1, sigma_test)

            width_info[m_name] = {
                'q': q_m,
                'sigma_cal1': sigma_cal1,
                'sigma_cal2': resid_model.predict(X_cal2),
                'sigma_test': sigma_test,
            }

        # Full local widths: L_m(x) = 2 × q_m × σ̂_m(x) on each set
        width_cal1 = {}
        width_cal2 = {}
        width_test = {}
        for m_name in base_models:
            q_m = width_info[m_name]['q']
            width_cal1[m_name] = 2.0 * q_m * width_info[m_name]['sigma_cal1']
            width_cal2[m_name] = 2.0 * q_m * width_info[m_name]['sigma_cal2']
            width_test[m_name] = 2.0 * q_m * width_info[m_name]['sigma_test']

        # τ = median of per-model median widths on Cal1 (not Cal2—keep Cal2 clean for Stage 2)
        tau = np.median([np.median(width_cal1[m]) for m in base_models])

        def softmax_weights(w_dict, model_names, tau_val):
            """w_m(x) = exp(-L_m(x)/τ) / Σ_k exp(-L_k(x)/τ)"""
            L = np.column_stack([w_dict[m] for m in model_names])   # (n, 3)
            exp_L = np.exp(-L / tau_val)
            return exp_L / exp_L.sum(axis=1, keepdims=True)         # (n, 3)

        # ── Aggregation weights ──
        w_eq = np.ones(len(base_models)) / len(base_models)

        cal1_stack_X = np.column_stack([preds[m]['cal1'] for m in base_models])
        stacking_reg = LinearRegression(fit_intercept=True).fit(cal1_stack_X, y_cal1)
        stack_coef = np.maximum(stacking_reg.coef_, 0)
        if stack_coef.sum() > 0:
            stack_coef = stack_coef / stack_coef.sum()
        else:
            stack_coef = np.ones(len(base_models)) / len(base_models)

        w_cp_cal2 = softmax_weights(width_cal2, base_models, tau)
        w_cp_test = softmax_weights(width_test, base_models, tau)

        # ── Evaluate individual models ──
        # Use Cal2 for standard split-CP on each base model → coverage baseline
        indiv_results = {}  # per-model: {'mse': float, 'cp_width': float, 'coverage': float}
        for m_name in base_models:
            resid_indiv = np.abs(y_cal2 - preds[m_name]['cal2'])
            q_indiv = np.quantile(resid_indiv,
                                   min(np.ceil((len(y_cal2) + 1) * 0.90) / len(y_cal2), 1.0),
                                   method='higher')
            test_pred_m = preds[m_name]['test']
            mse_m = mean_squared_error(y_test, test_pred_m)
            cov_m = np.mean(np.abs(y_test - test_pred_m) <= q_indiv)
            indiv_results[m_name] = {'mse': mse_m, 'cp_width': 2.0 * q_indiv, 'coverage': cov_m}
            rep_results[m_name].append(indiv_results[m_name])

        # ── Best single upper bounds ──
        # Oracle: lowest test MSE among base models (infeasible; upper bound)
        best_oracle_mse = min(indiv_results[m]['mse'] for m in base_models)
        # Validation-best: lowest Cal1 MSE → pick one model
        cal1_mses = {m: mean_squared_error(y_cal1, preds[m]['cal1']) for m in base_models}
        best_val_model = min(cal1_mses, key=cal1_mses.get)
        best_val_mse = indiv_results[best_val_model]['mse']
        best_val_width = indiv_results[best_val_model]['cp_width']
        best_val_cov = indiv_results[best_val_model]['coverage']

        rep_results['best_oracle'].append({'mse': best_oracle_mse, 'cp_width': 0, 'coverage': 0})
        rep_results['best_val'].append({'mse': best_val_mse, 'cp_width': best_val_width, 'coverage': best_val_cov})

        # ── Stage 2: ensemble CP on Cal2 (re-conformalize fused estimator) ──
        for method_name, w_const in [('equal_weight', w_eq), ('stacking', stack_coef)]:
            cal2_pred = sum(w_const[i] * preds[m]['cal2'] for i, m in enumerate(base_models))
            test_pred = sum(w_const[i] * preds[m]['test'] for i, m in enumerate(base_models))

            q_ens = np.quantile(np.abs(y_cal2 - cal2_pred),
                                min(np.ceil((len(y_cal2) + 1) * 0.90) / len(y_cal2), 1.0),
                                method='higher')

            rep_results[method_name].append({
                'mse':      mean_squared_error(y_test, test_pred),
                'cp_width': 2.0 * q_ens,
                'coverage': np.mean(np.abs(y_test - test_pred) <= q_ens),
            })

        # CP-guided ensemble (per-sample weights)
        cal2_pred_cp = np.sum(w_cp_cal2 * np.column_stack([preds[m]['cal2'] for m in base_models]), axis=1)
        test_pred_cp = np.sum(w_cp_test * np.column_stack([preds[m]['test'] for m in base_models]), axis=1)

        q_cp = np.quantile(np.abs(y_cal2 - cal2_pred_cp),
                           min(np.ceil((len(y_cal2) + 1) * 0.90) / len(y_cal2), 1.0),
                           method='higher')

        rep_results['cp_guided'].append({
            'mse':      mean_squared_error(y_test, test_pred_cp),
            'cp_width': 2.0 * q_cp,
            'coverage': np.mean(np.abs(y_test - test_pred_cp) <= q_cp),
        })

        if rep % 50 == 0 and rep > 0:
            print(f'  {scenario_name}: rep {rep}/{N_REPS} done', flush=True)

    # ── Aggregate results ──
    summary = {}
    for method in all_methods:
        metrics = rep_results[method]
        out = {
            'mse_mean':    float(np.mean([r['mse'] for r in metrics])),
            'mse_sd':      float(np.std([r['mse'] for r in metrics])),
            'width_mean':  float(np.mean([r['cp_width'] for r in metrics])),
            'width_sd':    float(np.std([r['cp_width'] for r in metrics])),
        }
        if 'coverage' in metrics[0]:
            covs = [r['coverage'] for r in metrics]
            out['coverage_mean'] = float(np.mean(covs))
            out['coverage_sd']   = float(np.std(covs))
        summary[method] = out

    print(f'\n  Results for {scenario_name}:')
    print(f'  {"Method":<20s} {"MSE":>10s} {"Width":>10s} {"Coverage":>10s}')
    print(f'  {"-"*50}')
    show_order = base_models + ['equal_weight', 'stacking', 'cp_guided', 'best_val', 'best_oracle']
    for method in show_order:
        s = summary[method]
        cov_str = f'{s.get("coverage_mean", 0):.4f}'
        print(f'  {method:<20s} {s["mse_mean"]:>8.4f}  {s["width_mean"]:>8.3f}  {cov_str:>8s}')

    return summary


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('  CP-Width-Guided Adaptive Aggregation for Point Prediction')
    print('=' * 60)
    print(f'  Replications: {N_REPS}, n={N}, CP alpha={CP_ALPHA}')
    print(f'  Device: {DEVICE}')
    print()

    scenarios = {
        'linear':        dgp_linear,
        'semiparametric': dgp_semiparametric,
        'nonlinear':      dgp_nonlinear,
        'highdim':        dgp_highdim,
    }

    all_results = {}
    for sname, dgp_func in scenarios.items():
        result = run_aggregation_scenario(sname, dgp_func)
        all_results[sname] = result

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {OUTPUT_FILE}')

    # Final table
    print('\n\n' + '=' * 120)
    print(f'{"Method":<20s}', end='')
    for sc in scenarios:
        print(f'{"MSE":>16s} {"Width":>10s} {"Cov":>8s}  ', end='')
    print()
    print('-' * 120)

    show_methods = ['ols', 'rf', 'dnn', 'equal_weight', 'stacking', 'cp_guided', 'best_val', 'best_oracle']
    for m in show_methods:
        print(f'{m:<20s}', end='')
        for sc in scenarios:
            if sc in all_results and m in all_results[sc]:
                r = all_results[sc][m]
                cov = r.get('coverage_mean', 0)
                print(f'{r["mse_mean"]:>8.4f} ±{r["mse_sd"]:.4f} {r["width_mean"]:>8.3f} {cov:>7.3f}  ', end='')
            else:
                print(f'{"N/A":>16s} {"N/A":>10s} {"N/A":>8s}  ', end='')
        print()
    print('=' * 120)
