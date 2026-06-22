#!/usr/bin/env python3
"""
Bootstrap confidence interval coverage analysis.
Evaluates whether ML-based estimators achieve nominal 95% coverage
for the regression function at test points.

Linear and semiparametric scenarios (true f(X) known).
Metrics: empirical coverage rate, average CI width.
"""

import os, json, warnings, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {DEVICE}')

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'coverage_results.json')
N_REPS = 50
N = 500
P = 10
SIGMA = 1.0
B_BOOT = 200  # bootstrap resamples per rep
MC_DROPOUT_PASSES = 500  # forward passes for DNN MC Dropout

# DGP coefficients
BETA = np.array([2, -1.5, 0.8, 0.5, -0.3, 0, 0, 0, 0, 0])


# --- DNN with dropout for MC Dropout ---
class DNNWithDropout(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


def train_dnn(model, X_tr, y_tr, X_val, y_val, epochs=200, bs=64, patience=20):
    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.FloatTensor(y_tr)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    best, best_state, wait = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = nn.MSELoss()(model(Xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.MSELoss()(model(X_val_t.to(DEVICE)), y_val_t.to(DEVICE)).item()
        if vl < best - 1e-5:
            best, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model


def dgp_linear(seed):
    np.random.seed(seed)
    X = np.random.randn(N, P)
    f = X @ BETA
    y = f + np.random.randn(N) * SIGMA
    return X, y, f


def dgp_semiparametric(seed):
    np.random.seed(seed)
    X = np.random.randn(N, P)
    f = X @ BETA + 0.3 * np.sin(X[:, 0])
    y = f + np.random.randn(N) * SIGMA
    return X, y, f


def bootstrap_ci(predictions, alpha=0.05):
    """Compute percentile bootstrap CI for each test point."""
    lower = np.percentile(predictions, 100 * alpha / 2, axis=0)
    upper = np.percentile(predictions, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def run_scenario(name, dgp_fn):
    print(f'\n{"="*60}')
    print(f'Scenario: {name}')
    print(f'{"="*60}')

    methods = ['ols', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn']
    results = {m: {'coverage': np.zeros(N_REPS), 'ci_width': np.zeros(N_REPS)} for m in methods}

    for rep in range(N_REPS):
        if rep % 10 == 0:
            print(f'  rep {rep}/{N_REPS}', flush=True)
        seed = 2024 + rep
        X, y, f_true = dgp_fn(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr_s, y_tr, test_size=0.2, random_state=seed)

        # True signal at test points (no noise)
        if 'semiparametric' in name.lower():
            _, _, f_tr, f_te = train_test_split(X, f_true, test_size=0.3, random_state=seed)
        else:
            _, _, f_tr, f_te = train_test_split(X, f_true, test_size=0.3, random_state=seed)

        n_test = X_te.shape[0]

        # --- OLS: analytical CI ---
        m_ols = LinearRegression().fit(X_tr, y_tr)
        y_pred_ols = m_ols.predict(X_te)
        resid = y_tr - m_ols.predict(X_tr)
        sigma_hat = np.std(resid, ddof=P)
        X_tr_aug = np.column_stack([np.ones(X_tr.shape[0]), X_tr])
        X_te_aug = np.column_stack([np.ones(X_te.shape[0]), X_te])
        cov = sigma_hat**2 * np.linalg.inv(X_tr_aug.T @ X_tr_aug)
        se_pred = np.sqrt(np.diag(X_te_aug @ cov @ X_te_aug.T))
        t_val = 1.96  #近似
        ci_lower = y_pred_ols - t_val * se_pred
        ci_upper = y_pred_ols + t_val * se_pred
        covered = (f_te >= ci_lower) & (f_te <= ci_upper)
        results['ols']['coverage'][rep] = covered.mean()
        results['ols']['ci_width'][rep] = (ci_upper - ci_lower).mean()

        # --- Lasso: bootstrap CI ---
        m_lasso = Lasso(alpha=0.01, max_iter=5000, random_state=seed).fit(X_tr, y_tr)
        boot_preds = np.zeros((B_BOOT, n_test))
        for b in range(B_BOOT):
            idx = np.random.choice(X_tr.shape[0], X_tr.shape[0], replace=True)
            m_boot = Lasso(alpha=0.01, max_iter=3000, random_state=seed + b).fit(X_tr[idx], y_tr[idx])
            boot_preds[b] = m_boot.predict(X_te)
        lower, upper = bootstrap_ci(boot_preds)
        covered = (f_te >= lower) & (f_te <= upper)
        results['lasso']['coverage'][rep] = covered.mean()
        results['lasso']['ci_width'][rep] = (upper - lower).mean()

        # --- RF: bootstrap CI ---
        m_rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                                     min_samples_leaf=5, random_state=seed).fit(X_tr, y_tr)
        boot_preds = np.zeros((B_BOOT, n_test))
        for b in range(B_BOOT):
            idx = np.random.choice(X_tr.shape[0], X_tr.shape[0], replace=True)
            m_boot = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_leaf=5,
                random_state=seed + b).fit(X_tr[idx], y_tr[idx])
            boot_preds[b] = m_boot.predict(X_te)
        lower, upper = bootstrap_ci(boot_preds)
        covered = (f_te >= lower) & (f_te <= upper)
        results['rf']['coverage'][rep] = covered.mean()
        results['rf']['ci_width'][rep] = (upper - lower).mean()

        # --- XGBoost: bootstrap CI ---
        m_xgb = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  random_state=seed, verbosity=0).fit(X_tr, y_tr)
        boot_preds = np.zeros((B_BOOT, n_test))
        for b in range(B_BOOT):
            idx = np.random.choice(X_tr.shape[0], X_tr.shape[0], replace=True)
            m_boot = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                       random_state=seed + b, verbosity=0).fit(X_tr[idx], y_tr[idx])
            boot_preds[b] = m_boot.predict(X_te)
        lower, upper = bootstrap_ci(boot_preds)
        covered = (f_te >= lower) & (f_te <= upper)
        results['xgboost']['coverage'][rep] = covered.mean()
        results['xgboost']['ci_width'][rep] = (upper - lower).mean()

        # --- LightGBM: bootstrap CI ---
        m_lgb = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   verbose=-1, random_state=seed).fit(X_tr, y_tr)
        boot_preds = np.zeros((B_BOOT, n_test))
        for b in range(B_BOOT):
            idx = np.random.choice(X_tr.shape[0], X_tr.shape[0], replace=True)
            m_boot = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                                        verbose=-1, random_state=seed + b).fit(X_tr[idx], y_tr[idx])
            boot_preds[b] = m_boot.predict(X_te)
        lower, upper = bootstrap_ci(boot_preds)
        covered = (f_te >= lower) & (f_te <= upper)
        results['lightgbm']['coverage'][rep] = covered.mean()
        results['lightgbm']['ci_width'][rep] = (upper - lower).mean()

        # --- DNN: MC Dropout CI ---
        torch.manual_seed(seed)
        model = DNNWithDropout(X_tr2.shape[1])
        model = train_dnn(model, X_tr2, y_tr2, X_val, y_val)
        model.eval()
        model.enable_dropout()
        X_te_t = torch.FloatTensor(X_te_s).to(DEVICE)
        mc_preds = np.zeros((MC_DROPOUT_PASSES, n_test))
        with torch.no_grad():
            for b in range(MC_DROPOUT_PASSES):
                mc_preds[b] = model(X_te_t).cpu().numpy()
        lower, upper = bootstrap_ci(mc_preds)
        covered = (f_te >= lower) & (f_te <= upper)
        results['dnn']['coverage'][rep] = covered.mean()
        results['dnn']['ci_width'][rep] = (upper - lower).mean()

    # Summary
    print(f'\n--- Coverage Summary: {name} ---')
    print(f'{"Method":<12} {"Coverage (mean)":>18} {"CI Width (mean)":>18}')
    print('-' * 50)
    summary = {}
    for m in methods:
        cov_mean = float(np.mean(results[m]['coverage']))
        cov_sd = float(np.std(results[m]['coverage']))
        width_mean = float(np.mean(results[m]['ci_width']))
        width_sd = float(np.std(results[m]['ci_width']))
        summary[m] = {
            'coverage_mean': cov_mean,
            'coverage_sd': cov_sd,
            'ci_width_mean': width_mean,
            'ci_width_sd': width_sd,
        }
        print(f'{m:<12} {cov_mean:.3f} ({cov_sd:.3f})     {width_mean:.3f} ({width_sd:.3f})')

    return summary


if __name__ == '__main__':
    t0 = time.time()
    print(f'CI Coverage Analysis\n')
    all_results = {}

    for scenario, fn in [('linear', dgp_linear), ('semiparametric', dgp_semiparametric)]:
        all_results[scenario] = run_scenario(scenario, fn)

    all_results['config'] = {
        'n_reps': N_REPS,
        'n': N,
        'p': P,
        'sigma': SIGMA,
        'b_bootstrap': B_BOOT,
        'mc_dropout_passes': MC_DROPOUT_PASSES,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved to {OUTPUT_FILE}')
    print(f'Total time: {time.time() - t0:.1f}s')
