#!/usr/bin/env python3
"""
Split conformal prediction coverage analysis — replaces bootstrap CI approach.
Covers all 4 simulation scenarios (linear, semiparametric, nonlinear, high-dim).
Reports empirical coverage + average interval width for 90% CP intervals.
500 Monte Carlo reps per scenario.
"""
import os, sys, json, warnings, time
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
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {DEVICE}')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'cp_coverage_results.json')

N_REPS = 500
N = 500
P_LINEAR = 10
P_HIGHDIM = 100
S_HIGHDIM = 5
SIGMA = 1.0
ALPHA = 0.10       # 90% prediction intervals
CALIB_SPLIT = 0.4  # 40% calibration

BETA = np.array([2, -1.5, 0.8, 0.5, -0.3, 0, 0, 0, 0, 0])


class DNNModel(nn.Module):
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


def train_dnn(model, X_tr, y_tr, X_val, y_val, epochs=200, bs=64, patience=20):
    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True)

    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.001)
    best_state, best_loss, wait = None, float('inf'), 0

    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            opt.zero_grad()
            loss = nn.MSELoss()(model(Xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.MSELoss()(model(X_val_t), y_val_t).item()
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


def split_conformal(y_test, pred_test, cal_scores, alpha=ALPHA):
    """Compute split conformal prediction intervals."""
    n_cal = len(cal_scores)
    q = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    cutoff = np.quantile(cal_scores, q, method='higher')
    lower = pred_test - cutoff
    upper = pred_test + cutoff
    covered = ((y_test >= lower) & (y_test <= upper)).mean()
    width = (upper - lower).mean()
    return covered, width


def run_cp_scenario(name, make_X_y, p=P_LINEAR):
    print(f'\n=== {name} (p={p}) ===')
    methods = ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn']
    # Coverage + width for each rep per method
    results = {m: {'cover': [], 'width': [], 'mse': []} for m in methods}

    for rep in range(N_REPS):
        if rep % 50 == 0:
            print(f'  Rep {rep}/{N_REPS}...', flush=True)
        np.random.seed(rep)
        torch.manual_seed(rep)

        X, y = make_X_y()

        # Main split: train+calibration vs test
        X_rest, X_test, y_rest, y_test = train_test_split(
            X, y, test_size=0.3, random_state=rep)
        # Split rest into proper train (60% of total) and calibration (40% of total)
        cal_size = int(len(y) * CALIB_SPLIT)
        X_tr = X_rest[cal_size:]
        y_tr = y_rest[cal_size:]
        X_cal = X_rest[:cal_size]
        y_cal = y_rest[:cal_size]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_cal_s = scaler.transform(X_cal)
        X_test_s = scaler.transform(X_test)

        # Validation split for DNN early stopping
        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr_s, y_tr, test_size=0.2, random_state=rep)

        # --- OLS ---
        m = LinearRegression().fit(X_tr, y_tr)
        pred_cal = m.predict(X_cal)
        pred_test = m.predict(X_test)
        cal_scores = np.abs(y_cal - pred_cal)
        cov, wid = split_conformal(y_test, pred_test, cal_scores)
        results['ols']['cover'].append(cov)
        results['ols']['width'].append(wid)
        results['ols']['mse'].append(mean_squared_error(y_test, pred_test))

        # --- Ridge ---
        m = Ridge(alpha=1.0).fit(X_tr, y_tr)
        pred_cal = m.predict(X_cal)
        pred_test = m.predict(X_test)
        cal_scores = np.abs(y_cal - pred_cal)
        cov, wid = split_conformal(y_test, pred_test, cal_scores)
        results['ridge']['cover'].append(cov)
        results['ridge']['width'].append(wid)
        results['ridge']['mse'].append(mean_squared_error(y_test, pred_test))

        # --- Lasso ---
        m = Lasso(alpha=0.01, max_iter=5000).fit(X_tr, y_tr)
        pred_cal = m.predict(X_cal)
        pred_test = m.predict(X_test)
        cal_scores = np.abs(y_cal - pred_cal)
        cov, wid = split_conformal(y_test, pred_test, cal_scores)
        results['lasso']['cover'].append(cov)
        results['lasso']['width'].append(wid)
        results['lasso']['mse'].append(mean_squared_error(y_test, pred_test))

        # --- RF ---
        m = RandomForestRegressor(n_estimators=200, max_depth=10,
                                  min_samples_leaf=5, random_state=rep).fit(X_tr, y_tr)
        pred_cal = m.predict(X_cal)
        pred_test = m.predict(X_test)
        cal_scores = np.abs(y_cal - pred_cal)
        cov, wid = split_conformal(y_test, pred_test, cal_scores)
        results['rf']['cover'].append(cov)
        results['rf']['width'].append(wid)
        results['rf']['mse'].append(mean_squared_error(y_test, pred_test))

        # --- XGBoost ---
        m = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                             random_state=rep, verbosity=0).fit(X_tr, y_tr)
        pred_cal = m.predict(X_cal)
        pred_test = m.predict(X_test)
        cal_scores = np.abs(y_cal - pred_cal)
        cov, wid = split_conformal(y_test, pred_test, cal_scores)
        results['xgboost']['cover'].append(cov)
        results['xgboost']['width'].append(wid)
        results['xgboost']['mse'].append(mean_squared_error(y_test, pred_test))

        # --- LightGBM ---
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                              verbose=-1, random_state=rep).fit(X_tr, y_tr)
        pred_cal = m.predict(X_cal)
        pred_test = m.predict(X_test)
        cal_scores = np.abs(y_cal - pred_cal)
        cov, wid = split_conformal(y_test, pred_test, cal_scores)
        results['lightgbm']['cover'].append(cov)
        results['lightgbm']['width'].append(wid)
        results['lightgbm']['mse'].append(mean_squared_error(y_test, pred_test))

        # --- DNN ---
        torch.manual_seed(rep)
        model = DNNModel(X_tr2.shape[1])
        model = train_dnn(model, X_tr2, y_tr2, X_val, y_val)
        model.eval()
        with torch.no_grad():
            pred_cal_dnn = model(torch.FloatTensor(X_cal_s).to(DEVICE)).cpu().numpy()
            pred_test_dnn = model(torch.FloatTensor(X_test_s).to(DEVICE)).cpu().numpy()
        cal_scores = np.abs(y_cal - pred_cal_dnn)
        cov, wid = split_conformal(y_test, pred_test_dnn, cal_scores)
        results['dnn']['cover'].append(cov)
        results['dnn']['width'].append(wid)
        results['dnn']['mse'].append(mean_squared_error(y_test, pred_test_dnn))

    # Summary
    summary = {}
    for method, res in results.items():
        summary[method] = {
            'coverage_mean': float(np.mean(res['cover'])),
            'coverage_sd': float(np.std(res['cover'])),
            'width_mean': float(np.mean(res['width'])),
            'width_sd': float(np.std(res['width'])),
            'mse_mean': float(np.mean(res['mse'])),
        }
        print(f'  {method:10s}: cov={summary[method]["coverage_mean"]:.3f} '
              f'width={summary[method]["width_mean"]:.3f} '
              f'mse={summary[method]["mse_mean"]:.3f}')

    return summary


# === Data Generating Processes ===
def dgp_linear():
    beta = BETA
    X = np.random.randn(N, P_LINEAR)
    y = X @ beta + np.random.randn(N) * SIGMA
    return X, y

def dgp_semiparametric():
    beta = BETA
    X = np.random.randn(N, P_LINEAR)
    y = X @ beta + 0.3 * np.sin(X[:, 0]) + np.random.randn(N) * SIGMA
    return X, y

def dgp_nonlinear():
    X = np.random.randn(N, 10)
    y = np.sin(X[:, 0]) + np.log(1 + np.abs(X[:, 1])) + X[:, 2] * X[:, 3] + np.random.randn(N) * SIGMA
    return X, y

def dgp_highdim():
    beta = np.zeros(P_HIGHDIM)
    beta[:S_HIGHDIM] = [2, -1.5, 0.8, 0.5, -0.3]
    X = np.random.randn(N, P_HIGHDIM)
    y = X @ beta + np.random.randn(N) * SIGMA
    return X, y


# === Main ===
if __name__ == '__main__':
    t0 = time.time()
    scenarios = {
        'linear': dgp_linear,
        'semiparametric': dgp_semiparametric,
        'nonlinear': dgp_nonlinear,
        'highdim': dgp_highdim,
    }
    all_results = {}
    for name, dgp_fn in scenarios.items():
        p = P_HIGHDIM if name == 'highdim' else P_LINEAR
        all_results[name] = run_cp_scenario(name, dgp_fn, p)

    all_results['config'] = {
        'n_reps': N_REPS,
        'n': N,
        'sigma': SIGMA,
        'alpha': ALPHA,
        'calib_split': CALIB_SPLIT,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {OUTPUT_FILE}')
    print(f'Total time: {(time.time() - t0) / 60:.1f} min')
