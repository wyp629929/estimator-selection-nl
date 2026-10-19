#!/usr/bin/env python3
"""
Sample size sensitivity analysis: run nonlinear scenario at n=200 / n=500 / n=2000.
Focus on whether DNN gap vs tree ensembles narrows with more data.
"""

import os, sys, time, json, warnings
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
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'sample_size_results.json')

N_REPS = 500
SIGMA = 1.0
P = 10


class DNNOriginal(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class DNNTuned(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.l2 = 1e-4
    def forward(self, x):
        return self.net(x).squeeze(-1)
    def l2_reg(self):
        return self.l2 * sum(p.pow(2).sum() for p in self.parameters())


def train_dnn(model, X_tr, y_tr, X_val, y_val, epochs=300, bs=64, patience=20, tuned=False):
    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.FloatTensor(y_tr)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True)
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4 if tuned else 0)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-6) if tuned else None

    best, state, wait = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = nn.MSELoss()(model(Xb), yb)
            if tuned: loss = loss + model.l2_reg()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.MSELoss()(model(X_val_t.to(DEVICE)), y_val_t.to(DEVICE)).item()
        if sched: sched.step(vl)
        if vl < best - 1e-5:
            best, state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    if state: model.load_state_dict(state)
    return model


def dgp_nonlinear(n):
    X = np.random.randn(n, P)
    y = np.sin(X[:, 0]) + np.log(1 + np.abs(X[:, 1])) + X[:, 2] * X[:, 3] + np.random.randn(n) * SIGMA
    return X, y


def run_n(n):
    """Run nonlinear scenario at sample size n."""
    methods = ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn_original', 'dnn_tuned']
    res = {m: [] for m in methods}

    for rep in range(N_REPS):
        seed = 2024 + rep
        if rep % 10 == 0: print(f'  n={n}: rep {rep}/{N_REPS}', flush=True)

        np.random.seed(seed); torch.manual_seed(seed)
        X, y = dgp_nonlinear(n)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr_s, y_tr, test_size=0.2, random_state=seed)

        # OLS
        t0 = time.time()
        m = LinearRegression().fit(X_tr, y_tr)
        res['ols'].append({'mse': mean_squared_error(y_te, m.predict(X_te)), 'time': time.time() - t0})

        # Ridge
        t0 = time.time()
        m = Ridge(alpha=1.0).fit(X_tr, y_tr)
        res['ridge'].append({'mse': mean_squared_error(y_te, m.predict(X_te)), 'time': time.time() - t0})

        # Lasso
        t0 = time.time()
        m = Lasso(alpha=0.01, max_iter=5000).fit(X_tr, y_tr)
        res['lasso'].append({'mse': mean_squared_error(y_te, m.predict(X_te)), 'time': time.time() - t0})

        # RF
        t0 = time.time()
        m = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=seed).fit(X_tr, y_tr)
        res['rf'].append({'mse': mean_squared_error(y_te, m.predict(X_te)), 'time': time.time() - t0})

        # XGBoost
        t0 = time.time()
        m = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=seed, verbosity=0).fit(X_tr, y_tr)
        res['xgboost'].append({'mse': mean_squared_error(y_te, m.predict(X_te)), 'time': time.time() - t0})

        # LightGBM
        t0 = time.time()
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, verbose=-1, random_state=seed).fit(X_tr, y_tr)
        res['lightgbm'].append({'mse': mean_squared_error(y_te, m.predict(X_te)), 'time': time.time() - t0})

        # DNN Original
        torch.manual_seed(seed)
        t0 = time.time()
        model = DNNOriginal(X_tr2.shape[1])
        model = train_dnn(model, X_tr2, y_tr2, X_val, y_val, epochs=300, bs=64, patience=20)
        model.eval()
        with torch.no_grad():
            p = model(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy()
        res['dnn_original'].append({'mse': mean_squared_error(y_te, p), 'time': time.time() - t0})

        # DNN Tuned
        torch.manual_seed(seed + 999)
        t0 = time.time()
        model_t = DNNTuned(X_tr2.shape[1])
        model_t = train_dnn(model_t, X_tr2, y_tr2, X_val, y_val, epochs=300, bs=32, patience=30, tuned=True)
        model_t.eval()
        with torch.no_grad():
            p = model_t(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy()
        res['dnn_tuned'].append({'mse': mean_squared_error(y_te, p), 'time': time.time() - t0})

    summary = {}
    for m in methods:
        ms = [r['mse'] for r in res[m]]
        ts = [r['time'] for r in res[m]]
        summary[m] = {'mse_mean': float(np.mean(ms)), 'mse_sd': float(np.std(ms)), 'time_mean': float(np.mean(ts))}
    return summary


if __name__ == '__main__':
    print(f'Sample size sensitivity — Nonlinear scenario, {N_REPS} reps each')
    sample_sizes = [200, 500, 2000]
    all_results = {}

    for n in sample_sizes:
        print(f'\n===== n = {n} =====')
        r = run_n(n)
        all_results[f'n={n}'] = r
        for m in ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn_original', 'dnn_tuned']:
            print(f'  {m:15s}  MSE = {r[m]["mse_mean"]:.4f} ± {r[m]["mse_sd"]:.4f}')

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved to {OUTPUT_FILE}')

    # Summary table
    print(f'\n{"="*80}')
    print(f'{"Method":<15}', end='')
    for n in sample_sizes:
        print(f'{"n="+str(n):>20}', end='')
    print()
    print('-'*80)
    for m in ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn_original', 'dnn_tuned']:
        print(f'{m:<15}', end='')
        for n in sample_sizes:
            r = all_results[f'n={n}'][m]
            print(f'{r["mse_mean"]:.3f}±{r["mse_sd"]:.3f}'.rjust(20), end='')
        print()
    print('='*80)
