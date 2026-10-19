#!/usr/bin/env python3
"""Run Superconductivity regression only. Merges results into real_data_results.json."""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {DEVICE}')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SC_FILE = os.path.join(BASE_DIR, '..', 'superconductivty+data', 'train.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'real_data_results.json')
SC_REPS = 10


class DNNRegOriginal(nn.Module):
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


class DNNRegTuned(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.l2_lambda = 1e-4
    def forward(self, x):
        return self.net(x).squeeze(-1)
    def l2_reg(self):
        return self.l2_lambda * sum(p.pow(2).sum() for p in self.parameters())


def train_dnn(model, X_tr, y_tr, X_val, y_val, epochs=200, bs=64, patience=20, tuned=False):
    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.FloatTensor(y_tr.values) if hasattr(y_tr, 'values') else torch.FloatTensor(y_tr)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val.values) if hasattr(y_val, 'values') else torch.FloatTensor(y_val)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4 if tuned else 0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-6) if tuned else None
    best, best_state, wait = float('inf'), None, 0
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
            best, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model


print('Running Superconductivity regression...')
sc = pd.read_csv(SC_FILE)
y_sc = sc['critical_temp'].values
X_sc = sc.drop(columns=['critical_temp']).values
print(f'Data: {X_sc.shape}')

methods = ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn_original', 'dnn_tuned']
res = {m: [] for m in methods}


# Grid search for tree models (one-time, unscaled data)
print('  Grid search for tree hyperparameters...', flush=True)
seed_gs = 2024
X_gs_tr, X_gs_te, y_gs_tr, y_gs_te = train_test_split(X_sc, y_sc, test_size=0.3, random_state=seed_gs)
rf_cands = [
    RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=seed_gs),
    RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=seed_gs),
    RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3, random_state=seed_gs)]
best_rf_sc = min(rf_cands, key=lambda m: mean_squared_error(y_gs_te, m.fit(X_gs_tr, y_gs_tr).predict(X_gs_te)))
print(f'    RF best: {best_rf_sc}')
xgb_cands = [
    xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=seed_gs, verbosity=0),
    xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=seed_gs, verbosity=0),
    xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.15, random_state=seed_gs, verbosity=0)]
best_xgb_sc = min(xgb_cands, key=lambda m: mean_squared_error(y_gs_te, m.fit(X_gs_tr, y_gs_tr).predict(X_gs_te)))
print(f'    XGB best: {best_xgb_sc}')
lgb_cands = [
    lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, verbose=-1, random_state=seed_gs),
    lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, verbose=-1, random_state=seed_gs),
    lgb.LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.15, verbose=-1, random_state=seed_gs)]
best_lgb_sc = min(lgb_cands, key=lambda m: mean_squared_error(y_gs_te, m.fit(X_gs_tr, y_gs_tr).predict(X_gs_te)))
print(f'    LGB best: {best_lgb_sc}')

for rep in range(SC_REPS):
    if rep % 5 == 0: print(f'  rep {rep}/{SC_REPS}', flush=True)
    seed = 2024 + rep
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_sc, test_size=0.3, random_state=seed)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr_s, y_tr, test_size=0.2, random_state=seed)

    t0 = time.time()
    m = LinearRegression().fit(X_tr, y_tr)
    res['ols'].append({'mse': mean_squared_error(y_te, m.predict(X_te)), 'time': time.time() - t0})

    t0 = time.time()
    m = Ridge(alpha=1.0, random_state=seed).fit(X_tr, y_tr)
    res['ridge'].append({'mse': mean_squared_error(y_te, m.predict(X_te)), 'time': time.time() - t0})

    t0 = time.time()
    m = Lasso(alpha=0.01, max_iter=5000, random_state=seed).fit(X_tr, y_tr)
    res['lasso'].append({'mse': mean_squared_error(y_te, m.predict(X_te)), 'time': time.time() - t0})

    t0 = time.time()
    m = RandomForestRegressor(n_estimators=best_rf_sc.n_estimators, max_depth=best_rf_sc.max_depth, min_samples_leaf=best_rf_sc.min_samples_leaf, random_state=seed).fit(X_tr, y_tr)
    res['rf'].append({'mse': mean_squared_error(y_te, m.predict(X_te)), 'time': time.time() - t0})

    t0 = time.time()
    m = xgb.XGBRegressor(n_estimators=best_xgb_sc.n_estimators, max_depth=best_xgb_sc.max_depth, learning_rate=best_xgb_sc.learning_rate, random_state=seed, verbosity=0).fit(X_tr, y_tr)
    res['xgboost'].append({'mse': mean_squared_error(y_te, m.predict(X_te)), 'time': time.time() - t0})

    t0 = time.time()
    m = lgb.LGBMRegressor(n_estimators=best_lgb_sc.n_estimators, max_depth=best_lgb_sc.max_depth, learning_rate=best_lgb_sc.learning_rate, verbose=-1, random_state=seed).fit(X_tr, y_tr)
    res['lightgbm'].append({'mse': mean_squared_error(y_te, m.predict(X_te)), 'time': time.time() - t0})

    torch.manual_seed(seed)
    t0 = time.time()
    model = DNNRegOriginal(X_tr2.shape[1])
    model = train_dnn(model, X_tr2, y_tr2, X_val, y_val)
    model.eval()
    with torch.no_grad():
        p = model(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy()
    res['dnn_original'].append({'mse': mean_squared_error(y_te, p), 'time': time.time() - t0})

    torch.manual_seed(seed + 999)
    t0 = time.time()
    model_t = DNNRegTuned(X_tr2.shape[1])
    model_t = train_dnn(model_t, X_tr2, y_tr2, X_val, y_val, epochs=300, bs=32, patience=30, tuned=True)
    model_t.eval()
    with torch.no_grad():
        p = model_t(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy()
    res['dnn_tuned'].append({'mse': mean_squared_error(y_te, p), 'time': time.time() - t0})

# Summary
summary = {}
for m in methods:
    ms = [r['mse'] for r in res[m]]
    ts = [r['time'] for r in res[m]]
    summary[m] = {'mse_mean': float(np.mean(ms)), 'mse_sd': float(np.std(ms)), 'time_mean': float(np.mean(ts))}
    print(f'  {m:15s}  MSE = {summary[m]["mse_mean"]:.4f} ± {summary[m]["mse_sd"]:.4f}')

# Merge with existing results
existing = {}
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        existing = json.load(f)
existing['superconductivity'] = summary
with open(OUTPUT_FILE, 'w') as f:
    json.dump(existing, f, indent=2)
print(f'\nSaved SC results to {OUTPUT_FILE}')

# Print combined table
print(f'\n{"="*90}')
print(f'{"Method":<15} {"PIMA Brier":>15} {"HC Brier":>20} {"SC MSE":>20}')
print(f'{"-"*90}')
for m in ['lr', 'ridge', 'rf', 'xgboost', 'lightgbm', 'dnn_original', 'dnn_tuned']:
    # Map method names from the existing data
    m_map = {'lr': 'lr', 'ridge': 'ridge', 'rf': 'rf', 'xgboost': 'xgboost',
             'lightgbm': 'lightgbm', 'dnn_original': 'dnn_original', 'dnn_tuned': 'dnn_tuned'}
    row = f'{m:<15}'
    for ds in ['pima_diabetes', 'home_credit']:
        if ds in existing:
            r = existing[ds].get(m_map[m], {})
            b_mean = r.get('brier_mean', 0)
            b_sd = r.get('brier_sd', 0)
            row += f'{b_mean:.4f}±{b_sd:.4f}'.rjust(20)
    if 'superconductivity' in existing:
        r_sc = existing['superconductivity'].get(m_map[m], {})
        if m in ['lr', 'ridge', 'rf', 'xgboost', 'lightgbm', 'dnn_original', 'dnn_tuned']:
            sc_mean = r_sc.get('mse_mean', 0)
            sc_sd = r_sc.get('mse_sd', 0)
            row += f'{sc_mean:.2f}±{sc_sd:.2f}'.rjust(20)
    print(row)
print(f'{"="*90}')
