#!/usr/bin/env python3
"""
Real-data DNN tuning experiment: PIMA Diabetes + Home Credit Default Risk.
Matches paper tasks: binary classification for both.
Metric: Brier score (not MSE - addressing P2 R3 from review).
"""

import os, sys, time, json, warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {DEVICE}')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'real_data_results.json')
HC_FILE = os.path.join(BASE_DIR, '..', 'data', 'application_train.csv')
PIMA_FILE = os.path.join(BASE_DIR, '..', 'data', 'pima_diabetes.csv')
SC_FILE = os.path.join(BASE_DIR, '..', 'superconductivty+data', 'train.csv')

# Reps
PIMA_REPS = 20
HC_REPS = 5        # Home Credit full (307K, DNN-heavy)
SC_REPS = 10       # Superconductivity


class DNNOriginal(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
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
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.l2_lambda = 1e-4
    def forward(self, x):
        return self.net(x).squeeze(-1)
    def l2_reg(self):
        return self.l2_lambda * sum(p.pow(2).sum() for p in self.parameters())


def train_dnn(model, X_tr, y_tr, X_val, y_val, epochs=200, bs=64, patience=20, tuned=False):
    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.FloatTensor(y_tr.values if hasattr(y_tr, 'values') else y_tr)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True)
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4 if tuned else 0)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-6) if tuned else None

    best, best_state, wait = float('inf'), None, 0
    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = nn.BCELoss()(model(Xb), yb)
            if tuned: loss = loss + model.l2_reg()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.BCELoss()(model(X_val_t.to(DEVICE)), y_val_t.to(DEVICE)).item()
        if sched: sched.step(vl)
        if vl < best - 1e-5:
            best, best_state, wait = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model


def brier(y_true, y_prob):
    return float(brier_score_loss(y_true, y_prob))


def run(name, X, y, reps, sample_size=None):
    print(f'\n{"="*60}\nDataset: {name} ({X.shape[0]} rows, {X.shape[1]} cols)\n{"="*60}')
    if sample_size and X.shape[0] > sample_size:
        idx = np.random.RandomState(42).choice(X.shape[0], sample_size, replace=False)
        X, y = X[idx], y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
        print(f'  Sampled to {sample_size}')

    methods = ['lr', 'ridge', 'rf', 'xgboost', 'lightgbm', 'dnn_original', 'dnn_tuned']
    res = {m: [] for m in methods}

    # Grid search for tree models (one-time, first rep)
    print('  Grid search for tree hyperparameters...', flush=True)
    seed_gs = 2024
    X_gs_tr, X_gs_te, y_gs_tr, y_gs_te = train_test_split(X, y, test_size=0.3, random_state=seed_gs)
    # RF
    rf_cands = [
        RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=seed_gs),
        RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=seed_gs),
        RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=3, random_state=seed_gs)]
    best_rf = max(rf_cands, key=lambda m: m.fit(X_gs_tr, y_gs_tr).score(X_gs_te, y_gs_te))
    print(f'    RF best: {best_rf}')
    # XGBoost
    xgb_cands = [
        xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=seed_gs, verbosity=0),
        xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=seed_gs, verbosity=0),
        xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.15, random_state=seed_gs, verbosity=0)]
    best_xgb = max(xgb_cands, key=lambda m: m.fit(X_gs_tr, y_gs_tr).score(X_gs_te, y_gs_te))
    print(f'    XGB best: {best_xgb}')
    # LightGBM
    lgb_cands = [
        lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, verbose=-1, random_state=seed_gs),
        lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, verbose=-1, random_state=seed_gs),
        lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.15, verbose=-1, random_state=seed_gs)]
    best_lgb = max(lgb_cands, key=lambda m: m.fit(X_gs_tr, y_gs_tr).score(X_gs_te, y_gs_te))
    print(f'    LGB best: {best_lgb}')

    for rep in range(reps):
        seed = 2024 + rep
        if rep % 5 == 0: print(f'  Rep {rep}/{reps}...', flush=True)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr_s, y_tr, test_size=0.2, random_state=seed)

        # Logistic Regression
        t0 = time.time()
        m = LogisticRegression(max_iter=1000, random_state=seed).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        res['lr'].append({'brier': brier(y_te, p), 'time': time.time() - t0})

        # Ridge Classifier
        t0 = time.time()
        m = RidgeClassifier(alpha=1.0, random_state=seed).fit(X_tr, y_tr)
        p = m.decision_function(X_te)
        p_prob = 1 / (1 + np.exp(-np.clip(p, -20, 20)))
        res['ridge'].append({'brier': brier(y_te, p_prob), 'time': time.time() - t0})

        # RF
        t0 = time.time()
        m = RandomForestClassifier(n_estimators=best_rf.n_estimators, max_depth=best_rf.max_depth, min_samples_leaf=best_rf.min_samples_leaf, random_state=seed).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        res['rf'].append({'brier': brier(y_te, p), 'time': time.time() - t0})

        # XGBoost
        t0 = time.time()
        m = xgb.XGBClassifier(n_estimators=best_xgb.n_estimators, max_depth=best_xgb.max_depth, learning_rate=best_xgb.learning_rate, random_state=seed, verbosity=0).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        res['xgboost'].append({'brier': brier(y_te, p), 'time': time.time() - t0})

        # LightGBM
        t0 = time.time()
        m = lgb.LGBMClassifier(n_estimators=best_lgb.n_estimators, max_depth=best_lgb.max_depth, learning_rate=best_lgb.learning_rate, verbose=-1, random_state=seed).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        res['lightgbm'].append({'brier': brier(y_te, p), 'time': time.time() - t0})

        # DNN Original
        torch.manual_seed(seed)
        t0 = time.time()
        model = DNNOriginal(X_tr2.shape[1])
        model = train_dnn(model, X_tr2, y_tr2, X_val, y_val, epochs=200, bs=64, patience=20)
        model.eval()
        with torch.no_grad():
            p = model(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy()
        res['dnn_original'].append({'brier': brier(y_te, p), 'time': time.time() - t0})

        # DNN Tuned
        torch.manual_seed(seed + 999)
        t0 = time.time()
        model_t = DNNTuned(X_tr2.shape[1])
        model_t = train_dnn(model_t, X_tr2, y_tr2, X_val, y_val, epochs=300, bs=32, patience=30, tuned=True)
        model_t.eval()
        with torch.no_grad():
            p = model_t(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy()
        res['dnn_tuned'].append({'brier': brier(y_te, p), 'time': time.time() - t0})

    summary = {}
    for m in methods:
        bs = [r['brier'] for r in res[m]]
        ts = [r['time'] for r in res[m]]
        summary[m] = {'brier_mean': float(np.mean(bs)), 'brier_sd': float(np.std(bs)), 'time_mean': float(np.mean(ts))}
        print(f'  {m:15s}  Brier = {summary[m]["brier_mean"]:.4f} ± {summary[m]["brier_sd"]:.4f}')
    return summary


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


def train_dnn_reg(model, X_tr, y_tr, X_val, y_val, epochs=200, bs=64, patience=20, tuned=False):
    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.FloatTensor(y_tr.values if hasattr(y_tr, 'values') else y_tr)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True)
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4 if tuned else 0)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-6) if tuned else None
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


def run_regression(name, X, y, reps):
    print(f'\n{"="*60}\nRegression: {name} ({X.shape[0]} rows, {X.shape[1]} cols)\n{"="*60}')
    methods = ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn_original', 'dnn_tuned']
    res = {m: [] for m in methods}
    for rep in range(reps):
        seed = 2024 + rep
        if rep % 5 == 0: print(f'  Rep {rep}/{reps}...', flush=True)
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
        m = Ridge(alpha=1.0, random_state=seed).fit(X_tr, y_tr)
        res['ridge'].append({'mse': mean_squared_error(y_te, m.predict(X_te)), 'time': time.time() - t0})
        # Lasso
        t0 = time.time()
        m = Lasso(alpha=0.01, max_iter=5000, random_state=seed).fit(X_tr, y_tr)
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
        model = DNNRegOriginal(X_tr2.shape[1])
        model = train_dnn_reg(model, X_tr2, y_tr2, X_val, y_val)
        model.eval()
        with torch.no_grad():
            p = model(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy()
        res['dnn_original'].append({'mse': mean_squared_error(y_te, p), 'time': time.time() - t0})
        # DNN Tuned
        torch.manual_seed(seed + 999)
        t0 = time.time()
        model_t = DNNRegTuned(X_tr2.shape[1])
        model_t = train_dnn_reg(model_t, X_tr2, y_tr2, X_val, y_val, epochs=300, bs=32, patience=30, tuned=True)
        model_t.eval()
        with torch.no_grad():
            p = model_t(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy()
        res['dnn_tuned'].append({'mse': mean_squared_error(y_te, p), 'time': time.time() - t0})
    summary = {}
    for m in methods:
        ms = [r['mse'] for r in res[m]]
        ts = [r['time'] for r in res[m]]
        summary[m] = {'mse_mean': float(np.mean(ms)), 'mse_sd': float(np.std(ms)), 'time_mean': float(np.mean(ts))}
        print(f'  {m:15s}  MSE = {summary[m]["mse_mean"]:.4f} ± {summary[m]["mse_sd"]:.4f}')
    return summary


if __name__ == '__main__':
    print(f'Real-data experiment: classification + regression\n')
    all_results = {}

    # PIMA
    df = pd.read_csv(PIMA_FILE)
    y_pima = df['outcome'].values
    X_pima = df.drop(columns=['outcome']).values
    print(f'PIMA: {X_pima.shape}')
    all_results['pima_diabetes'] = run('PIMA Diabetes', X_pima, y_pima, PIMA_REPS)

    # Home Credit
    print(f'\nLoading Home Credit...')
    df_hc = pd.read_csv(HC_FILE)
    hc_num = df_hc.select_dtypes(include=[np.number])
    hc_num = hc_num.drop(columns=['SK_ID_CURR'], errors='ignore')
    miss = hc_num.isnull().mean()
    keep = miss[miss < 0.5].index.tolist()
    hc_num = hc_num[keep]
    imp = SimpleImputer(strategy='median')
    X_hc = imp.fit_transform(hc_num.drop(columns=['TARGET']))
    y_hc = hc_num['TARGET'].values
    print(f'Home Credit: {X_hc.shape}')
    all_results['home_credit'] = run('Home Credit (full)', X_hc, y_hc, HC_REPS)

    # Superconductivity (regression)
    print(f'\nLoading Superconductivity data...')
    sc = pd.read_csv(SC_FILE)
    y_sc = sc['critical_temp'].values
    X_sc = sc.drop(columns=['critical_temp']).values
    print(f'Superconductivity: {X_sc.shape}')
    all_results['superconductivity'] = run_regression('Superconductivity', X_sc, y_sc, SC_REPS)

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved to {OUTPUT_FILE}')

    # Table
    print(f'\n{"="*90}')
    print(f'{"Method":<15} {"PIMA Brier":>15} {"HC Brier":>20} {"SC MSE":>20}')
    print(f'{"-"*90}')
    for m in ['lr', 'ridge', 'rf', 'xgboost', 'lightgbm', 'dnn_original', 'dnn_tuned']:
        row = f'{m:<15}'
        for ds in ['pima_diabetes', 'home_credit']:
            r = all_results[ds].get(m, {})
            row += f'{r.get("brier_mean", 0):.4f}±{r.get("brier_sd", 0):.4f}'.rjust(20)
        r_sc = all_results['superconductivity'].get(m, {})
        row += f'{r_sc.get("mse_mean", 0):.2f}±{r_sc.get("mse_sd", 0):.2f}'.rjust(20)
        print(row)
    print(f'{"="*90}')
