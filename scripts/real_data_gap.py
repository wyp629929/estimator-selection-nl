#!/usr/bin/env python3
"""
Real-data validation of the ranking-decision gap.

Computes whether CP-based model selection (by interval width and by regret bound U_m)
identifies the empirically best model on held-out test data across 4 datasets.

Output: Δ (misalignment rate) for each dataset × signal combination.
"""

import os, sys, time, json, warnings, gzip
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
PROJ_DIR = os.path.join(BASE_DIR, '..')
OUTPUT_FILE = os.path.join(PROJ_DIR, 'results', 'real_data_gap.json')

# ── DNN ───────────────────────────────────────────────────────
class DNNRegressor(nn.Module):
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
        return model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy().flatten()

# ── Helpers ───────────────────────────────────────────────────
def cp_quantile(residuals, alpha=0.10):
    n = len(residuals)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(residuals, level, method='higher')

ALL_MODELS = ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn']

def evaluate_split(X_train, y_train, X_test, y_test, seed, label=''):
    """Train 7 models, compute CP widths and test MSE. Return best-model indicators."""
    n_train = len(X_train)
    n_cal = int(n_train * 0.30)
    X_tr, X_cal = X_train[:-n_cal], X_train[-n_cal:]
    y_tr, y_cal = y_train[:-n_cal], y_train[-n_cal:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    # Train models
    models = {}

    models['ols'] = LinearRegression().fit(X_tr, y_tr)

    models['ridge'] = Ridge(alpha=1.0).fit(X_tr, y_tr)

    models['lasso'] = Lasso(alpha=0.01, max_iter=5000).fit(X_tr, y_tr)

    models['rf'] = RandomForestRegressor(
        200, max_depth=10, min_samples_leaf=5, random_state=seed, n_jobs=-1).fit(X_tr, y_tr)

    models['xgboost'] = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=seed, verbosity=0).fit(X_tr, y_tr)

    models['lightgbm'] = lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        verbose=-1, random_state=seed).fit(X_tr, y_tr)

    # DNN: validation split from training
    X_tr2, X_va, y_tr2, y_va = train_test_split(X_tr_s, y_tr, test_size=0.2, random_state=seed)
    dnn = DNNRegressor(X_tr2.shape[1])
    dnn = train_dnn(dnn, X_tr2, y_tr2, X_va, y_va)
    models['dnn'] = dnn

    def predict(m, Xr, Xs):
        if isinstance(m, DNNRegressor):
            return predict_dnn(m, Xs)
        return m.predict(Xr)

    # ── Test MSE and CP widths ──
    results = {}
    for mn in ALL_MODELS:
        pred_cal = predict(models[mn], X_cal, X_cal_s)
        pred_test = predict(models[mn], X_test, X_test_s)
        mse = mean_squared_error(y_test, pred_test)
        q = cp_quantile(np.abs(y_cal - pred_cal))
        width = 2.0 * q
        results[mn] = {'mse': mse, 'cp_width': width}

    # ── Misalignment check ──
    # Best model by test MSE (ground truth for real data)
    best_mse = min(results.items(), key=lambda x: x[1]['mse'])[0]
    # Best model by CP width
    best_width = min(results.items(), key=lambda x: x[1]['cp_width'])[0]

    misalignment_width = 1.0 if best_width != best_mse else 0.0

    return {
        'misalignment_width': misalignment_width,
        'best_mse': best_mse,
        'best_width': best_width,
        'mses': {mn: results[mn]['mse'] for mn in ALL_MODELS},
    }


# ── Dataset loaders ──────────────────────────────────────────

def load_pima():
    """PIMA Diabetes. Classification → Brier score as MSE."""
    df = pd.read_csv(os.path.join(PROJ_DIR, 'data', 'pima_diabetes.csv'))
    y = df['outcome'].values.astype(float)
    X = df.drop(columns=['outcome']).values
    return X, y

def load_home_credit():
    """Home Credit Default Risk. Use a sample (first 50K for speed)."""
    df = pd.read_csv(os.path.join(PROJ_DIR, 'data', 'application_train.csv'))
    y = df['TARGET'].values.astype(float)
    df = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    # Keep numeric columns only
    df = df.select_dtypes(include=[np.number])
    # Fill NA with median
    df = df.fillna(df.median())
    X = df.values
    # Sample 50K for manageable DNN training
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), 50000, replace=False)
    X, y = X[idx], y[idx]
    print(f'  Home Credit: {X.shape}')
    return X, y

def load_superconductivity():
    """Superconductivity regression."""
    df = pd.read_csv(os.path.join(PROJ_DIR, 'superconductivty+data', 'train.csv'))
    y = df['critical_temp'].values
    X = df.drop(columns=['critical_temp']).values
    return X, y

def load_higgs():
    """HIGGS (1M sample). Classification → Brier score."""
    df = pd.read_csv(os.path.join(PROJ_DIR, 'HIGGS_1M.csv'), header=None)
    y = df.iloc[:, 0].values.astype(float)
    X = df.iloc[:, 1:].values
    return X, y


def run_dataset(name, loader, n_splits, is_classification=False):
    """Run dataset with n_splits train-test splits."""
    print(f'\n=== {name} ===')
    X, y = loader()

    results = []
    t0 = time.time()
    for split in range(n_splits):
        seed = 2024 + split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=seed)
        res = evaluate_split(X_tr, y_tr, X_te, y_te, seed, label=f'{name} split={split}')
        results.append(res)
        if (split + 1) % 10 == 0:
            print(f'  split {split+1}/{n_splits} [{time.time()-t0:.0f}s]', flush=True)

    delta_width = np.mean([r['misalignment_width'] for r in results])

    print(f'  Δ (Width): {delta_width:.4f} ({n_splits} splits)')
    print(f'  Elapsed: {time.time()-t0:.0f}s')

    return {
        'n_splits': n_splits,
        'n': len(X),
        'p': X.shape[1],
        'delta_width': float(delta_width),
        'best_model_by_mse': max(set(r['best_mse'] for r in results), key=lambda m:
                                  sum(1 for r in results if r['best_mse'] == m)),
    }


if __name__ == '__main__':
    print('=' * 60)
    print('  Real-Data Ranking–Decision Gap Validation')
    print(f'  Device: {DEVICE}')
    print('=' * 60)

    # Fewer splits for large datasets (DNN is slow)
    datasets = [
        ('PIMA',            load_pima,              50),
        ('HomeCredit',      load_home_credit,        20),
        ('Superconductivity', load_superconductivity, 30),
        # HIGGS excluded: 1M rows × DNN training per split is too slow
    ]

    all_results = {}
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    for name, loader, n_splits in datasets:
        res = run_dataset(name, loader, n_splits)
        all_results[name] = res
        # Save incrementally so partial results aren't lost
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'  [saved to {OUTPUT_FILE}]')
    print(f'\nResults saved to {OUTPUT_FILE}')
