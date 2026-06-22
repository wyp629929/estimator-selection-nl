#!/usr/bin/env python3
"""
Higgs Boson experiment: 1M subset, binary classification.
Metrics: Brier score, training time.
6 methods: LR, RF, XGBoost, LightGBM, DNN original, DNN tuned.
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
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
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'real_data_results.json')
HIGGS_FILE = os.path.join(BASE_DIR, '..', 'HIGGS_1M.csv')
REPS = 5
SUBSAMPLE = 1_000_000
RANDOM_SEED = 42


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


def train_dnn(model, X_tr, y_tr, X_val, y_val, epochs=100, bs=256, patience=15, tuned=False):
    X_tr_t = torch.FloatTensor(X_tr)
    y_tr_t = torch.FloatTensor(y_tr.values if hasattr(y_tr, 'values') else y_tr)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True,
        num_workers=0, pin_memory=False)
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4 if tuned else 0)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1e-6) if tuned else None
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


def run():
    print(f'HIGGS experiment: subsample={SUBSAMPLE:,}, reps={REPS}')
    print(f'Loading {HIGGS_FILE}...')
    df = pd.read_csv(HIGGS_FILE, header=None)
    y = df[0].values
    X = df.iloc[:, 1:].values
    print(f'Data: {X.shape}')

    methods = ['lr', 'rf', 'xgboost', 'lightgbm', 'dnn_original', 'dnn_tuned']
    res = {m: [] for m in methods}

    for rep in range(REPS):
        seed = 2024 + rep
        print(f'\n  Rep {rep}/{REPS}...')
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr_s, y_tr, test_size=0.2, random_state=seed)

        t0 = time.time()
        m = LogisticRegression(max_iter=1000, random_state=seed).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        res['lr'].append({'brier': brier(y_te, p), 'time': time.time() - t0})
        print(f'    LR done ({time.time()-t0:.1f}s)')

        t0 = time.time()
        m = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5,
                                   random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        res['rf'].append({'brier': brier(y_te, p), 'time': time.time() - t0})
        print(f'    RF done ({time.time()-t0:.1f}s)')

        t0 = time.time()
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              random_state=seed, verbosity=0).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        res['xgboost'].append({'brier': brier(y_te, p), 'time': time.time() - t0})
        print(f'    XGBoost done ({time.time()-t0:.1f}s)')

        t0 = time.time()
        m = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               verbose=-1, random_state=seed).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        res['lightgbm'].append({'brier': brier(y_te, p), 'time': time.time() - t0})
        print(f'    LightGBM done ({time.time()-t0:.1f}s)')

        torch.manual_seed(seed)
        t0 = time.time()
        model = DNNOriginal(X_tr2.shape[1])
        model = train_dnn(model, X_tr2, y_tr2, X_val, y_val)
        model.eval()
        with torch.no_grad():
            p = model(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy()
        res['dnn_original'].append({'brier': brier(y_te, p), 'time': time.time() - t0})
        print(f'    DNN orig done ({time.time()-t0:.1f}s)')

        torch.manual_seed(seed + 999)
        t0 = time.time()
        model_t = DNNTuned(X_tr2.shape[1])
        model_t = train_dnn(model_t, X_tr2, y_tr2, X_val, y_val,
                            epochs=150, bs=256, patience=20, tuned=True)
        model_t.eval()
        with torch.no_grad():
            p = model_t(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy()
        res['dnn_tuned'].append({'brier': brier(y_te, p), 'time': time.time() - t0})
        print(f'    DNN tuned done ({time.time()-t0:.1f}s)')

    # Summary
    summary = {}
    print(f'\n{"="*50}')
    print(f'HIGGS (1M) Results')
    print(f'{"="*50}')
    for m in methods:
        bs = [r['brier'] for r in res[m]]
        ts = [r['time'] for r in res[m]]
        summary[m] = {
            'brier_mean': float(np.mean(bs)),
            'brier_sd': float(np.std(bs)),
            'time_mean': float(np.mean(ts))
        }
        print(f'  {m:15s}  Brier = {summary[m]["brier_mean"]:.4f} ± {summary[m]["brier_sd"]:.4f}  '
              f'({summary[m]["time_mean"]:.0f}s)')

    # Merge with existing results
    existing = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
    existing['higgs_1m'] = summary
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'\nSaved to {OUTPUT_FILE}')

    # Full table
    print(f'\n{"="*110}')
    print(f'{"Method":<15} {"PIMA":>15} {"HC":>15} {"SC":>15} {"HIGGS(1M)":>15}')
    print(f'{"-"*110}')
    for m in ['lr', 'rf', 'xgboost', 'lightgbm', 'dnn_original', 'dnn_tuned']:
        row = f'{m:<15}'
        for ds in ['pima_diabetes', 'home_credit', 'superconductivity', 'higgs_1m']:
            if ds in existing:
                r = existing[ds].get(m, {})
                if 'brier_mean' in r:
                    row += f'{r["brier_mean"]:.4f}'.rjust(15)
                elif 'mse_mean' in r:
                    row += f'{r["mse_mean"]:.1f}'.rjust(15)
                else:
                    row += f'{"-":>15}'
        print(row)
    print(f'{"="*110}')


if __name__ == '__main__':
    run()
