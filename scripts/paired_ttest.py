#!/usr/bin/env python3
"""
Paired t-test: compare methods' MSE across 50 Monte Carlo reps.
Focus on nonlinear scenario at n=500.
"""

import os, json, warnings
import numpy as np
from scipy import stats

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

N_REPS = 500; N = 500; P = 10; SIGMA = 1.0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'paired_ttest_results.json')

class DNNOrig(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self, x): return self.net(x).squeeze(-1)

class DNNTuned(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Dropout(0.3),nn.Linear(256,128),nn.ReLU(),nn.BatchNorm1d(128),nn.Dropout(0.3),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,1))
        self.l2 = 1e-4
    def forward(self, x): return self.net(x).squeeze(-1)
    def l2_reg(self): return self.l2 * sum(p.pow(2).sum() for p in self.parameters())

def train_dnn(model, X_tr, y_tr, X_val, y_val, epochs=200, bs=64, patience=20, tuned=False):
    X_tr_t, y_tr_t = torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)
    X_val_t, y_val_t = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True)
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
            best, state, wait = vl, {k: v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    if state: model.load_state_dict(state)
    return model

def dgp_nonlinear(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    X = np.random.randn(N, P)
    y = np.sin(X[:,0]) + np.log(1+np.abs(X[:,1])) + X[:,2]*X[:,3] + np.random.randn(N)*SIGMA
    return X, y

print(f'Running {N_REPS} reps for paired t-test...')
methods = ['ols','ridge','lasso','rf','xgboost','lightgbm','dnn_original','dnn_tuned']
mse = {m: np.zeros(N_REPS) for m in methods}

for rep in range(N_REPS):
    if rep % 10 == 0: print(f'  rep {rep}/{N_REPS}', flush=True)
    seed = 2024 + rep
    X, y = dgp_nonlinear(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr); X_te_s = scaler.transform(X_te)
    X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr_s, y_tr, test_size=0.2, random_state=seed)

    mse['ols'][rep] = mean_squared_error(y_te, LinearRegression().fit(X_tr, y_tr).predict(X_te))
    mse['ridge'][rep] = mean_squared_error(y_te, Ridge(alpha=1.0).fit(X_tr, y_tr).predict(X_te))
    mse['lasso'][rep] = mean_squared_error(y_te, Lasso(alpha=0.01, max_iter=5000).fit(X_tr, y_tr).predict(X_te))
    mse['rf'][rep] = mean_squared_error(y_te, RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=seed).fit(X_tr, y_tr).predict(X_te))
    mse['xgboost'][rep] = mean_squared_error(y_te, xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=seed, verbosity=0).fit(X_tr, y_tr).predict(X_te))
    mse['lightgbm'][rep] = mean_squared_error(y_te, lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, verbose=-1, random_state=seed).fit(X_tr, y_tr).predict(X_te))
    torch.manual_seed(seed)
    mo = DNNOrig(X_tr2.shape[1]); mo = train_dnn(mo, X_tr2, y_tr2, X_val, y_val); mo.eval()
    with torch.no_grad(): mse['dnn_original'][rep] = mean_squared_error(y_te, mo(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy())
    torch.manual_seed(seed+999)
    mt = DNNTuned(X_tr2.shape[1]); mt = train_dnn(mt, X_tr2, y_tr2, X_val, y_val, tuned=True, bs=32, patience=30); mt.eval()
    with torch.no_grad(): mse['dnn_tuned'][rep] = mean_squared_error(y_te, mt(torch.FloatTensor(X_te_s).to(DEVICE)).cpu().numpy())

print('\n=== Paired t-test: nonlinear scenario, n=500, 500 reps ===')
print(f'{"Comparison":<30} {"Diff":>7} {"t":>7} {"p":>7} {"d":>6}  Signif')
print('-'*66)
label = {'ols':'OLS','ridge':'Ridge','lasso':'Lasso','rf':'RF','xgboost':'XGBoost','lightgbm':'LightGBM','dnn_original':'DNN_Orig','dnn_tuned':'DNN_Tuned'}
paired = []
for i, m1 in enumerate(methods):
    for m2 in methods[i+1:]:
        t_stat, p = stats.ttest_rel(mse[m1], mse[m2])
        d = np.mean(mse[m1] - mse[m2])
        d_cohen = np.mean(mse[m1] - mse[m2]) / np.std(mse[m1] - mse[m2])
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        paired.append({'m1':m1,'m2':m2,'diff':float(d),'t':float(t_stat),'p':float(p),'cohens_d':float(d_cohen),'sig':sig})
        print(f'{label[m1]:<12} vs {label[m2]:<12}  {d:+7.4f}  {t_stat:+6.2f}  {p:.4f}  {d_cohen:5.2f}  {sig}')

means = {m: float(np.mean(mse[m])) for m in methods}
ranked = sorted(means.items(), key=lambda x: x[1])
print(f'\n=== Ranked by mean MSE ===')
for i,(m,v) in enumerate(ranked):
    print(f'  {i+1}. {m:15s}  {v:.4f}')

output = {'mse_by_method':{m:mse[m].tolist() for m in methods},'means':means,'paired_tests':paired}
with open(OUTPUT_FILE,'w') as f: json.dump(output,f,indent=2)
print(f'\nSaved to {OUTPUT_FILE}')
