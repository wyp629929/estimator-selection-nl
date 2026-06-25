#!/usr/bin/env python3
"""CQR experiment: conformalized quantile regression for all 6 regimes, 200 reps."""
import os, sys, time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import xgboost as xgb; import lightgbm as lgb

ALL = ['ols','ridge','lasso','rf','xgboost','lightgbm','dnn']
RD = '/Users/wangyaoping/Desktop/ML_Inference_Paper/results'

def est(m,s=0):
    if m=='ols': return LinearRegression()
    if m=='ridge': return Ridge(alpha=1.0)
    if m=='lasso': return Lasso(alpha=0.01,max_iter=5000)
    if m=='rf': return RandomForestRegressor(n_estimators=200,max_depth=10,min_samples_leaf=5,random_state=s)
    if m=='xgboost': return xgb.XGBRegressor(n_estimators=200,max_depth=6,learning_rate=0.1,random_state=s,verbosity=0)
    if m=='lightgbm': return lgb.LGBMRegressor(n_estimators=200,max_depth=6,learning_rate=0.1,verbose=-1,random_state=s)
    if m=='dnn': return MLPRegressor(hidden_layer_sizes=(128,64,32),max_iter=200,random_state=s)

def cq(r,a=0.10):
    n=len(r);l=min(np.ceil((n+1)*(1-a))/n,1.0);return np.quantile(r,l,method='higher')

# Quantile regressor factory
def qr(alpha, s=0):
    return HistGradientBoostingRegressor(loss='quantile', quantile=alpha, max_iter=200,
                                         max_depth=5, random_state=s, early_stopping=False)

# Signals
def s_lin(X): return X[:,:10]@np.array([2,-1.5,0.8,0.5,-0.3,0,0,0,0,0])
REGS = [
    ('linear',          lambda X,r: r.randn(len(X)), 10, s_lin),
    ('mildly_nonlinear',lambda X,r: r.randn(len(X)), 10, lambda X: s_lin(X)+0.3*np.sin(X[:,0])),
    ('nonlinear',       lambda X,r: r.randn(len(X)), 10, lambda X: np.sin(X[:,0])+np.log(1+np.abs(X[:,1]))+X[:,2]*X[:,3]),
    ('threshold',       lambda X,r: r.randn(len(X)), 10, lambda X: 2*(X[:,0]>0)+np.log(1+np.abs(X[:,1]))*(X[:,3]>0)-1),
    ('highdim',         lambda X,r: r.randn(len(X)), 100, lambda X: X[:,:100]@np.array([2,-1.5,0.8,0.5,-0.3]+[0]*95)),
    ('heteroscedastic', lambda X,r: r.randn(len(X))*(0.5+0.5*np.abs(X[:,0])), 10, s_lin),
]

t0 = time.time()
results = {}

for reg_name, noise_fn, p, sig in REGS:
    print(f'\n{reg_name} [t={time.time()-t0:.0f}s]', flush=True)
    rep_data = []

    for rep in range(200):
        rng = np.random.RandomState(2024 + rep)
        n = 500; X = rng.randn(n, p)
        f0 = sig(X); y = f0 + noise_fn(X, rng)

        # Split: 50% training, 20% calibration (for CQR), 15% test
        tr_sz = int(n * 0.50); cal_sz = int(n * 0.20)
        X_tr, X_ca, X_te = X[:tr_sz], X[tr_sz:tr_sz+cal_sz], X[-int(n*0.15):]
        y_tr, y_ca, y_te = y[:tr_sz], y[tr_sz:tr_sz+cal_sz], f0[-int(n*0.15):]

        # For oracle: models trained on full training set for test predictions
        models_full = {m: est(m, rep).fit(X_tr, y_tr) for m in ALL}

        # CQR: fit quantile regressors, calibrate, compute test widths
        alpha = 0.10  # 90% prediction interval
        cqr_widths = {}

        for m in ALL:
            # Fit lower and upper quantile regressors
            q_low = qr(alpha/2, rep).fit(X_tr, y_tr)
            q_high = qr(1-alpha/2, rep).fit(X_tr, y_tr)

            # Conformal calibration
            pred_low_ca = q_low.predict(X_ca)
            pred_high_ca = q_high.predict(X_ca)
            scores = np.maximum(pred_low_ca - y_ca, y_ca - pred_high_ca)
            q_cal = cq(scores)

            # Per-point test widths
            pred_low_te = q_low.predict(X_te)
            pred_high_te = q_high.predict(X_te)
            cqr_widths[m] = (pred_high_te - pred_low_te) + 2 * q_cal

        # Per-point CQR best
        cqr_best_pp = np.argmin(np.column_stack([cqr_widths[m] for m in ALL]), axis=1)

        # Oracle per-point
        mse_t = {m: mean_squared_error(y_te, models_full[m].predict(X_te)) for m in ALL}
        oracle_pp = np.argmin(np.column_stack([mse_t[m] for m in ALL]), axis=1)

        # Delta for CQR per-point
        delta_cqr = float(np.mean(cqr_best_pp != oracle_pp))

        rep_data.append(delta_cqr)

        if (rep+1) % 50 == 0:
            dc = np.mean(rep_data)
            print(f'  {rep+1}/200 Δ_cqr={dc:.3f} [t={time.time()-t0:.0f}s]', flush=True)

    dc = np.mean(rep_data); se = np.std(rep_data)/200**0.5
    results[reg_name] = {'delta_cqr': dc, 'se_cqr': se}
    with open(f'{RD}/cqr_results.json','w') as f:
        json.dump({**results,'_n_reps':200}, f, indent=2)
    print(f'  DONE: Δ_cqr={dc:.4f}({se:.4f}) [t={time.time()-t0:.0f}s]', flush=True)

print(f'\nTOTAL: {time.time()-t0:.0f}s')
for rn, rd in results.items():
    print(f'  {rn:20s} Δ_cqr={rd["delta_cqr"]:.4f}')
