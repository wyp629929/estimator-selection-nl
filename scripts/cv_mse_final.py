#!/usr/bin/env python3
"""Final CV-MSE experiment — uses code path verified directly above."""
import os, time, json, warnings, sys
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import xgboost as xgb; import lightgbm as lgb

ALL = ['ols','ridge','lasso','rf','xgboost','lightgbm','dnn']
RESULT_DIR = '/Users/wangyaoping/Desktop/ML_Inference_Paper/results'

def est(m, s=0):
    if m=='ols': return LinearRegression()
    if m=='ridge': return Ridge(alpha=1.0)
    if m=='lasso': return Lasso(alpha=0.01, max_iter=5000)
    if m=='rf': return RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=s)
    if m=='xgboost': return xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=s, verbosity=0)
    if m=='lightgbm': return lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, verbose=-1, random_state=s)
    if m=='dnn': return MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=200, random_state=s)

def cq(r, a=0.10):
    n=len(r); l=min(np.ceil((n+1)*(1-a))/n, 1.0); return np.quantile(r, l, method='higher')

# Signals
def s_lin(X): return X[:,:10]@np.array([2,-1.5,0.8,0.5,-0.3,0,0,0,0,0])
REGS = [
    ('linear',          lambda X,r: r.randn(len(X)), 10, s_lin),
    ('semiparametric',  lambda X,r: r.randn(len(X)), 10, lambda X: s_lin(X)+0.3*np.sin(X[:,0])),
    ('nonlinear',       lambda X,r: r.randn(len(X)), 10, lambda X: np.sin(X[:,0])+np.log(1+np.abs(X[:,1]))+X[:,2]*X[:,3]),
    ('threshold',       lambda X,r: r.randn(len(X)), 10, lambda X: 2*(X[:,0]>0)+np.log(1+np.abs(X[:,1]))*(X[:,3]>0)-1),
    ('highdim',         lambda X,r: r.randn(len(X)), 100, lambda X: X[:,:100]@np.array([2,-1.5,0.8,0.5,-0.3]+[0]*95)),
    ('heteroscedastic', lambda X,r: r.randn(len(X))*(0.5+0.5*np.abs(X[:,0])), 10, s_lin),
]

t0 = time.time()
results = {}
log = open(f'{RESULT_DIR}/progress_log.txt', 'w', buffering=1)

for reg_name, noise_fn, p, sig in REGS:
    log.write(f'\n{reg_name} [t={time.time()-t0:.0f}s]\n'); log.flush()
    rep_data = []

    for rep in range(200):
        rng = np.random.RandomState(2024 + rep)
        X = rng.randn(500, p)
        f0 = sig(X); y = f0 + noise_fn(X, rng)
        X_tr, X_te = X[:250], X[-75:]; y_tr = y[:250]; f_te = f0[-75:]

        # Train 7 models
        models = {m: est(m, rep).fit(X_tr, y_tr) for m in ALL}

        # CP width (calibration 80/20)
        X_tr2, X_ca = X_tr[:-50], X_tr[-50:]
        y_tr2, y_ca = y_tr[:-50], y_tr[-50:]
        models2 = {m: est(m, rep).fit(X_tr2, y_tr2) for m in ALL}
        cw = {m: 2*cq(np.abs(y_ca - models2[m].predict(X_ca))) for m in ALL}
        cp_best = min(ALL, key=lambda m: cw[m])

        # Validation-MSE (holdout 80/20)
        X_tr3, X_va = X_tr[:-50], X_tr[-50:]
        y_tr3, y_va = y_tr[:-50], y_tr[-50:]
        valid_err = {}
        for m in ALL:
            mm = est(m, rep).fit(X_tr3, y_tr3)
            valid_err[m] = mean_squared_error(y_va, mm.predict(X_va))
        val_best = min(ALL, key=lambda m: valid_err[m])

        # Oracle (noiseless test)
        mse_t = {m: mean_squared_error(f_te, models[m].predict(X_te)) for m in ALL}
        oracle = min(ALL, key=lambda m: mse_t[m])

        rep_data.append((1.0 if cp_best != oracle else 0.0,
                         1.0 if val_best != oracle else 0.0))

        if (rep+1) % 50 == 0:
            dc = np.mean([r[0] for r in rep_data])
            dv = np.mean([r[1] for r in rep_data])
            log.write(f'  {rep+1}/200 Δ_cp={dc:.3f} Δ_val={dv:.3f} [t={time.time()-t0:.0f}s]\n'); log.flush()

    # Save
    dc = np.mean([r[0] for r in rep_data]); sc = np.std([r[0] for r in rep_data])/200**0.5
    dv = np.mean([r[1] for r in rep_data]); sv = np.std([r[1] for r in rep_data])/200**0.5
    results[reg_name] = {'delta_cp':dc,'se_cp':sc,'delta_val':dv,'se_val':sv}
    with open(f'{RESULT_DIR}/cv_mse_final_results.json','w') as f:
        json.dump({**results,'_n_reps':200,'_status':f'completed {reg_name}'}, f, indent=2)

    log.write(f'  DONE: Δ_cp={dc:.4f} Δ_val={dv:.4f} [t={time.time()-t0:.0f}s]\n'); log.flush()

log.write(f'\nTOTAL: {time.time()-t0:.0f}s\n')
for rn, rd in results.items():
    log.write(f'  {rn:15s} Δ_cp={rd["delta_cp"]:.4f} Δ_val={rd["delta_val"]:.4f}\n')
log.close()
# Also print to stdout for the Bash tool
print(f'TOTAL: {time.time()-t0:.0f}s')
for rn, rd in results.items():
    print(f'  {rn:15s} Δ_cp={rd["delta_cp"]:.4f} Δ_val={rd["delta_val"]:.4f}')
