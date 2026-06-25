#!/usr/bin/env python3
"""Simple CV-MSE experiment: all 6 regimes, 200 reps. No parallelism, no DNN in CV."""
import os, time, json, warnings
warnings.filterwarnings('ignore')
os.environ['LIGHTGBM_VERBOSE'] = '-1'

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import sys

BASE_SEED = 2024; ALL = ['ols','ridge','lasso','rf','xgboost','lightgbm','dnn']
RESULT_DIR = '/Users/wangyaoping/Desktop/ML_Inference_Paper/results'

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

# Signals
def s_lin(X): return X[:,:10]@np.array([2,-1.5,0.8,0.5,-0.3,0,0,0,0,0])
def s_semi(X): return s_lin(X)+0.3*np.sin(X[:,0])
def s_nl(X): return np.sin(X[:,0])+np.log(1+np.abs(X[:,1]))+X[:,2]*X[:,3]
def s_thr(X): return 2*(X[:,0]>0)+np.log(1+np.abs(X[:,1]))*(X[:,3]>0)-1
def s_hd(X): return X[:,:100]@np.array([2,-1.5,0.8,0.5,-0.3]+[0]*95)
def s_ht(X): return s_lin(X)

REGIMES = [
    ('linear', s_lin, 10, lambda X,r: r.randn(len(X))),
    ('semiparametric', s_semi, 10, lambda X,r: r.randn(len(X))),
    ('nonlinear', s_nl, 10, lambda X,r: r.randn(len(X))),
    ('threshold', s_thr, 10, lambda X,r: r.randn(len(X))),
    ('highdim', s_hd, 100, lambda X,r: r.randn(len(X))),
    ('heteroscedastic', s_ht, 10, lambda X,r: r.randn(len(X))*(0.5+0.5*np.abs(X[:,0]))),
]

t0 = time.time()
all_results = {}

for reg_name, sig, p, noise_fn in REGIMES:
    print(f'\n{reg_name} [t={time.time()-t0:.0f}s]', flush=True)
    rep_data = []
    debug_file = open(f'{RESULT_DIR}/debug_{reg_name}.txt', 'w', buffering=1)

    for rep in range(200):
        if rep % 10 == 0:
            debug_file.write(f'{reg_name} rep {rep}/200 [{time.time()-t0:.0f}s]\n')
            debug_file.flush()
        rng = np.random.RandomState(BASE_SEED + rep)
        n = 500; X = rng.randn(n, p)
        f0 = sig(X); y = f0 + noise_fn(X, rng)
        ts = int(n*0.50); te_sz = int(n*0.15)
        X_tr, y_tr = X[:ts], y[:ts]
        X_te, f_te = X[-te_sz:], f0[-te_sz:]

        # Train all models (on full training set)
        models = {}
        for m in ALL:
            try:
                models[m] = est(m, rep).fit(X_tr, y_tr)
            except Exception as e:
                debug_file.write(f'  ERROR training {m}: {e}\n')
                # Use a simple fallback
                models[m] = LinearRegression().fit(X_tr, y_tr)
        debug_file.write(f'{reg_name} rep {rep} models trained [{time.time()-t0:.0f}s]\n')

        # CP width via calibration set
        cs = int(ts*0.2)
        X_tr2, y_tr2 = X_tr[:-cs], y_tr[:-cs]
        X_ca, y_ca = X_tr[-cs:], y_tr[-cs:]
        models2 = {}
        for m in ALL:
            try:
                models2[m] = est(m, rep).fit(X_tr2, y_tr2)
            except:
                models2[m] = LinearRegression().fit(X_tr2, y_tr2)
        cw = {m: 2*cq(np.abs(y_ca - models2[m].predict(X_ca))) for m in ALL}
        cp_best = min(ALL, key=lambda m: cw[m])

        # Validation-MSE (80/20 split from training, for all 7 models)
        vs = int(ts*0.2)
        X_tr2, X_val = X_tr[:-vs], X_tr[-vs:]
        y_tr2, y_val = y_tr[:-vs], y_tr[-vs:]
        valid_err = {}
        for m in ALL:
            try:
                mm = est(m, rep).fit(X_tr2, y_tr2)
                valid_err[m] = mean_squared_error(y_val, mm.predict(X_val))
            except:
                valid_err[m] = 999.0
        val_best = min(ALL, key=lambda m: valid_err[m])

        # Test-set oracle (noiseless)
        mse_t = {m: mean_squared_error(f_te, models[m].predict(X_te)) for m in ALL}
        oracle = min(ALL, key=lambda m: mse_t[m])

        rep_data.append((1.0 if cp_best != oracle else 0.0, 1.0 if val_best != oracle else 0.0))

        if (rep+1) % 50 == 0:
            dc = np.mean([r[0] for r in rep_data])
            dv = np.mean([r[1] for r in rep_data])
            print(f'  {rep+1}/200 Δ_cp={dc:.3f} Δ_val={dv:.3f} [t={time.time()-t0:.0f}s]', flush=True)

    # Results for this regime
    dc = np.mean([r[0] for r in rep_data]); sc = np.std([r[0] for r in rep_data])/200**0.5
    dv = np.mean([r[1] for r in rep_data]); sv = np.std([r[1] for r in rep_data])/200**0.5
    all_results[reg_name] = {'delta_cp':dc,'se_cp':sc,'delta_val':dv,'se_val':sv}

    # Save after each regime
    with open(f'{RESULT_DIR}/cv_mse_simple_results.json','w') as f:
        json.dump({**all_results,'_n_reps':200,'_status':f'completed {reg_name}'}, f, indent=2)

    debug_file.close()
    print(f'  DONE: Δ_cp={dc:.4f} Δ_val={dv:.4f} [t={time.time()-t0:.0f}s]', flush=True)

print(f'\nTOTAL: {time.time()-t0:.0f}s')
for rn, rd in all_results.items():
    print(f'  {rn:15s} Δ_cp={rd["delta_cp"]:.4f} Δ_val={rd["delta_val"]:.4f}')
