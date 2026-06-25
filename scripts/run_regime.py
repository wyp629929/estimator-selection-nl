#!/usr/bin/env python3
"""Run a single regime for the CV-MSE experiment. Usage: python3 run_regime.py <regime_name>"""
import os, sys, time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
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

def s_lin(X): return X[:,:10]@np.array([2,-1.5,0.8,0.5,-0.3,0,0,0,0,0])
REGS = {
    'linear':          (10, s_lin, lambda X,r: r.randn(len(X))),
    'semiparametric':  (10, lambda X: s_lin(X)+0.3*np.sin(X[:,0]), lambda X,r: r.randn(len(X))),
    'nonlinear':       (10, lambda X: np.sin(X[:,0])+np.log(1+np.abs(X[:,1]))+X[:,2]*X[:,3], lambda X,r: r.randn(len(X))),
    'threshold':       (10, lambda X: 2*(X[:,0]>0)+np.log(1+np.abs(X[:,1]))*(X[:,3]>0)-1, lambda X,r: r.randn(len(X))),
    'highdim':         (100, lambda X: X[:,:100]@np.array([2,-1.5,0.8,0.5,-0.3]+[0]*95), lambda X,r: r.randn(len(X))),
    'heteroscedastic': (10, s_lin, lambda X,r: r.randn(len(X))*(0.5+0.5*np.abs(X[:,0]))),
}

reg_name = sys.argv[1]
if reg_name not in REGS:
    print(f'Unknown regime: {reg_name}. Choose from: {list(REGS.keys())}')
    sys.exit(1)

p, sig, noise_fn = REGS[reg_name]
t0 = time.time()
rep_data = []

for rep in range(200):
    rng = np.random.RandomState(2024 + rep)
    X = rng.randn(500, p)
    f0 = sig(X); y = f0 + noise_fn(X, rng)
    X_tr, X_te = X[:250], X[-75:]; y_tr = y[:250]; f_te = f0[-75:]

    dnn_model = est('dnn', rep).fit(X_tr, y_tr)
    models = {}
    for m in ALL:
        if m == 'dnn': models[m] = dnn_model
        else: models[m] = est(m, rep).fit(X_tr, y_tr)

    # CP width (separate DNN fit on X_tr2 for fair comparison)
    X_tr2, X_ca = X_tr[:-50], X_tr[-50:]; y_tr2, y_ca = y_tr[:-50], y_tr[-50:]
    cw = {}
    for m in ALL:
        mm = est(m, rep).fit(X_tr2, y_tr2)
        cw[m] = 2*cq(np.abs(y_ca - mm.predict(X_ca)))
    cp_best = min(ALL, key=lambda m: cw[m])

    # Validation (separate DNN fit on X_tr3)
    X_tr3, X_va = X_tr[:-50], X_tr[-50:]; y_tr3, y_va = y_tr[:-50], y_tr[-50:]
    valid_err = {}
    for m in ALL:
        mm = est(m, rep).fit(X_tr3, y_tr3)
        valid_err[m] = mean_squared_error(y_va, mm.predict(X_va))
    val_best = min(ALL, key=lambda m: valid_err[m])

    mse_t = {m: mean_squared_error(f_te, models[m].predict(X_te)) for m in ALL}
    oracle = min(ALL, key=lambda m: mse_t[m])

    rep_data.append((1.0 if cp_best != oracle else 0.0, 1.0 if val_best != oracle else 0.0))

    if (rep+1) % 50 == 0:
        dc = np.mean([r[0] for r in rep_data]); dv = np.mean([r[1] for r in rep_data])
        print(f'{reg_name} {rep+1}/200 Δ_cp={dc:.4f} Δ_val={dv:.4f} [{time.time()-t0:.0f}s]', flush=True)

dc = np.mean([r[0] for r in rep_data]); sc = np.std([r[0] for r in rep_data])/200**0.5
dv = np.mean([r[1] for r in rep_data]); sv = np.std([r[1] for r in rep_data])/200**0.5

# Save individual regime result
result = {'delta_cp': dc, 'se_cp': sc, 'delta_val': dv, 'se_val': sv}
with open(f'{RD}/regime_{reg_name}.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f'{reg_name} DONE Δ_cp={dc:.4f} Δ_val={dv:.4f} [{time.time()-t0:.0f}s]', flush=True)
