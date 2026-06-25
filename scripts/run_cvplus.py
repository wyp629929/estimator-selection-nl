#!/usr/bin/env python3
"""CV+ experiment: per-point CV+ widths for all 6 regimes, 200 reps."""
import os, time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
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
REGS = [
    ('linear',          lambda X,r: r.randn(len(X)), 10, s_lin),
    ('mildly_nonlinear',lambda X,r: r.randn(len(X)), 10, lambda X: s_lin(X)+0.3*np.sin(X[:,0])),
    ('nonlinear',       lambda X,r: r.randn(len(X)), 10, lambda X: np.sin(X[:,0])+np.log(1+np.abs(X[:,1]))+X[:,2]*X[:,3]),
    ('threshold',       lambda X,r: r.randn(len(X)), 10, lambda X: 2*(X[:,0]>0)+np.log(1+np.abs(X[:,1]))*(X[:,3]>0)-1),
    ('highdim',         lambda X,r: r.randn(len(X)), 100, lambda X: X[:,:100]@np.array([2,-1.5,0.8,0.5,-0.3]+[0]*95)),
    ('heteroscedastic', lambda X,r: r.randn(len(X))*(0.5+0.5*np.abs(X[:,0])), 10, s_lin),
]

t0 = time.time(); results = {}

for reg_name, noise_fn, p, sig in REGS:
    print(f'\n{reg_name} [t={time.time()-t0:.0f}s]', flush=True)
    rep_data = []

    for rep in range(200):
        rng = np.random.RandomState(2024+rep)
        n=500; X=rng.randn(n,p)
        f0=sig(X); y=f0+noise_fn(X,rng)
        X_tr, X_te = X[:350], X[-75:]; y_tr = y[:350]; f_te = f0[-75:]

        # CV+: 5-fold on training data
        kf = KFold(5, shuffle=True, random_state=rep)
        cv_models = {m: [] for m in ALL}
        cv_preds = {m: np.zeros(len(X_tr)) for m in ALL}
        cv_test_preds = {m: np.zeros((5, len(X_te))) for m in ALL}

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
            X_fold_tr, X_fold_va = X_tr[tr_idx], X_tr[va_idx]
            y_fold_tr, y_fold_va = y_tr[tr_idx], y_tr[va_idx]
            for m in ALL:
                mm = est(m, rep*10+fold).fit(X_fold_tr, y_fold_tr)
                cv_models[m].append(mm)
                cv_preds[m][va_idx] = mm.predict(X_fold_va)
                cv_test_preds[m][fold] = mm.predict(X_te)

        # CV+ interval widths for test points
        alpha=0.10; cvp_widths = {}
        for m in ALL:
            resid = np.abs(y_tr - cv_preds[m])
            q_cv = cq(resid)
            # Per-point width: range of fold predictions + 2*q_cv
            fold_range = np.max(cv_test_preds[m], axis=0) - np.min(cv_test_preds[m], axis=0)
            cvp_widths[m] = fold_range + 2*q_cv

        # Oracle per-point
        models_full = {m: est(m,rep).fit(X_tr,y_tr) for m in ALL}
        mse_t = {m: mean_squared_error(f_te, models_full[m].predict(X_te)) for m in ALL}
        oracle_pp = np.argmin(np.column_stack([mse_t[m] for m in ALL]), axis=1)

        # CV+ per-point best
        cvp_best_pp = np.argmin(np.column_stack([cvp_widths[m] for m in ALL]), axis=1)
        delta_cvp = float(np.mean(cvp_best_pp != oracle_pp))
        rep_data.append(delta_cvp)

        if (rep+1)%50==0:
            dc=np.mean(rep_data)
            print(f'  {rep+1}/200 Δ_cvp={dc:.3f} [{time.time()-t0:.0f}s]', flush=True)

    dc=np.mean(rep_data); se=np.std(rep_data)/200**0.5
    results[reg_name]={'delta_cvplus':dc,'se_cvplus':se}
    with open(f'{RD}/cvplus_results.json','w') as f:
        json.dump({**results,'_n_reps':200},f,indent=2)
    print(f'  DONE: Δ_cvp={dc:.4f}({se:.4f}) [{time.time()-t0:.0f}s]', flush=True)

print(f'\nTOTAL: {time.time()-t0:.0f}s')
for rn,rd in results.items():
    print(f'  {rn:20s} Δ_cvp={rd["delta_cvplus"]:.4f}')
