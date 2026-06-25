#!/usr/bin/env python3
"""Master script: CV-MSE (B=200) + Oracle U_M (B=100) — all 6 regimes.
Writes results to JSON incrementally."""
import os, sys, time, json, warnings, logging, contextlib

# Write to log file directly (avoids shell redirect buffering issues)
log_file = open('/Users/wangyaoping/Desktop/ML_Inference_Paper/scripts/experiment_log.txt', 'w', buffering=1)
def log(msg):
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()
warnings.filterwarnings('ignore')
os.environ['LIGHTGBM_VERBOSE'] = '-1'
logging.getLogger('lightgbm').setLevel(logging.ERROR)

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

BASE_SEED = 2024
ALL_MODELS = ['ols','ridge','lasso','rf','xgboost','lightgbm','dnn']
RESULT_DIR = '/Users/wangyaoping/Desktop/ML_Inference_Paper/results'

def make_est(m, s=0):
    kw = {'random_state':s,'verbosity':0,'verbose':-1}
    if m=='ols': return LinearRegression()
    if m=='ridge': return Ridge(alpha=1.0)
    if m=='lasso': return Lasso(alpha=0.01,max_iter=5000)
    if m=='rf': return RandomForestRegressor(n_estimators=200,max_depth=10,min_samples_leaf=5,random_state=s,n_jobs=2)
    if m=='xgboost': return xgb.XGBRegressor(n_estimators=200,max_depth=6,learning_rate=0.1,**kw)
    if m=='lightgbm': return lgb.LGBMRegressor(n_estimators=200,max_depth=6,learning_rate=0.1,**kw)
    if m=='dnn': return MLPRegressor(hidden_layer_sizes=(128,64,32),max_iter=200,random_state=s)

def cq(r,a=0.10):
    n=len(r);l=min(np.ceil((n+1)*(1-a))/n,1.0);return np.quantile(r,l,method='higher')

# Signal definitions
SIG = {}
def _sig_linear(X): return X[:,:10]@np.array([2,-1.5,0.8,0.5,-0.3,0,0,0,0,0])
SIG['linear'] = (_sig_linear, 10, lambda X,rng: rng.randn(len(X)))
def _sig_semi(X): return _sig_linear(X) + 0.3*np.sin(X[:,0])
SIG['semiparametric'] = (_sig_semi, 10, lambda X,rng: rng.randn(len(X)))
def _sig_nonlinear(X): return np.sin(X[:,0])+np.log(1+np.abs(X[:,1]))+X[:,2]*X[:,3]
SIG['nonlinear'] = (_sig_nonlinear, 10, lambda X,rng: rng.randn(len(X)))
def _sig_threshold(X): return 2*(X[:,0]>0)+np.log(1+np.abs(X[:,1]))*(X[:,3]>0)-1
SIG['threshold'] = (_sig_threshold, 10, lambda X,rng: rng.randn(len(X)))
def _sig_highdim(X): return X[:,:100]@np.array([2,-1.5,0.8,0.5,-0.3]+[0]*95)
SIG['highdim'] = (_sig_highdim, 100, lambda X,rng: rng.randn(len(X)))
def _sig_hetero(X): return _sig_linear(X)
SIG['heteroscedastic'] = (_sig_hetero, 10, lambda X,rng: rng.randn(len(X))*(0.5+0.5*np.abs(X[:,0])))

cv_results = {}
um_results = {}
t_start = time.time()

for reg_name, (sig_fn, p, noise_fn) in SIG.items():
    log(f'\n===== {reg_name} =====')

    # ── CV-MSE (200 reps, test MSE as oracle) ──
    cv_per_rep = []
    for rep in range(200):
        rng = np.random.RandomState(BASE_SEED + rep)
        n = 500; X = rng.randn(n, p)
        f0 = sig_fn(X); y = f0 + noise_fn(X, rng)
        ts = int(n*0.50)
        X_tr, X_te = X[:ts], X[-int(n*0.15):]
        y_tr = y[:ts]; f_te = f0[-int(n*0.15):]

        models = {m: make_est(m, rep).fit(X_tr, y_tr) for m in ALL_MODELS}
        if rep==0: log(f'  CV: model training done [t={time.time()-t_start:.0f}s]')

        # CP width
        cs = int(ts*0.2)
        X_tr2, X_ca = X_tr[:-cs], X_tr[-cs:]
        y_tr2 = y_tr[:-cs]; y_ca = y_tr[-cs:]
        models2 = {m: make_est(m, rep).fit(X_tr2, y_tr2) for m in ALL_MODELS}
        cw = {m: 2*cq(np.abs(y_ca-models2[m].predict(X_ca))) for m in ALL_MODELS}
        cp_best = min(ALL_MODELS, key=lambda m: cw[m])

        # CV-MSE (3-fold)
        cv_e = {}
        for m in ['ols','ridge','lasso','rf','xgboost','lightgbm']:
            sc = cross_val_score(make_est(m, rep), X_tr, y_tr, cv=3, scoring='neg_mean_squared_error')
            cv_e[m] = -sc.mean()
        vs = int(ts*0.2)
        cv_e['dnn'] = mean_squared_error(y_tr[-vs:], make_est('dnn', rep).fit(X_tr[:-vs],y_tr[:-vs]).predict(X_tr[-vs:]))
        cv_best = min(ALL_MODELS, key=lambda m: cv_e[m])

        mse_t = {m: mean_squared_error(f_te, models[m].predict(X_te)) for m in ALL_MODELS}
        test_best = min(ALL_MODELS, key=lambda m: mse_t[m])
        cv_per_rep.append({'cp':1.0 if cp_best!=test_best else 0.0,
                           'cv':1.0 if cv_best!=test_best else 0.0})
        if (rep+1)%50==0: log(f'  CV: {rep+1}/200 [t={time.time()-t_start:.0f}s]')

    dc = np.mean([r['cp'] for r in cv_per_rep]); sc = np.std([r['cp'] for r in cv_per_rep])/200**0.5
    dv = np.mean([r['cv'] for r in cv_per_rep]); sv = np.std([r['cv'] for r in cv_per_rep])/200**0.5
    cv_results[reg_name] = {'delta_cp':dc,'se_cp':sc,'delta_cv':dv,'se_cv':sv}

    # ── Oracle U_M (100 reps, all 7 models estimated via 10 draws each) ──
    um_per_rep = []
    for rep in range(100):
        rng = np.random.RandomState(BASE_SEED + rep)
        X = rng.randn(n, p); f0 = sig_fn(X)
        y = f0 + noise_fn(X, rng)
        te = int(n*0.15); X_te = X[-te:]; f_te = f0[-te:]; X_tr = X[:int(n*0.50)]

        # Oracle: 10 draws for each of 7 models
        mu = {}
        for m in ALL_MODELS:
            nd = 10
            fh = np.zeros((nd, te))
            for o in range(nd):
                yo = sig_fn(X_tr) + rng.randn(len(X_tr))
                moi = make_est(m, rep*10000+o).fit(X_tr, yo)
                fh[o] = moi.predict(X_te)
            b = np.mean(fh,0)-f_te; v = np.var(fh,0)
            mu[m] = b**2 + v

        R_all = np.column_stack([mu[m] for m in ALL_MODELS])
        nc = te//2; R_c, R_e = R_all[:nc], R_all[nc:]
        sc_i = np.max(R_c/(np.std(R_c,0,keepdims=True)+1e-8),1)
        qu = cq(sc_i)
        Uo = np.maximum(0, R_e + qu*(np.std(R_e,0,keepdims=True)+1e-8))
        ob = np.argmin(R_e,1); Ub = np.argmin(Uo,1)
        dU = float(np.mean(Ub!=ob))

        # Pairwise
        ne = len(Ub); pc=0; pt=0
        for i in range(ne):
            for a in range(7):
                for b in range(a+1,7):
                    pt+=1
                    if (Uo[i,a]<Uo[i,b])==(R_e[i,a]<R_e[i,b]): pc+=1
        pwU = pc/max(pt,1)
        um_per_rep.append({'dU':dU,'pwU':pwU})

        if (rep+1)%50==0:
            du = np.mean([r['dU'] for r in um_per_rep])
            log(f'  UM: {rep+1}/100 [t={time.time()-t_start:.0f}s] Δ={du:.4f}')

    du = np.mean([r['dU'] for r in um_per_rep]); seu = np.std([r['dU'] for r in um_per_rep])/100**0.5
    pw = np.mean([r['pwU'] for r in um_per_rep])
    um_results[reg_name] = {'delta_U_oracle':du,'se_U_oracle':seu,'pairwise_U_oracle':pw}

    # Save checkpoints
    with open(f'{RESULT_DIR}/cv_mse_full_results.json','w') as f:
        json.dump({**cv_results, '_n_reps':200}, f, indent=2)
    with open(f'{RESULT_DIR}/oracle_um_full_results.json','w') as f:
        json.dump(um_results, f, indent=2)

    log(f'  {reg_name} DONE: CV Δ_cp={dc:.4f} Δ_cv={dv:.4f}  UM Δ_Uo={du:.4f}  [t={time.time()-t_start:.0f}s]')

log(f'\n{"="*60}')
log(f'ALL COMPLETE — Total: {time.time()-t_start:.0f}s')
for reg in cv_results:
    a=cv_results[reg]; b=um_results[reg]
    log(f'  {reg:15s}  CV: cp={a["delta_cp"]:.4f} cv={a["delta_cv"]:.4f}  |  UM: Δ={b["delta_U_oracle"]:.4f} PR={b["pairwise_U_oracle"]:.4f}')
log_file.close()
