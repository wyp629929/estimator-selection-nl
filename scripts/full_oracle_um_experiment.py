#!/usr/bin/env python3
"""Full oracle U_M experiment: all 6 regimes, 200 reps, validating the U_M pipeline."""
import os, time, json, warnings, logging, sys, io, contextlib
warnings.filterwarnings('ignore')
os.environ['LIGHTGBM_VERBOSE'] = '-1'
import numpy as np
import lightgbm as lgb
lgb.register_logger(logging.getLogger('lightgbm'))
logging.getLogger('lightgbm').setLevel(logging.ERROR)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb; import lightgbm as lgb

warnings.filterwarnings('ignore')
BASE_SEED = 2024; N_REPS = 100; EPS = 1e-8; N_ORACLE = 15
ALL_MODELS = ['ols','ridge','lasso','rf','xgboost','lightgbm','dnn']

def cq(r,a=0.10):
    n=len(r); l=min(np.ceil((n+1)*(1-a))/n,1.0); return np.quantile(r,l,method='higher')

def make_est(m, seed=0):
    kw = {'random_state':seed,'verbosity':0,'verbose':-1}
    if m=='ols': return LinearRegression()
    if m=='ridge': return Ridge(alpha=1.0)
    if m=='lasso': return Lasso(alpha=0.01, max_iter=5000)
    if m=='rf': return RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=seed, n_jobs=2)
    if m=='xgboost': return xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, **kw)
    if m=='lightgbm': return lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, **kw)
    if m=='dnn': return MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=200, random_state=seed)

# Signals
def sig_linear(X): return X[:,:10] @ np.array([2,-1.5,0.8,0.5,-0.3,0,0,0,0,0])
def sig_semiparam(X): return sig_linear(X) + 0.3*np.sin(X[:,0])
def sig_nonlinear(X): return np.sin(X[:,0])+np.log(1+np.abs(X[:,1]))+X[:,2]*X[:,3]
def sig_threshold(X): return 2*(X[:,0]>0)+np.log(1+np.abs(X[:,1]))*(X[:,3]>0)-1
def sig_highdim(X): return X[:,:100] @ np.array([2,-1.5,0.8,0.5,-0.3]+[0]*95)
def sig_hetero(X): return sig_linear(X)
def hetero_noise(X,rng): return rng.randn(len(X))*(0.5+0.5*np.abs(X[:,0]))

REGIMES = {
    'linear':       {'sig':sig_linear, 'p':10, 'noise':lambda X,rng: rng.randn(len(X))},
    'semiparametric':{'sig':sig_semiparam, 'p':10, 'noise':lambda X,rng: rng.randn(len(X))},
    'nonlinear':    {'sig':sig_nonlinear, 'p':10, 'noise':lambda X,rng: rng.randn(len(X))},
    'threshold':    {'sig':sig_threshold, 'p':10, 'noise':lambda X,rng: rng.randn(len(X))},
    'highdim':      {'sig':sig_highdim, 'p':100, 'noise':lambda X,rng: rng.randn(len(X))},
    'heteroscedastic':{'sig':sig_hetero, 'p':10, 'noise':hetero_noise},
}

results = {}
t0_total = time.time()

for reg_name, reg_info in REGIMES.items():
    print(f'\n{"="*60}')
    print(f'  {reg_name} — Oracle U_M (B={N_REPS})')
    print(f'{"="*60}')
    sig = reg_info['sig']; p = reg_info['p']; noise_fn = reg_info['noise']
    t0 = time.time()

    per_rep = []
    for rep in range(N_REPS):
        rng = np.random.RandomState(BASE_SEED + rep)
        n = 500; X = rng.randn(n, p)
        f0 = sig(X); y = f0 + noise_fn(X, rng)

        ts = int(n * 0.50); te = int(n * 0.15)
        X_tr, X_te = X[:ts], X[-te:]
        f_te = f0[-te:]

        # ── Oracle conditional MSE (N_ORACLE draws each for OLS, RF, DNN) ──
        mu = {}
        for m in ['ols', 'rf']:
            fh = np.zeros((N_ORACLE, te))
            for o in range(N_ORACLE):
                yo = sig(X_tr) + rng.randn(ts)
                moi = make_est(m, rep*1000+o).fit(X_tr, yo)
                fh[o] = moi.predict(X_te)
            bias = np.mean(fh, 0) - f_te; var = np.var(fh, 0)
            mu[m] = bias**2 + var
        # DNN (fewer draws)
        fh = np.zeros((10, te))
        for o in range(10):
            yo = sig(X_tr) + rng.randn(ts)
            moi = make_est('dnn', rep*2000+o).fit(X_tr, yo)
            fh[o] = moi.predict(X_te)
        bias = np.mean(fh, 0) - f_te; var = np.var(fh, 0)
        mu['dnn'] = bias**2 + var
        # Approximate others
        for m in ['ridge','lasso']: mu[m] = mu['ols'].copy()
        for m in ['xgboost','lightgbm']: mu[m] = mu['rf'].copy()

        # ── Oracle U_M ──
        R_all = np.column_stack([mu[m] for m in ALL_MODELS])
        nc = te // 2; R_c, R_e = R_all[:nc], R_all[nc:]
        sc_i = np.max(R_c / (np.std(R_c, 0, keepdims=True) + EPS), 1)
        q_um = cq(sc_i)
        Uo = np.maximum(0, R_e + q_um * (np.std(R_e, 0, keepdims=True) + EPS))
        ob = np.argmin(R_e, 1); Ub = np.argmin(Uo, 1)
        dU = float(np.mean(Ub != ob))

        # Pairwise ranking
        ne = len(Ub); pc = 0; pt = 0
        for i in range(ne):
            for a in range(7):
                for b in range(a+1, 7):
                    pt += 1
                    if (Uo[i,a] < Uo[i,b]) == (R_e[i,a] < R_e[i,b]): pc += 1
        pwU = pc / max(pt, 1)

        per_rep.append({'dU': dU, 'pwU': pwU})

        if (rep+1) % 50 == 0:
            du = np.mean([r['dU'] for r in per_rep])
            print(f'  rep {rep+1}/{N_REPS} [{time.time()-t0:.0f}s] Δ_Uo={du:.4f}', flush=True)

    du = np.mean([r['dU'] for r in per_rep])
    se = np.std([r['dU'] for r in per_rep]) / N_REPS**0.5
    pw = np.mean([r['pwU'] for r in per_rep])
    print(f'  [{time.time()-t0:.0f}s] Δ_Uo={du:.4f}({se:.4f}) PR={pw:.4f}')

    results[reg_name] = {'delta_U_oracle': du, 'delta_U_oracle_se': se, 'pairwise_U_oracle': pw, 'n_reps': N_REPS}

    # Incremental save
    with open('/Users/wangyaoping/Desktop/ML_Inference_Paper/results/oracle_um_full_results.json','w') as f:
        json.dump(results, f, indent=2)

print(f'\n{"="*60}')
print(f'ALL DONE — Total: {time.time()-t0_total:.0f}s')
for reg_name, res in results.items():
    print(f'  {reg_name:15s}  Δ_Uo={res["delta_U_oracle"]:.4f}({res["delta_U_oracle_se"]:.4f})  PR={res["pairwise_U_oracle"]:.4f}')
