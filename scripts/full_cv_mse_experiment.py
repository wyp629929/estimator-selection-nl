#!/usr/bin/env python3
"""Full CV-MSE baseline: all 6 regimes, 200 reps, using test MSE as oracle."""
import os, time, json, warnings, logging, sys, io
warnings.filterwarnings('ignore')
os.environ['LIGHTGBM_VERBOSE'] = '-1'
import numpy as np

# Suppress LightGBM C-level warnings (goes to stdout via fprintf)
def silent_lgb():
    """Context to suppress LightGBM output."""
    return contextlib.redirect_stdout(io.StringIO())
import contextlib

# Redirect stderr during LightGBM
import lightgbm as lgb
lgb.register_logger(logging.getLogger('lightgbm'))
logging.getLogger('lightgbm').setLevel(logging.ERROR)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import xgboost as xgb; import lightgbm as lgb

warnings.filterwarnings('ignore')
BASE_SEED = 2024; N_REPS = 200; EPS = 1e-8
ALL_MODELS = ['ols','ridge','lasso','rf','xgboost','lightgbm','dnn']

# Estimator constructors (for CV and training)
def make_est(m, seed=0):
    kw = {'random_state':seed,'verbosity':0,'verbose':-1}
    if m=='ols': return LinearRegression()
    if m=='ridge': return Ridge(alpha=1.0)
    if m=='lasso': return Lasso(alpha=0.01, max_iter=5000)
    if m=='rf': return RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=seed, n_jobs=2)
    if m=='xgboost': return xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, **kw)
    if m=='lightgbm': return lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, **kw)
    if m=='dnn': return MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=200, random_state=seed)

def cq(r,a=0.10):
    n=len(r); l=min(np.ceil((n+1)*(1-a))/n,1.0); return np.quantile(r,l,method='higher')

# Signal functions
def sig_linear(X): return X[:,:10] @ np.array([2,-1.5,0.8,0.5,-0.3,0,0,0,0,0])
def sig_semiparam(X): return sig_linear(X) + 0.3*np.sin(X[:,0])
def sig_nonlinear(X): return np.sin(X[:,0])+np.log(1+np.abs(X[:,1]))+X[:,2]*X[:,3]
def sig_threshold(X): return 2*(X[:,0]>0)+np.log(1+np.abs(X[:,1]))*(X[:,3]>0)-1
def sig_highdim(X): return X[:,:100] @ np.array([2,-1.5,0.8,0.5,-0.3]+[0]*95)
def sig_heteroscedastic_mean(X): return sig_linear(X)

def heteroscedastic_noise(X, rng): return rng.randn(len(X)) * (0.5+0.5*np.abs(X[:,0]))

REGIMES = {
    'linear':       {'sig': sig_linear,              'p':10,  'noise': lambda X,rng,s: rng.randn(len(X)),
                     'label':'Linear'},
    'semiparametric':{'sig': sig_semiparam,           'p':10,  'noise': lambda X,rng,s: rng.randn(len(X)),
                     'label':'Semiparam.'},
    'nonlinear':    {'sig': sig_nonlinear,            'p':10,  'noise': lambda X,rng,s: rng.randn(len(X)),
                     'label':'Nonlinear'},
    'threshold':    {'sig': sig_threshold,            'p':10,  'noise': lambda X,rng,s: rng.randn(len(X)),
                     'label':'Threshold'},
    'highdim':      {'sig': sig_highdim,              'p':100, 'noise': lambda X,rng,s: rng.randn(len(X)),
                     'label':'High-dim'},
    'heteroscedastic':{'sig': sig_heteroscedastic_mean,'p':10, 'noise': heteroscedastic_noise,
                     'label':'Heterosc.'},
}

cv_results = {}
t0_total = time.time()

for reg_name, reg_info in REGIMES.items():
    print(f'\n{"="*60}')
    print(f'  {reg_info["label"]} — B={N_REPS}')
    print(f'{"="*60}')
    sig = reg_info['sig']
    p = reg_info['p']
    noise_fn = reg_info['noise']
    t0 = time.time()

    per_rep = []
    for rep in range(N_REPS):
        seed = BASE_SEED + rep
        rng = np.random.RandomState(seed)
        n = 500
        X = rng.randn(n, p)
        f0 = sig(X)
        eps = noise_fn(X, rng, seed)
        y = f0 + eps

        ts = int(n * 0.50)
        X_tr, X_te = X[:ts], X[-int(n*0.15):]
        y_tr, y_te = y[:ts], f0[-int(n*0.15):]  # use f0 (noiseless) for oracle comparison
        # y_te_obs = y[-int(n*0.15):]  # noisy test observations

        # Train 7 models
        models = {}
        for m in ALL_MODELS:
            models[m] = make_est(m, seed).fit(X_tr, y_tr)

        # CP width selection (on calibration set)
        cs = int(ts * 0.2)
        X_tr2, X_ca = X_tr[:-cs], X_tr[-cs:]
        y_tr2, y_ca = y_tr[:-cs], y_tr[-cs:]
        models2 = {}
        for m in ALL_MODELS:
            models2[m] = make_est(m, rep).fit(X_tr2, y_tr2)
        cw = {}
        for m in ALL_MODELS:
            q = cq(np.abs(y_ca - models2[m].predict(X_ca)))
            cw[m] = 2.0 * q
        cp_best = min(ALL_MODELS, key=lambda m: cw[m])

        # CV-MSE selection (6 non-DNN models)
        cv_errs = {}
        for m in ['ols','ridge','lasso','rf','xgboost','lightgbm']:
            sc = cross_val_score(make_est(m, rep), X_tr, y_tr, cv=3, scoring='neg_mean_squared_error')
            cv_errs[m] = -sc.mean()
        # DNN: holdout validation
        vs = int(ts * 0.2)
        dnn_ms = mean_squared_error(y_tr[-vs:], make_est('dnn', rep).fit(X_tr[:-vs], y_tr[:-vs]).predict(X_tr[-vs:]))
        cv_errs['dnn'] = dnn_ms
        cv_best = min(ALL_MODELS, key=lambda m: cv_errs[m])

        # Test MSE oracle: model with lowest MSE on noiseless test set
        mse_test = {m: mean_squared_error(y_te, models[m].predict(X_te)) for m in ALL_MODELS}
        test_best = min(ALL_MODELS, key=lambda m: mse_test[m])

        per_rep.append({
            'cp_best': cp_best, 'cv_best': cv_best, 'test_best': test_best,
            'delta_cp': 1.0 if cp_best != test_best else 0.0,
            'delta_cv': 1.0 if cv_best != test_best else 0.0,
        })

        if (rep+1) % 50 == 0:
            dc = np.mean([r['delta_cp'] for r in per_rep])
            dv = np.mean([r['delta_cv'] for r in per_rep])
            print(f'  rep {rep+1}/{N_REPS} [{time.time()-t0:.0f}s] Δ_cp={dc:.3f} Δ_cv={dv:.3f}', flush=True)

    dc = np.mean([r['delta_cp'] for r in per_rep])
    dv = np.mean([r['delta_cv'] for r in per_rep])
    se_c = np.std([r['delta_cp'] for r in per_rep])/N_REPS**0.5
    se_v = np.std([r['delta_cv'] for r in per_rep])/N_REPS**0.5
    print(f'  [{time.time()-t0:.0f}s] FINAL: Δ_cp={dc:.4f}({se_c:.4f}) Δ_cv={dv:.4f}({se_v:.4f})')

    cv_results[reg_name] = {
        'delta_cp': dc, 'delta_cp_se': se_c,
        'delta_cv': dv, 'delta_cv_se': se_v,
        'n_reps': N_REPS,
    }
    # Save incremental results
    tmp_out = {k: {'delta_cp':v['delta_cp'],'delta_cp_se':v['delta_cp_se'],
                   'delta_cv':v['delta_cv'],'delta_cv_se':v['delta_cv_se']}
               for k,v in cv_results.items()}
    tmp_out['_n_reps'] = N_REPS
    with open('/Users/wangyaoping/Desktop/ML_Inference_Paper/results/cv_mse_full_results.json','w') as f:
        json.dump(tmp_out, f, indent=2)

print(f'\n{"="*60}')
print(f'ALL DONE — Total time: {time.time()-t0_total:.0f}s')
print(f'{"="*60}')
for reg_name, res in cv_results.items():
    print(f'  {REGIMES[reg_name]["label"]:15s}  CP Δ={res["delta_cp"]:.4f}({res["delta_cp_se"]:.4f})  CV Δ={res["delta_cv"]:.4f}({res["delta_cv_se"]:.4f})')
