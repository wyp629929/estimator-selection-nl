#!/usr/bin/env python3
"""Oracle U_M validation: compare oracle-U_M vs estimated U_M for nonlinear regime."""
import os, sys, time, json, warnings
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')
BASE_SEED = 2024
N_REPS = 200
EPS = 1e-8

ALL_MODELS = ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn']

def signal_nonlinear(X):
    return np.sin(X[:, 0]) + np.log(1 + np.abs(X[:, 1])) + X[:, 2] * X[:, 3]

def cp_quantile(residuals, alpha=0.10):
    n = len(residuals)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(residuals, level, method='higher')

results = []
t0 = time.time()

for rep in range(N_REPS):
    seed = BASE_SEED + rep
    rng = np.random.RandomState(seed)
    n = 500
    p = 10
    X = rng.randn(n, p)
    f0 = signal_nonlinear(X)
    eps = rng.randn(n)
    y = f0 + eps

    # Split
    train_size = int(n * 0.50)
    meta_size = int(n * 0.15)
    mc_size = int(n * 0.10)
    cal_size = int(n * 0.10)
    test_size = int(n * 0.15)

    X_tr, X_te = X[:train_size], X[-test_size:]
    y_tr, y_te = y[:train_size], y[-test_size:]

    # Train 7 estimators
    models = {}
    models['ols'] = LinearRegression().fit(X_tr, y_tr)
    models['ridge'] = Ridge(alpha=1.0).fit(X_tr, y_tr)
    models['lasso'] = Lasso(alpha=0.01, max_iter=5000).fit(X_tr, y_tr)
    models['rf'] = RandomForestRegressor(200, max_depth=10, min_samples_leaf=5, random_state=seed).fit(X_tr, y_tr)
    models['xgboost'] = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=seed, verbosity=0).fit(X_tr, y_tr)
    models['lightgbm'] = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, verbose=-1, random_state=seed).fit(X_tr, y_tr)
    models['dnn'] = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=200, random_state=seed).fit(X_tr, y_tr)

    # Oracle conditional MSE via sampling (reduced draws)
    n_oracle = 100
    mu_test = {}
    for m in ALL_MODELS:
        f_hat_samples = np.zeros((n_oracle, test_size))
        for o in range(n_oracle):
            o_seed = seed + 10000 + o
            X_tr_o = X[:train_size]
            y_tr_o = signal_nonlinear(X_tr_o) + rng.randn(train_size)

            if m == 'ols':
                mo = LinearRegression().fit(X_tr_o, y_tr_o)
            elif m == 'ridge':
                mo = Ridge(alpha=1.0).fit(X_tr_o, y_tr_o)
            elif m == 'lasso':
                mo = Lasso(alpha=0.01, max_iter=5000).fit(X_tr_o, y_tr_o)
            elif m == 'rf':
                mo = RandomForestRegressor(200, max_depth=10, min_samples_leaf=5, random_state=o_seed).fit(X_tr_o, y_tr_o)
            elif m == 'xgboost':
                mo = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=o_seed, verbosity=0).fit(X_tr_o, y_tr_o)
            elif m == 'lightgbm':
                mo = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, verbose=-1, random_state=o_seed).fit(X_tr_o, y_tr_o)
            elif m == 'dnn':
                mo = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=200, random_state=o_seed).fit(X_tr_o, y_tr_o)

            f_hat_samples[o] = mo.predict(X_te)

        bias = np.mean(f_hat_samples, axis=0) - signal_nonlinear(X_te)
        var = np.var(f_hat_samples, axis=0)
        mu_test[m] = bias**2 + var

    # True excess loss per point
    ell = np.column_stack([mu_test[m] for m in ALL_MODELS])
    ell_min = np.min(ell, axis=1, keepdims=True)
    R_true = ell - ell_min  # shape (n_test, 7)

    # Oracle U_M: uses TRUE regret as signal
    # Conformalize the max-normalized true regret
    # Split test data: half for conformal calibration, half for evaluation
    n_test = test_size
    n_cal_u = n_test // 2
    X_cal_u = X_te[:n_cal_u]
    X_eval = X_te[n_cal_u:]
    R_cal = R_true[:n_cal_u]
    R_eval = R_true[n_cal_u:]

    # Joint conformal quantile from true regret on calibration half
    scores = np.max(R_cal / np.maximum(np.std(R_cal, axis=0, keepdims=True), EPS), axis=1)
    q_oracle = cp_quantile(scores)

    # Oracle U on evaluation half
    scale = np.std(R_eval, axis=0, keepdims=True) + EPS
    U_oracle = np.maximum(0, R_eval + q_oracle * scale)  # (n_eval, 7)

    # Per-point oracle best
    oracle_best = np.argmin(R_eval, axis=1)

    # U_M argmin
    U_best = np.argmin(U_oracle, axis=1)

    diag = {
        'delta_U_oracle': float(np.mean(U_best != oracle_best)),
        'pairwise_U_oracle': float(np.mean([
            1 for i in range(U_best.shape[0])
            for m1 in range(7) for m2 in range(m1+1, 7)
            if (U_oracle[i, m1] < U_oracle[i, m2]) == (R_eval[i, m1] < R_eval[i, m2])
        ])) if U_best.shape[0] > 0 else 0.5,
    }
    results.append(diag)

    if (rep + 1) % 50 == 0:
        print(f'  rep {rep}/{N_REPS} [{time.time()-t0:.0f}s, orcl_Δ={np.mean([r["delta_U_oracle"] for r in results]):.4f}]', flush=True)

delta_oracle = np.mean([r['delta_U_oracle'] for r in results])
se_oracle = np.std([r['delta_U_oracle'] for r in results]) / np.sqrt(N_REPS)
pairwise_oracle = np.mean([r['pairwise_U_oracle'] for r in results])

print(f'\n=== Oracle U_M Results (nonlinear, B={N_REPS}) ===')
print(f'Oracle U_M Δ = {delta_oracle:.4f} (SE = {se_oracle:.4f})')
print(f'Oracle U_M pairwise ranking = {pairwise_oracle:.4f}')
print(f'Estimated U_M Δ (from paper) ≈ 0.760')
print(f'Improvement: {0.760 - delta_oracle:.4f}')
print(f'Elapsed: {time.time()-t0:.0f}s')

out = {
    'regime': 'nonlinear',
    'n_reps': N_REPS,
    'delta_U_oracle': delta_oracle,
    'delta_U_oracle_se': se_oracle,
    'pairwise_U_oracle': pairwise_oracle,
    'delta_U_estimated_ref': 0.760,
    'improvement': 0.760 - delta_oracle,
}
os.makedirs('/Users/wangyaoping/Desktop/ML_Inference_Paper/results', exist_ok=True)
with open('/Users/wangyaoping/Desktop/ML_Inference_Paper/results/oracle_um_validation.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved to results/oracle_um_validation.json')
