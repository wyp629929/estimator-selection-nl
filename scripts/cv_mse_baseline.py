#!/usr/bin/env python3
"""CV-MSE baseline for ranking-decision gap (nonlinear regime, 200 reps)."""
import os, sys, time, json, warnings
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
BASE_SEED = 2024
N_REPS = 200

ALL_MODELS = ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn']

def signal_nonlinear(X):
    return np.sin(X[:, 0]) + np.log(1 + np.abs(X[:, 1])) + X[:, 2] * X[:, 3]

def cv_mse(estimator, X, y, cv=5):
    scores = cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_squared_error')
    return -scores.mean()

results = []
t0 = time.time()

for rep in range(N_REPS):
    seed = BASE_SEED + rep
    rng = np.random.RandomState(seed)
    n = 500
    p = 10
    X = rng.randn(n, p)
    f = signal_nonlinear(X)
    eps = rng.randn(n)
    y = f + eps

    train_size = int(n * 0.5)
    test_size = int(n * 0.15)
    X_tr, X_te = X[:train_size], X[-test_size:]
    y_tr, y_te = y[:train_size], y[-test_size:]

    # Estimate conditional MSE oracle (using reduced draws for speed)
    n_oracle = 100
    mu_test = {}
    for m in ALL_MODELS:
        f_hat_samples = np.zeros((n_oracle, test_size))
        for o in range(n_oracle):
            o_seed = seed + 10000 + o
            X_tr_o = X[:train_size]
            f0 = signal_nonlinear(X_tr_o)
            y_tr_o = f0 + rng.randn(train_size)

            if m == 'ols':
                m_o = LinearRegression().fit(X_tr_o, y_tr_o)
            elif m == 'ridge':
                m_o = Ridge(alpha=1.0).fit(X_tr_o, y_tr_o)
            elif m == 'lasso':
                m_o = Lasso(alpha=0.01, max_iter=5000).fit(X_tr_o, y_tr_o)
            elif m == 'rf':
                m_o = RandomForestRegressor(200, max_depth=10, min_samples_leaf=5, random_state=o_seed).fit(X_tr_o, y_tr_o)
            elif m == 'xgboost':
                m_o = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=o_seed, verbosity=0).fit(X_tr_o, y_tr_o)
            elif m == 'lightgbm':
                m_o = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, verbose=-1, random_state=o_seed).fit(X_tr_o, y_tr_o)
            elif m == 'dnn':
                from sklearn.neural_network import MLPRegressor
                m_o = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=200, random_state=o_seed).fit(X_tr_o, y_tr_o)

            f_hat_samples[o] = m_o.predict(X_te)

        bias = np.mean(f_hat_samples, axis=0) - signal_nonlinear(X_te)
        var = np.var(f_hat_samples, axis=0)
        mu_test[m] = bias**2 + var

    # Oracle: per-point best model
    oracle_per_point = np.argmin(np.column_stack([mu_test[m] for m in ALL_MODELS]), axis=1)
    global_oracle_idx = np.bincount(oracle_per_point).argmax()
    global_oracle = ALL_MODELS[global_oracle_idx]

    # CV-MSE selection
    cv_errors = {}
    for m in ALL_MODELS:
        if m == 'ols':
            est = LinearRegression()
        elif m == 'ridge':
            est = Ridge(alpha=1.0)
        elif m == 'lasso':
            est = Lasso(alpha=0.01, max_iter=5000)
        elif m == 'rf':
            est = RandomForestRegressor(200, max_depth=10, min_samples_leaf=5, random_state=seed)
        elif m == 'xgboost':
            est = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=seed, verbosity=0)
        elif m == 'lightgbm':
            est = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, verbose=-1, random_state=seed)
        elif m == 'dnn':
            est = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=200, random_state=seed)
        cv_errors[m] = cv_mse(est, X_tr, y_tr)

    cv_best = min(ALL_MODELS, key=lambda m: cv_errors[m])

    # Δ: does CV-MSE pick the globally optimal model?
    misalignment = 1.0 if cv_best != global_oracle else 0.0

    results.append({
        'rep': rep,
        'cv_best': cv_best,
        'global_oracle': global_oracle,
        'misalignment': misalignment,
        'cv_errors': cv_errors,
    })

    if (rep + 1) % 50 == 0:
        print(f'  rep {rep}/{N_REPS} [{time.time()-t0:.0f}s]', flush=True)

delta = np.mean([r['misalignment'] for r in results])
se = np.std([r['misalignment'] for r in results]) / np.sqrt(N_REPS)

print(f'\nNonlinear regime — CV-MSE baseline (B={N_REPS}):')
print(f'  Δ = {delta:.4f} (SE = {se:.4f})')
print(f'  Random baseline = {6/7:.4f}')
print(f'  CP width Δ = 0.530 (from main experiment)')

# Also report which models are selected most
from collections import Counter
cv_counts = Counter(r['cv_best'] for r in results)
oracle_counts = Counter(r['global_oracle'] for r in results)
print(f'\n  CV-MSE selection breakdown:')
for m, c in cv_counts.most_common():
    print(f'    {m}: {c}/{N_REPS}')
print(f'\n  Global oracle breakdown:')
for m, c in oracle_counts.most_common():
    print(f'    {m}: {c}/{N_REPS}')
print(f'\nElapsed: {time.time()-t0:.0f}s')

# Save
out = {
    'regime': 'nonlinear',
    'n_reps': N_REPS,
    'delta_cv_mse': delta,
    'delta_cv_mse_se': se,
    'cv_selection_counts': dict(cv_counts),
    'oracle_counts': dict(oracle_counts),
    'per_rep': results,
}
os.makedirs('/Users/wangyaoping/Desktop/ML_Inference_Paper/results', exist_ok=True)
with open('/Users/wangyaoping/Desktop/ML_Inference_Paper/results/cv_mse_baseline.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved to results/cv_mse_baseline.json')
