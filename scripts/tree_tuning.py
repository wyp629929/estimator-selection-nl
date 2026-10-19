#!/usr/bin/env python3
"""
LightGBM/XGBoost hyperparameter tuning on nonlinear scenario (n=500, 50 reps).
Checks if default params are near-optimal vs tuned params.
"""
import os, json, warnings, itertools, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'tree_tuning_results.json')
N_REPS = 50; N = 500; P = 10; SIGMA = 1.0

def dgp(seed):
    np.random.seed(seed)
    X = np.random.randn(N, P)
    y = np.sin(X[:,0]) + np.log(1+np.abs(X[:,1])) + X[:,2]*X[:,3] + np.random.randn(N)*SIGMA
    return X, y

# Grids
lgb_grid = {'max_depth': [3, 6, 9], 'learning_rate': [0.05, 0.1, 0.2], 'n_estimators': [100, 200]}
xgb_grid = {'max_depth': [3, 6, 9], 'learning_rate': [0.05, 0.1, 0.2], 'n_estimators': [100, 200]}
lgb_default = {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200}
xgb_default = {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200}

print('Tree method tuning — nonlinear scenario, n=500, 50 reps')
results = {'lightgbm_default': [], 'lightgbm_tuned': [], 'xgboost_default': [], 'xgboost_tuned': []}
best_lgb_params = None; best_xgb_params = None
best_lgb_mse = float('inf'); best_xgb_mse = float('inf')

for rep in range(N_REPS):
    if rep % 10 == 0: print(f'  rep {rep}/{N_REPS}', flush=True)
    seed = 2024 + rep
    X, y = dgp(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)

    # LightGBM default
    m = lgb.LGBMRegressor(max_depth=lgb_default['max_depth'], learning_rate=lgb_default['learning_rate'],
                          n_estimators=lgb_default['n_estimators'], verbose=-1, random_state=seed)
    results['lightgbm_default'].append(mean_squared_error(y_te, m.fit(X_tr, y_tr).predict(X_te)))

    # XGBoost default
    m = xgb.XGBRegressor(max_depth=xgb_default['max_depth'], learning_rate=xgb_default['learning_rate'],
                          n_estimators=xgb_default['n_estimators'], verbosity=0, random_state=seed)
    results['xgboost_default'].append(mean_squared_error(y_te, m.fit(X_tr, y_tr).predict(X_te)))

    # LightGBM grid search on first 20 reps, then use best params
    if rep < 20:
        for md, lr, ne in itertools.product(lgb_grid['max_depth'], lgb_grid['learning_rate'], lgb_grid['n_estimators']):
            m = lgb.LGBMRegressor(max_depth=md, learning_rate=lr, n_estimators=ne, verbose=-1, random_state=seed)
            cv_mse = np.mean([mean_squared_error(y_te, m.fit(X_tr, y_tr).predict(X_te))])  # just use test set for simplicity
            # Actually use a proper validation: split tr further
            X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)
            m2 = lgb.LGBMRegressor(max_depth=md, learning_rate=lr, n_estimators=ne, verbose=-1, random_state=seed)
            m2.fit(X_tr2, y_tr2)
            val_mse = mean_squared_error(y_val, m2.predict(X_val))
            if val_mse < best_lgb_mse:
                best_lgb_mse = val_mse
                best_lgb_params = {'max_depth': md, 'learning_rate': lr, 'n_estimators': ne}

        for md, lr, ne in itertools.product(xgb_grid['max_depth'], xgb_grid['learning_rate'], xgb_grid['n_estimators']):
            X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)
            m2 = xgb.XGBRegressor(max_depth=md, learning_rate=lr, n_estimators=ne, verbosity=0, random_state=seed)
            m2.fit(X_tr2, y_tr2)
            val_mse = mean_squared_error(y_val, m2.predict(X_val))
            if val_mse < best_xgb_mse:
                best_xgb_mse = val_mse
                best_xgb_params = {'max_depth': md, 'learning_rate': lr, 'n_estimators': ne}

    # Use tuned params
    if best_lgb_params:
        m = lgb.LGBMRegressor(**best_lgb_params, verbose=-1, random_state=seed)
        results['lightgbm_tuned'].append(mean_squared_error(y_te, m.fit(X_tr, y_tr).predict(X_te)))
    if best_xgb_params:
        m = xgb.XGBRegressor(**best_xgb_params, verbosity=0, random_state=seed)
        results['xgboost_tuned'].append(mean_squared_error(y_te, m.fit(X_tr, y_tr).predict(X_te)))

print(f'\nBest LightGBM params: {best_lgb_params} (val MSE = {best_lgb_mse:.4f})')
print(f'Best XGBoost params: {best_xgb_params} (val MSE = {best_xgb_mse:.4f})')
print(f'\nLightGBM default: {np.mean(results["lightgbm_default"]):.4f} ± {np.std(results["lightgbm_default"]):.4f}')
if results['lightgbm_tuned']:
    print(f'LightGBM tuned:  {np.mean(results["lightgbm_tuned"]):.4f} ± {np.std(results["lightgbm_tuned"]):.4f}')
print(f'XGBoost default: {np.mean(results["xgboost_default"]):.4f} ± {np.std(results["xgboost_default"]):.4f}')
if results['xgboost_tuned']:
    print(f'XGBoost tuned:   {np.mean(results["xgboost_tuned"]):.4f} ± {np.std(results["xgboost_tuned"]):.4f}')

output = {
    'best_lgb_params': best_lgb_params, 'best_xgb_params': best_xgb_params,
    'lightgbm_default': {'mean': float(np.mean(results['lightgbm_default'])), 'sd': float(np.std(results['lightgbm_default']))},
    'xgboost_default': {'mean': float(np.mean(results['xgboost_default'])), 'sd': float(np.std(results['xgboost_default']))},
}
if results['lightgbm_tuned']:
    output['lightgbm_tuned'] = {'mean': float(np.mean(results['lightgbm_tuned'])), 'sd': float(np.std(results['lightgbm_tuned']))}
if results['xgboost_tuned']:
    output['xgboost_tuned'] = {'mean': float(np.mean(results['xgboost_tuned'])), 'sd': float(np.std(results['xgboost_tuned']))}
with open(OUTPUT_FILE, 'w') as f: json.dump(output, f, indent=2)
print(f'\nSaved to {OUTPUT_FILE}')
