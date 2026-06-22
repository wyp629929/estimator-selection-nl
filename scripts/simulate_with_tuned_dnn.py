#!/usr/bin/env python3
"""
Extended simulation comparing original DNN vs tuned DNN across 4 scenarios.
Uses PyTorch for neural network training (avoids TF segfault on Python 3.13 + ARM macOS).
Runs 50 Monte Carlo reps per scenario.
"""

import os, sys, time, json, warnings
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')

# Config
BASE_SEED = 2024
N_REPS = 500
N = 500
TEST_SIZE = 0.3
P_LINEAR = 10
P_HIGHDIM = 100
S_HIGHDIM = 5
SIGMA = 1.0

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {DEVICE}')

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulation_results_tuned_dnn.json')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# === DNN Models in PyTorch ===

class DNNOriginal(nn.Module):
    """Original DNN: 3 layers 128-64-32, dropout 0.2."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DNNTuned(nn.Module):
    """Tuned DNN: wider (256-128-64), BatchNorm, stronger dropout, L2 regularization."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self._l2_lambda = 1e-4

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def l2_loss(self):
        l2 = 0.0
        for p in self.parameters():
            l2 += p.pow(2).sum()
        return self._l2_lambda * l2


def train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=300, lr=0.001, batch_size=64, patience=20, is_tuned=False):
    """Train a PyTorch model with early stopping and optional LR scheduling."""
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4 if is_tuned else 0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-6) if is_tuned else None

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = nn.MSELoss()(pred, yb)
            if is_tuned:
                loss = loss + model.l2_loss() if hasattr(model, 'l2_loss') else loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t.to(DEVICE))
            val_loss = nn.MSELoss()(val_pred, y_val_t.to(DEVICE)).item()

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def cross_fit_estimate(X_train, y_train, X_test, ml_model_fn, n_folds=5, seed=42):
    """K-fold cross-fit estimation: OLS on all-but-k, then ML on fold k residuals.

    Args:
        X_train, y_train: training data
        X_test: test features
        ml_model_fn: callable(X, y) -> fitted ML model with .predict()
        n_folds: number of cross-fit folds (default 5)
        seed: random seed for fold split

    Returns:
        preds: test predictions (n_test,)
        beta_cv: list of K OLS coefficient vectors
        pred_linear_cv: list of K linear-only predictions
        pred_nonlinear_cv: list of K ML-only predictions
    """
    from sklearn.model_selection import KFold
    n_train = X_train.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    beta_cv = []
    pred_linear_cv = []
    pred_nonlinear_cv = []

    # For OLS without bias term, we add intercept
    for train_idx, val_idx in kf.split(X_train):
        X_tr_fold = X_train[train_idx]
        y_tr_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]

        # Step 1: OLS on all-but-k
        ols = LinearRegression().fit(X_tr_fold, y_tr_fold)
        beta_cv.append(np.concatenate([[ols.intercept_], ols.coef_]))

        # Step 2: Honest residuals on fold k
        y_linear_val = ols.predict(X_val_fold)
        r_val = y_val_fold - y_linear_val

        # Step 3: Train ML on (X_k, r_k)
        ml_model = ml_model_fn(X_val_fold, r_val)

        # Collect test predictions
        pred_linear = ols.predict(X_test)
        pred_nonlinear = ml_model.predict(X_test)

        pred_linear_cv.append(pred_linear)
        pred_nonlinear_cv.append(pred_nonlinear)

    # Aggregate: average across folds
    pred_linear_avg = np.mean(pred_linear_cv, axis=0)
    pred_nonlinear_avg = np.mean(pred_nonlinear_cv, axis=0)
    preds = pred_linear_avg + pred_nonlinear_avg

    return preds, beta_cv, pred_linear_cv, pred_nonlinear_cv


def run_scenario(name, make_X_y, true_beta=None):
    """Run one scenario for N_REPS repetitions."""
    print(f'\n=== {name} ===')
    methods = ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn_original', 'dnn_tuned',
               'cf_rf', 'cf_xgboost', 'cf_lightgbm', 'cf_dnn']
    all_results = {m: [] for m in methods}

    for rep in range(N_REPS):
        seed = BASE_SEED + rep
        if rep % 10 == 0:
            print(f'  Rep {rep}/{N_REPS}...', flush=True)

        np.random.seed(seed)
        torch.manual_seed(seed)
        X, y = make_X_y()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Validation split (20% of training for early stopping)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=seed
        )

        # OLS
        t0 = time.time()
        m = LinearRegression().fit(X_train, y_train)
        all_results['ols'].append({'mse': mean_squared_error(y_test, m.predict(X_test)), 'time': time.time() - t0})

        # Ridge
        t0 = time.time()
        m = Ridge(alpha=1.0).fit(X_train, y_train)
        all_results['ridge'].append({'mse': mean_squared_error(y_test, m.predict(X_test)), 'time': time.time() - t0})

        # Lasso
        t0 = time.time()
        m = Lasso(alpha=0.01, max_iter=5000).fit(X_train, y_train)
        all_results['lasso'].append({'mse': mean_squared_error(y_test, m.predict(X_test)), 'time': time.time() - t0})

        # RF
        t0 = time.time()
        m = RandomForestRegressor(n_estimators=200, max_depth=10,
                                  min_samples_leaf=5, random_state=seed).fit(X_train, y_train)
        all_results['rf'].append({'mse': mean_squared_error(y_test, m.predict(X_test)), 'time': time.time() - t0})

        # XGBoost
        t0 = time.time()
        m = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                             random_state=seed, verbosity=0).fit(X_train, y_train)
        all_results['xgboost'].append({'mse': mean_squared_error(y_test, m.predict(X_test)), 'time': time.time() - t0})

        # LightGBM
        t0 = time.time()
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                              verbose=-1, random_state=seed).fit(X_train, y_train)
        all_results['lightgbm'].append({'mse': mean_squared_error(y_test, m.predict(X_test)), 'time': time.time() - t0})

        # DNN Original
        torch.manual_seed(seed)
        t0 = time.time()
        model_orig = DNNOriginal(X_tr.shape[1])
        model_orig = train_pytorch_model(model_orig, X_tr, y_tr, X_val, y_val,
                                          epochs=200, lr=0.001, batch_size=64, patience=20, is_tuned=False)
        model_orig.eval()
        with torch.no_grad():
            pred_orig = model_orig(torch.FloatTensor(X_test_scaled).to(DEVICE)).cpu().numpy()
        all_results['dnn_original'].append({'mse': mean_squared_error(y_test, pred_orig), 'time': time.time() - t0})

        # DNN Tuned
        torch.manual_seed(seed + 999)
        t0 = time.time()
        model_tuned = DNNTuned(X_tr.shape[1])
        model_tuned = train_pytorch_model(model_tuned, X_tr, y_tr, X_val, y_val,
                                           epochs=300, lr=0.001, batch_size=32, patience=30, is_tuned=True)
        model_tuned.eval()
        with torch.no_grad():
            pred_tuned = model_tuned(torch.FloatTensor(X_test_scaled).to(DEVICE)).cpu().numpy()
        all_results['dnn_tuned'].append({'mse': mean_squared_error(y_test, pred_tuned), 'time': time.time() - t0})

        # === CF-SPE Estimators (Cross-Fit Semi-Parametric Estimation) ===
        # CF + RF
        t0 = time.time()
        pred_cf_rf, beta_cv, _, _ = cross_fit_estimate(
            X_train, y_train, X_test,
            lambda X, y: RandomForestRegressor(n_estimators=200, max_depth=10,
                                               min_samples_leaf=5, random_state=seed).fit(X, y),
            n_folds=5, seed=seed
        )
        all_results['cf_rf'].append({'mse': mean_squared_error(y_test, pred_cf_rf), 'time': time.time() - t0})

        # CF + XGBoost
        t0 = time.time()
        pred_cf_xgb, _, _, _ = cross_fit_estimate(
            X_train, y_train, X_test,
            lambda X, y: xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                          random_state=seed, verbosity=0).fit(X, y),
            n_folds=5, seed=seed
        )
        all_results['cf_xgboost'].append({'mse': mean_squared_error(y_test, pred_cf_xgb), 'time': time.time() - t0})

        # CF + LightGBM
        t0 = time.time()
        pred_cf_lgb, _, _, _ = cross_fit_estimate(
            X_train, y_train, X_test,
            lambda X, y: lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                           verbose=-1, random_state=seed).fit(X, y),
            n_folds=5, seed=seed
        )
        all_results['cf_lightgbm'].append({'mse': mean_squared_error(y_test, pred_cf_lgb), 'time': time.time() - t0})

        # CF + DNN
        t0 = time.time()
        def train_cf_dnn(X_ml, y_ml):
            # Scale features for DNN
            scaler_ml = StandardScaler()
            X_ml_scaled = scaler_ml.fit_transform(X_ml)
            _, X_val_ml, _, y_val_ml = train_test_split(X_ml_scaled, y_ml, test_size=0.2, random_state=seed)
            model = DNNOriginal(X_ml_scaled.shape[1])
            model = train_pytorch_model(model, X_ml_scaled, y_ml, X_val_ml, y_val_ml,
                                        epochs=200, lr=0.001, batch_size=64, patience=15, is_tuned=False)
            model.eval()
            class DNNWrapper:
                def __init__(self, model, scaler, device):
                    self.model = model
                    self.scaler = scaler
                    self.device = device
                def predict(self, X):
                    X_scaled = self.scaler.transform(X)
                    with torch.no_grad():
                        return self.model(torch.FloatTensor(X_scaled).to(self.device)).cpu().numpy()
            return DNNWrapper(model.to(DEVICE), scaler_ml, DEVICE)

        pred_cf_dnn, _, _, _ = cross_fit_estimate(
            X_train, y_train, X_test, train_cf_dnn, n_folds=5, seed=seed
        )
        all_results['cf_dnn'].append({'mse': mean_squared_error(y_test, pred_cf_dnn), 'time': time.time() - t0})

    # Summary
    summary = {}
    for method, results in all_results.items():
        mses = [r['mse'] for r in results]
        times = [r['time'] for r in results]
        summary[method] = {
            'mse_mean': float(np.mean(mses)),
            'mse_sd': float(np.std(mses)),
            'time_mean': float(np.mean(times)),
        }

    print(f'\n  Results:')
    for method, s in sorted(summary.items()):
        print(f'    {method:15s}  MSE = {s["mse_mean"]:.4f} ± {s["mse_sd"]:.4f}  ({s["time_mean"]:.2f}s)')

    return summary


# Data generating processes
def dgp_linear():
    beta = np.array([2, -1.5, 0.8, 0.5, -0.3, 0, 0, 0, 0, 0])
    X = np.random.randn(N, P_LINEAR)
    y = X @ beta + np.random.randn(N) * SIGMA
    return X, y

def dgp_semiparametric():
    beta = np.array([2, -1.5, 0.8, 0.5, -0.3, 0, 0, 0, 0, 0])
    X = np.random.randn(N, P_LINEAR)
    y = X @ beta + 0.3 * np.sin(X[:, 0]) + np.random.randn(N) * SIGMA
    return X, y

def dgp_nonlinear():
    X = np.random.randn(N, P_LINEAR)
    y = np.sin(X[:, 0]) + np.log(1 + np.abs(X[:, 1])) + X[:, 2] * X[:, 3] + np.random.randn(N) * SIGMA
    return X, y

def dgp_highdim():
    beta = np.zeros(P_HIGHDIM)
    beta[:S_HIGHDIM] = [2, -1.5, 0.8, 0.5, -0.3]
    X = np.random.randn(N, P_HIGHDIM)
    y = X @ beta + np.random.randn(N) * SIGMA
    return X, y

METHODS_DISPLAY = ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn_original', 'dnn_tuned',
                   'cf_rf', 'cf_xgboost', 'cf_lightgbm', 'cf_dnn']

if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    print(f'Starting DNN tuning simulation ({N_REPS} reps per scenario)...')
    print(f'PyTorch: {torch.__version__}')

    scenarios = {
        'linear': dgp_linear,
        'semiparametric': dgp_semiparametric,
        'nonlinear': dgp_nonlinear,
        'highdim': dgp_highdim,
    }

    all_results = {}
    for name, dgp_func in scenarios.items():
        result = run_scenario(name, dgp_func)
        all_results[name] = result

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {OUTPUT_FILE}')

    # Print summary table
    print('\n\n' + '=' * 100)
    header = f"{'Method':<15}"
    for sc in scenarios:
        header += f'{sc:>20}'
    print(header)
    print('-' * 100)
    for m in METHODS_DISPLAY:
        row = f'{m:<15}'
        for sc in scenarios:
            if sc in all_results and m in all_results[sc]:
                r = all_results[sc][m]
                row += f'{r["mse_mean"]:.3f}±{r["mse_sd"]:.3f}'.rjust(20)
            else:
                row += f'{"N/A":>20}'
        print(row)
    print('=' * 100)
