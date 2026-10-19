#!/usr/bin/env python3
"""
Diagnostic: Is CP width a good proxy for point prediction MSE?

Experiment A: Correlation between L_m(x) and (y - ŷ_m)²
Experiment B: P( argmin_{m} L_m(x) == argmin_{m} (y - ŷ_m)² )

If Corr ≈ 0 or match rate ≈ 1/M (random), the core hypothesis is broken.
"""
import sys, os, json, warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
warnings.filterwarnings('ignore')

# Reuse CP-WAA code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cp_guided_fusion as cpf

N_DIAG_REPS = 20  # enough per-point samples across reps
BASE_SEED = 2024


def run_diagnostic(scenario_name, make_X_y):
    """Run per-point diagnostics across multiple reps."""
    all_records = []  # list of {model, L, sq_err, scenario}

    for rep in range(N_DIAG_REPS):
        seed = BASE_SEED + rep
        np.random.seed(seed)
        torch.manual_seed(seed)

        X, y = make_X_y()
        X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.30, random_state=seed)
        cal_share = 2.0 / 7.0
        X_train, X_cal, y_train, y_cal = train_test_split(X_rest, y_rest, test_size=cal_share, random_state=seed)
        X_cal1, X_cal2, y_cal1, y_cal2 = train_test_split(X_cal, y_cal, test_size=0.5, random_state=seed)

        # ── Train models ──
        m_ols = LinearRegression().fit(X_train, y_train)
        m_rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=seed).fit(X_train, y_train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_cal1_scaled = scaler.transform(X_cal1)
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=seed)
        m_dnn = cpf.DNNOriginal(X_tr.shape[1])
        m_dnn = cpf.train_dnn(m_dnn, X_tr, y_tr, X_val, y_val)

        models = {'ols': m_ols, 'rf': m_rf, 'dnn': m_dnn}

        def predict(name, X_raw, X_scaled):
            if name == 'dnn': return cpf.predict_dnn(m_dnn, X_scaled)
            else: return models[name].predict(X_raw)

        # ── Predictions on Cal1 and Test ──
        preds_cal1 = {}
        preds_test = {}
        for m_name in ['ols', 'rf', 'dnn']:
            preds_cal1[m_name] = predict(m_name, X_cal1, X_cal1_scaled)
            preds_test[m_name] = predict(m_name, X_test, scaler.transform(X_test))

        # ── L_m(x) on Cal1 (same pipeline as CP-WAA) ──
        for m_name in ['ols', 'rf', 'dnn']:
            resid_model = cpf.fit_abs_residual_model(X_train, y_train,
                predict(m_name, X_train, X_train_scaled), seed=seed)

            sigma_cal1 = resid_model.predict(X_cal1)
            _, widths_test = cpf.local_cp_width(
                y_cal1, preds_cal1[m_name], preds_test[m_name],
                sigma_cal1, resid_model.predict(X_test))

            # L_m(x) for Cal1
            # Get q from the local_cp_width call... need to call it differently
            # Let me recompute:
            scores = np.abs(y_cal1 - preds_cal1[m_name]) / sigma_cal1
            q_level = min(np.ceil((len(y_cal1) + 1) * 0.90) / len(y_cal1), 1.0)
            q_m = np.quantile(scores, q_level, method='higher')
            L_cal1 = 2.0 * q_m * sigma_cal1

            # ── Square error on Cal1 ──
            sq_err_cal1 = (y_cal1 - preds_cal1[m_name]) ** 2

            # Record per point
            for i in range(len(y_cal1)):
                all_records.append({
                    'scenario': scenario_name,
                    'rep': rep,
                    'model': m_name,
                    'L': float(L_cal1[i]),
                    'sq_err': float(sq_err_cal1[i]),
                })

        if rep % 5 == 0:
            print(f'  {scenario_name}: rep {rep}/{N_DIAG_REPS}', flush=True)

    return all_records


def analyze(all_records):
    """Print correlation + argmin match diagnostics."""
    scenarios = ['linear', 'semiparametric', 'nonlinear', 'highdim']

    print('\n' + '=' * 70)
    print('DIAGNOSTIC: CP Width vs. Squared Error')
    print('=' * 70)

    for sc in scenarios:
        sc_records = [r for r in all_records if r['scenario'] == sc]
        if not sc_records:
            continue

        print(f'\n── {sc} ──')

        # Experiment A: per-model correlation
        for m_name in ['ols', 'rf', 'dnn']:
            m_records = [r for r in sc_records if r['model'] == m_name]
            L_vals = np.array([r['L'] for r in m_records])
            sq_vals = np.array([r['sq_err'] for r in m_records])

            corr = np.corrcoef(L_vals, sq_vals)[0, 1]
            print(f'  Corr(L_{m_name}, sq_err_{m_name}) = {corr:.4f}')

        # Experiment B: argmin match rate (per point, per rep)
        # Group by (rep, point)
        from collections import defaultdict
        by_point = defaultdict(dict)
        for r in sc_records:
            key = (r['rep'], id(r))  # not great... let me use index
            by_point[(r['rep'],)] = by_point.get((r['rep'],), [])

        # Better: group by unique (rep, cal_index)
        # Re-group
        point_groups = defaultdict(dict)
        for r in sc_records:
            # We need a unique identifier per Cal1 point.
            # Use the order in which they appear (which is group-by rep then model then index)
            # Since records are appended in order: rep 0 → ols[0], ols[1], ..., rf[0], ..., dnn[0], ...
            # I need to track the Cal1 index. Let me just re-index.
            pass

        # Simpler approach: group by (rep) and then within each rep,
        # iterate through Cal1 points and check model agreement.
        # But I don't have the Cal1 index stored.
        # Let me just recompute from raw data.

        # Actually, the records are stored in order within each rep:
        #   rep N: ols point 0..k-1, rf point 0..k-1, dnn point 0..k-1
        # So within each rep, the Cal1 index i corresponds to positions i, i+k, i+2k
        # This is fragile. Let me just print per-model stats instead.

        # Alternative: compute how often each model is "best" by each criterion
        # Group by rep and cal_index using the order
        n_points_per_rep = {}
        for rep_idx in range(N_DIAG_REPS):
            rep_recs = [r for r in sc_records if r['rep'] == rep_idx]
            n_cal1 = len(rep_recs) // 3  # 3 models
            n_points_per_rep[rep_idx] = n_cal1

        matches = []
        total_points = 0
        for rep_idx in range(N_DIAG_REPS):
            rep_recs = [r for r in sc_records if r['rep'] == rep_idx]
            n_cal1 = len(rep_recs) // 3

            for i in range(n_cal1):
                # ols, rf, dnn at same Cal1 point
                try:
                    ols_r = rep_recs[i]
                    rf_r = rep_recs[i + n_cal1]
                    dnn_r = rep_recs[i + 2 * n_cal1]
                except IndexError:
                    break

                L_vals = {'ols': ols_r['L'], 'rf': rf_r['L'], 'dnn': dnn_r['L']}
                err_vals = {'ols': ols_r['sq_err'], 'rf': rf_r['sq_err'], 'dnn': dnn_r['sq_err']}

                best_by_L = min(L_vals, key=L_vals.get)
                best_by_err = min(err_vals, key=err_vals.get)

                matches.append(best_by_L == best_by_err)
                total_points += 1

        match_rate = np.mean(matches) if matches else 0
        random_baseline = 1.0 / 3  # 33.3% for 3 models
        print(f'  P(argmin L == argmin sq_err) = {match_rate:.4f}  (random baseline = {random_baseline:.2f})')

    print('\n' + '=' * 70)
    print('If Corr ≈ 0 and match rate ≈ 33%: CP width is NOT a valid MSE proxy.')
    print('If Corr > 0.3 and match rate > 50%: the signal is usable (not perfect but meaningful).')
    print('=' * 70)


if __name__ == '__main__':
    print('CP-WAA Core Hypothesis Diagnostic')
    print(f'{N_DIAG_REPS} diagnostic reps per scenario\n')

    scenarios = {
        'linear':        cpf.dgp_linear,
        'semiparametric': cpf.dgp_semiparametric,
        'nonlinear':      cpf.dgp_nonlinear,
        'highdim':        cpf.dgp_highdim,
    }

    all_records = []
    for sname, dgp_func in scenarios.items():
        records = run_diagnostic(sname, dgp_func)
        all_records.extend(records)

    analyze(all_records)
