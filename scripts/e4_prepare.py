#!/usr/bin/env python3
"""E4 data preparation (run locally; needs pandas, the raw CSVs in data/, and internet access
for the OpenML datasets, which are cached in data/openml_cache/).

Writes data/e4/<name>.npz with X (float32; missing numeric values kept as NaN and imputed later
with training medians inside each split), y (float64) and feature names. Existing files are
left untouched unless --overwrite is given, so the splits and fitted models of datasets that
have already been run do not change.

  pima        PIMA Indians Diabetes, 768 x 8, y = diabetes (0/1)
  supercond   UCI Superconductivity, 21,263 x 81, y = critical temperature
  homecredit  Home Credit Default Risk, random 50,000 of 307,511 applications (seed 2024),
              numeric columns, y = default (0/1)
  The datasets below were added on 2026-10-15 by one rule fixed before running: standard
  tabular binary benchmarks on OpenML spanning a wide range of prevalence. y = 1 is the
  minority class. Categorical features are one-hot encoded with missing values as a category.
  creditcard  OpenML 1597, random 50,000 of 284,807 transactions (seed 2024), y = fraud
  mammography OpenML 310, 11,183 x 6, y = malignant (class 1)
  bank        OpenML 1461 (bank-marketing), 45,211 x 16, y = subscribed (Class 2)
  default     OpenML 42477 (default-of-credit-card-clients), 30,000 x 23, y = default (y 1)
  adult       OpenML 1590 v2, 48,842 x 14, y = income > 50K
  The datasets in CTR23 below were added on 2026-10-18 for E7 (Appendix D of the manuscript) by one
  rule fixed before any data were downloaded: every dataset of OpenML-CTR23 (OpenML study 353, a
  curated regression benchmark suite) with at least 4,000 rows, except superconductivity (already
  used as supercond). y = the CTR23 task target, untransformed. More than 50,000 rows: random
  50,000 (seed 2024). Categorical features one-hot encoded with missing values as a category.
  The datasets in GRINSZTAJN below were added on 2026-10-21 for E8 (Section 4.5 and Appendix D of the manuscript) by one
  rule fixed before any data were downloaded: every dataset of the tabular benchmark of Grinsztajn,
  Oyallon and Varoquaux (2022) with at least 4,000 rows, one version per source dataset (numerical and
  categorical version when both exist; the regression version for house_16H and pol), excluding source
  datasets already used in this paper. The CSVs are read from data/e8_raw/ (downloaded from
  https://huggingface.co/datasets/inria-soda/tabular-benchmark; checksums in data/e8_raw/SHA256SUMS)
  and used as distributed: the benchmark balanced the two classes of each classification task and
  log-transformed heavy-tailed regression targets. The target is the last column. For classification,
  y = 1 for the larger of the two class labels. Non-numeric columns are one-hot encoded with missing
  values as a category. More than 50,000 rows: random 50,000 (seed 2024).
"""
import argparse
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, '..', 'data')
OUT = os.path.join(DATA, 'e4')
CACHE = os.path.join(DATA, 'openml_cache')
MAX_ROWS = 50000
SEED = 2024


def save(name, X, y, cols, note, overwrite):
    path = os.path.join(OUT, f'{name}.npz')
    if os.path.exists(path) and not overwrite:
        print(f'{name}: {path} exists, left untouched')
        return
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.float64), features=np.array(cols), note=np.array(note))
    print(f'{name}: X {X.shape}, missing {np.isnan(X).mean():.3%}, prevalence/mean y {y.mean():.4f} -> {path}')


def openml(name, version, positive, n_max=None):
    from sklearn.datasets import fetch_openml
    b = fetch_openml(name=name, version=version, as_frame=True, data_home=CACHE, parser='auto')
    df, t = b.data.copy(), b.target.astype(str)
    if n_max and len(df) > n_max:
        idx = df.sample(n=n_max, random_state=SEED).index
        df, t = df.loc[idx], t.loc[idx]
    cat = [c for c in df.columns if str(df[c].dtype) in ('category', 'object')]
    X = pd.get_dummies(df, columns=cat, dummy_na=True, dtype=float) if cat else df.astype(float)
    y = (t == positive).to_numpy(float)
    note = f'OpenML {b.details["id"]} v{b.details["version"]} ({name}); y = 1 if target == {positive!r}'
    return X.to_numpy(float), y, list(X.columns), note


# E7: name -> (OpenML data id, CTR23 task target)
CTR23 = {
    'abalone': (44956, 'rings'), 'white_wine': (44971, 'quality'), 'cpu_activity': (44978, 'usr'),
    'kin8nm': (44980, 'y'), 'pumadyn32nh': (44981, 'thetadd6'), 'grid_stability': (44973, 'stab'),
    'brazilian_houses': (44990, 'total'), 'naval_propulsion_plant': (44969, 'gt_compressor_decay_state_coefficient'),
    'miami_housing': (44983, 'SALE_PRC'), 'fifa': (45012, 'wage_eur'), 'california_housing': (44977, 'medianHouseValue'),
    'kings_county': (44989, 'price'), 'health_insurance': (44993, 'whrswk'), 'fps_benchmark': (44992, 'FPS'),
    'cps88wages': (44984, 'wage'), 'physiochemical_protein': (44963, 'RMSD'), 'sarcos': (44976, 'V22'),
    'diamonds': (44979, 'price'), 'video_transcoding': (44974, 'utime'), 'wave_energy': (44975, 'energy_total'),
}


def openml_regression(data_id, target, n_max=MAX_ROWS):
    from sklearn.datasets import fetch_openml
    b = fetch_openml(data_id=data_id, target_column=target, as_frame=True, data_home=CACHE, parser='auto')
    df, t = b.data.copy(), b.target.astype(float)
    n_all = len(df)
    if n_max and len(df) > n_max:
        idx = df.sample(n=n_max, random_state=SEED).index
        df, t = df.loc[idx], t.loc[idx]
    cat = [c for c in df.columns if str(df[c].dtype) in ('category', 'object', 'string', 'bool')]
    X = pd.get_dummies(df, columns=cat, dummy_na=True, dtype=float) if cat else df.astype(float)
    note = (f'OpenML {b.details["id"]} v{b.details["version"]} ({b.details["name"]}), CTR23; y = {target}; '
            f'{len(df)} of {n_all} rows' + (' (random, seed 2024)' if len(df) < n_all else ''))
    return X.to_numpy(float), t.to_numpy(float), list(X.columns), note


# E8: name -> path in data/e8_raw/ (Grinsztajn et al. 2022 tabular benchmark)
GRINSZTAJN = {
    'albert': 'clf_cat/albert.csv', 'compas': 'clf_cat/compas-two-years.csv', 'covertype': 'clf_cat/covertype.csv',
    'electricity': 'clf_cat/electricity.csv', 'eye_movements': 'clf_cat/eye_movements.csv',
    'road_safety': 'clf_cat/road-safety.csv', 'diabetes130us': 'clf_num/Diabetes130US.csv', 'higgs': 'clf_num/Higgs.csv',
    'magic_telescope': 'clf_num/MagicTelescope.csv', 'miniboone': 'clf_num/MiniBooNE.csv', 'credit': 'clf_num/credit.csv',
    'heloc': 'clf_num/heloc.csv', 'jannis': 'clf_num/jannis.csv',
    'airlines_depdelay': 'reg_cat/Airlines_DepDelay_1M.csv', 'allstate_claims': 'reg_cat/Allstate_Claims_Severity.csv',
    'bike_sharing': 'reg_cat/Bike_Sharing_Demand.csv', 'mercedes_benz': 'reg_cat/Mercedes_Benz_Greener_Manufacturing.csv',
    'sgemm_gpu': 'reg_cat/SGEMM_GPU_kernel_performance.csv', 'analcatdata_supreme': 'reg_cat/analcatdata_supreme.csv',
    'delays_zurich': 'reg_cat/delays_zurich_transport.csv', 'medical_charges': 'reg_cat/medical_charges.csv',
    'nyc_taxi_green': 'reg_cat/nyc-taxi-green-dec-2016.csv', 'particulate_matter': 'reg_cat/particulate-matter-ukair-2017.csv',
    'seattlecrime6': 'reg_cat/seattlecrime6.csv', 'topo_2_1': 'reg_cat/topo_2_1.csv',
    'visualizing_soil': 'reg_cat/visualizing_soil.csv', 'ailerons': 'reg_num/Ailerons.csv',
    'elevators': 'reg_num/elevators.csv', 'house_16h': 'reg_num/house_16H.csv', 'pol': 'reg_num/pol.csv',
    'sulfur': 'reg_num/sulfur.csv',
}   # yprop_4_1 dropped: same rows and target as topo_2_1 (one version per source dataset)


def grinsztajn(path, n_max=MAX_ROWS):
    df = pd.read_csv(os.path.join(DATA, 'e8_raw', path))
    n_all = len(df)
    if n_max and len(df) > n_max:
        df = df.sample(n=n_max, random_state=SEED)
    target = df.columns[-1]
    t, df = df[target], df.drop(columns=[target])
    if path.startswith('clf'):
        labels = sorted(t.unique())
        assert len(labels) == 2, (path, labels)
        y = (t == labels[1]).to_numpy(float)
        ydesc = f'y = 1 if {target} == {labels[1]!r}'
    else:
        y = t.to_numpy(float)
        ydesc = f'y = {target}'
    cat = [c for c in df.columns if str(df[c].dtype) in ('category', 'object', 'string', 'bool')]
    X = pd.get_dummies(df, columns=cat, dummy_na=True, dtype=float) if cat else df.astype(float)
    note = (f'Grinsztajn et al. (2022) tabular benchmark, {path}; {ydesc}; {len(df)} of {n_all} rows'
            + (' (random, seed 2024)' if len(df) < n_all else ''))
    return X.to_numpy(float), y, list(X.columns), note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    ow = args.overwrite

    df = pd.read_csv(os.path.join(DATA, 'pima_diabetes.csv'))
    save('pima', df.drop(columns=['outcome']).to_numpy(float), df['outcome'].to_numpy(float),
         list(df.columns.drop('outcome')), 'PIMA diabetes; y = outcome (0/1)', ow)

    if ow or not os.path.exists(os.path.join(OUT, 'homecredit.npz')):
        df = pd.read_csv(os.path.join(DATA, 'application_train.csv')).sample(n=MAX_ROWS, random_state=SEED)
        num = df.drop(columns=['TARGET', 'SK_ID_CURR']).select_dtypes(include=[np.number])
        save('homecredit', num.to_numpy(float), df['TARGET'].to_numpy(float), list(num.columns),
             'Home Credit; random 50,000 of 307,511 applications (seed 2024); numeric columns; y = TARGET (0/1)', ow)
    else:
        save('homecredit', None, None, None, None, ow)

    df = pd.read_csv(os.path.join(DATA, 'superconductivity', 'train.csv'))
    save('supercond', df.drop(columns=['critical_temp']).to_numpy(float), df['critical_temp'].to_numpy(float),
         list(df.columns.drop('critical_temp')), 'UCI Superconductivity; y = critical temperature (K)', ow)

    for name, (oname, ver, pos, nmax) in {
        'creditcard': ('creditcard', 1, '1', MAX_ROWS),
        'mammography': ('mammography', 1, '1', None),
        'bank': ('bank-marketing', 1, '2', None),
        'default': ('default-of-credit-card-clients', 1, '1', None),
        'adult': ('adult', 2, '>50K', None),
    }.items():
        if os.path.exists(os.path.join(OUT, f'{name}.npz')) and not ow:
            save(name, None, None, None, None, ow)
            continue
        save(name, *openml(oname, ver, pos, nmax), ow)

    for name, (did, target) in CTR23.items():
        if os.path.exists(os.path.join(OUT, f'{name}.npz')) and not ow:
            save(name, None, None, None, None, ow)
            continue
        save(name, *openml_regression(did, target), ow)

    for name, path in GRINSZTAJN.items():
        if os.path.exists(os.path.join(OUT, f'{name}.npz')) and not ow:
            save(name, None, None, None, None, ow)
            continue
        save(name, *grinsztajn(path), ow)


if __name__ == '__main__':
    main()
