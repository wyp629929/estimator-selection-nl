#!/usr/bin/env python3
"""Download real datasets for DNN tuning experiment."""
import os, sys, urllib.request, csv, io, warnings
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 1. PIMA Indians Diabetes
print('Downloading PIMA Diabetes...')
url_pima = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
try:
    df_pima = pd.read_csv(url_pima, header=None)
    df_pima.columns = ['preg', 'gluc', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'outcome']
    print(f'  {df_pima.shape[0]} rows, {df_pima.shape[1]} cols')
    print(f'  Columns: {list(df_pima.columns)}')
    print(f'  Outcome distribution: {df_pima["outcome"].value_counts().to_dict()}')
    df_pima.to_csv(os.path.join(DATA_DIR, 'pima_diabetes.csv'), index=False)
    print(f'  Saved to {DATA_DIR}/pima_diabetes.csv')
except Exception as e:
    print(f'  Failed: {e}')

# 2. Home Credit Default Risk (via kagglehub)
print('\nDownloading Home Credit Default Risk...')
try:
    import kagglehub
    # Download competition data
    path = kagglehub.competition_download("home-credit-default-risk")
    print(f'  Download path: {path}')

    # Use application_train.csv (the main file)
    app_train_path = os.path.join(path, 'application_train.csv')
    if os.path.exists(app_train_path):
        # Read only first 104 numeric columns and target
        df_hc = pd.read_csv(app_train_path)
        print(f'  Raw: {df_hc.shape[0]} rows, {df_hc.shape[1]} cols')

        # Select numeric features + target
        target = df_hc['TARGET']
        numeric_cols = df_hc.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('TARGET')
        print(f'  Numeric features: {len(numeric_cols)}')

        # Save a processed version
        df_out = df_hc[['SK_ID_CURR', 'TARGET'] + numeric_cols[:104]]
        df_out.to_csv(os.path.join(DATA_DIR, 'home_credit_sample.csv'), index=False)
        print(f'  Saved {df_out.shape[0]}x{df_out.shape[1]} to {DATA_DIR}/home_credit_sample.csv')
    else:
        print(f'  application_train.csv not found in {path}')
        # List files
        for f in os.listdir(path):
            print(f'    - {f}')
except Exception as e:
    print(f'  Failed: {e}')

print('\nDone.')
