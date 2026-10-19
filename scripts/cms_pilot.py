#!/usr/bin/env python3
"""
Conformalized Model Selection — quick pilot.

Construct S(x) = {candidate optimal models at x} with
P(m*(x) ∈ S(x)) ≥ 1-α, where m*(x) = argmin_m μ_m(x).

Tests whether U_m (conformalized regret bound) can reliably
identify which model is conditionally optimal at each point.
"""
import os, sys, warnings
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn, torch.optim as optim
warnings.filterwarnings('ignore')

BASE_SEED = 2024
N_REPS = 30
N = 500
P_LINEAR = 10; P_HIGHDIM = 100; S_HIGHDIM = 5
SIGMA = 1.0; CP_ALPHA = 0.10
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ── DGP signals ──
def sf_linear(X):
    b = np.array([2, -1.5, 0.8, 0.5, -0.3, 0, 0, 0, 0, 0])
    return X @ b
def sf_semi(X):
    return sf_linear(X) + 0.3 * np.sin(X[:, 0])
def sf_nonlin(X):
    return np.sin(X[:,0]) + np.log(1+np.abs(X[:,1])) + X[:,2]*X[:,3]
def sf_highdim(X):
    b = np.zeros(100); b[:5] = [2, -1.5, 0.8, 0.5, -0.3]
    return X @ b

def dgp(name, fn):
    def f():
        p = 100 if name == 'highdim' else 10
        X = np.random.randn(N, p)
        y = fn(X) + np.random.randn(N) * SIGMA
        return X, y
    return f

# ── DNN ──
class DNN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self, x): return self.net(x).squeeze(-1)

def tr_dnn(m, Xt, yt, Xv, yv):
    l = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(Xt),torch.FloatTensor(yt)),
        batch_size=64, shuffle=True)
    m=m.to(DEVICE); o=optim.Adam(m.parameters(),lr=1e-3)
    bL, bS, w = float('inf'), None, 0
    Xvt,yvt=torch.FloatTensor(Xv).to(DEVICE),torch.FloatTensor(yv).to(DEVICE)
    for _ in range(200):
        m.train()
        for Xb,yb in l:
            Xb,yb=Xb.to(DEVICE),yb.to(DEVICE); o.zero_grad()
            nn.MSELoss()(m(Xb),yb).backward(); o.step()
        m.eval()
        with torch.no_grad():
            L = nn.MSELoss()(m(Xvt),yvt).item()
        if L < bL - 1e-5: bL, bS, w = L, {k:v.cpu().clone() for k,v in m.state_dict().items()}, 0
        else: w += 1
        if w >= 20: break
    if bS: m.load_state_dict(bS)
    return m

def pdnn(m, X):
    m.eval()
    with torch.no_grad(): return m(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()

def cp_q(r, a=CP_ALPHA):
    n=len(r); lvl=min(np.ceil((n+1)*(1-a))/n,1.0)
    return np.quantile(r,lvl,method='higher')

def cond_mse(f0, yh):
    return (f0-yh)**2 + SIGMA**2

# ── One replication ──
def one_rep(seed, signal_fn, dgp_fn, sc_name):
    np.random.seed(seed); torch.manual_seed(seed)
    X, y = dgp_fn()
    n = len(y)
    nb = int(n*0.50); nmf = int(n*0.15); nmc = int(n*0.15)
    idx = np.random.RandomState(seed).permutation(n)
    Xs, ys = X[idx], y[idx]
    Xb, yb = Xs[:nb], ys[:nb]
    Xmf, ymf = Xs[nb:nb+nmf], ys[nb:nb+nmf]
    Xmc, ymc = Xs[nb+nmf:nb+nmf+nmc], ys[nb+nmf:nb+nmf+nmc]
    Xt, yt = Xs[nb+nmf+nmc:], ys[nb+nmf+nmc:]

    ols = LinearRegression().fit(Xb, yb)
    rf = RFR(n_estimators=200,max_depth=10,min_samples_leaf=5,random_state=seed).fit(Xb, yb)
    xg = xgb.XGBRegressor(n_estimators=200,max_depth=6,learning_rate=0.1,random_state=seed,verbosity=0).fit(Xb, yb)
    ss = StandardScaler()
    Xbs=ss.fit_transform(Xb); Xmfs=ss.transform(Xmf); Xmcs=ss.transform(Xmc); Xts=ss.transform(Xt)
    Xtr,Xva,ytr,yva = train_test_split(Xbs,yb,test_size=0.2,random_state=seed)
    dn = DNN(Xtr.shape[1]); dn = tr_dnn(dn, Xtr, ytr, Xva, yva)

    mod = {'ols':ols,'rf':rf,'xgb':xg,'dnn':dn}
    def prd(m, Xr, Xs):
        if m=='dnn': return pdnn(dn, Xs)
        return mod[m].predict(Xr)
    bn = ['ols','rf','xgb','dnn']

    pr = {}
    for sn,Xr,Xs in [('b',Xb,Xbs),('mf',Xmf,Xmfs),('mc',Xmc,Xmcs),('t',Xt,Xts)]:
        pr[sn] = {m:prd(m,Xr,Xs) for m in bn}

    # Regret model
    ell = {}
    for m in bn: ell[m] = (ymf - pr['mf'][m])**2
    em = np.min(list(ell.values()),0)
    R = {}
    for m in bn: R[m] = ell[m] - em

    rh = {}; ah = {}
    for m in bn:
        rfr = RFR(n_estimators=100,max_depth=5,min_samples_leaf=10,oob_score=True,random_state=seed).fit(Xmf, R[m])
        rh[m] = rfr
        oof = rfr.oob_prediction_
        rfr2 = RFR(n_estimators=100,max_depth=5,min_samples_leaf=10,random_state=seed+1).fit(Xmf, np.abs(R[m]-oof))
        ah[m] = rfr2

    # Joint CP
    eps=1e-6; scs=[]
    ell_mc = {}
    for m in bn: ell_mc[m] = (ymc - pr['mc'][m])**2
    em_mc = np.min(list(ell_mc.values()),0)
    for i in range(len(ymc)):
        si=[]
        for m in bn:
            rv=rh[m].predict(Xmc[i:i+1])[0]; av=max(ah[m].predict(Xmc[i:i+1])[0],eps)
            si.append((ell_mc[m][i]-em_mc[i]-rv)/av)
        scs.append(max(si))
    qr = cp_q(np.array(scs))

    def get_U(Xs):
        U={}
        for m in bn:
            rv=rh[m].predict(Xs); av=np.maximum(ah[m].predict(Xs),eps)
            U[m]=np.maximum(0, rv+qr*av)
        return U

    U_t = get_U(Xt); U_mc = get_U(Xmc)

    # True conditional MSE
    f0t = signal_fn(Xt); f0mc = signal_fn(Xmc)
    mu_t = {}; mu_mc = {}
    for m in bn:
        mu_t[m]=cond_mse(f0t, pr['t'][m])
        mu_mc[m]=cond_mse(f0mc, pr['mc'][m])

    # Best model per point
    bm_t = np.argmin(np.column_stack([mu_t[m] for m in bn]),1)
    bm_mc = np.argmin(np.column_stack([mu_mc[m] for m in bn]),1)

    # Rank by U
    Ua_t = np.column_stack([U_t[m] for m in bn])
    Uk_t = np.argsort(np.argsort(Ua_t,1),1)  # rank 0,1,2
    Ua_mc = np.column_stack([U_mc[m] for m in bn])
    Uk_mc = np.argsort(np.argsort(Ua_mc,1),1)

    # Top-k coverage
    res = {}
    for k in [1,2,3]:
        cov_mc = np.mean([Uk_mc[i, bm_mc[i]] < k for i in range(len(bm_mc))])
        cov_t = np.mean([Uk_t[i, bm_t[i]] < k for i in range(len(bm_t))])
        res[f'top{k}_mc'] = float(cov_mc)
        res[f'top{k}_test'] = float(cov_t)

    # Adaptive threshold: include all models with U <= pct * min(U)
    for pct in [1.1, 1.25, 1.5, 2.0]:
        sizes = []; covs = []
        for i in range(len(bm_t)):
            th = pct * np.min(Ua_t[i])
            in_s = Ua_t[i] <= th
            sizes.append(np.sum(in_s))
            covs.append(bm_t[i] in np.where(in_s)[0])
        res[f'pct{pct}_cov'] = float(np.mean(covs))
        res[f'pct{pct}_sz'] = float(np.mean(sizes))

    # Top-1 U = best by μ (how often does U pick the right model?)
    res['u_best_match'] = float(np.mean(np.argmin(Ua_t,1) == bm_t))
    res['cond_mse_oracle'] = float(np.mean(np.min(np.column_stack([mu_t[m] for m in bn]),1)))

    return res


if __name__ == '__main__':
    print('='*60)
    print('  Conformalized Model Selection — Quick Pilot')
    print(f'  Reps: {N_REPS}, n={N}')
    print('='*60)

    scenarios = [
        ('linear', sf_linear),
        ('nonlinear', sf_nonlin),
    ]

    for sc_name, sfn in scenarios:
        dg = dgp(sc_name, sfn)
        reps = [one_rep(BASE_SEED+r, sfn, dg, sc_name) for r in range(N_REPS)]

        print(f'\n── {sc_name} ──')
        print(f'  {"Metric":<30s} {"Value":>8s}')
        print(f'  {"-"*40}')

        # Top-k coverage (test)
        for k in [1,2,3]:
            v = np.mean([r[f'top{k}_test'] for r in reps])
            print(f'  Top-{k} coverage (test):              {v:>7.4f}')

        # Calibration coverage (MC) to compare
        for k in [1,2,3]:
            v = np.mean([r[f'top{k}_mc'] for r in reps])
            print(f'  Top-{k} coverage (calib):             {v:>7.4f}')

        # U best match rate
        v = np.mean([r['u_best_match'] for r in reps])
        print(f'  U picks μ-best model:                {v:>7.4f}')

        # Adaptive threshold
        for pct in [1.1, 1.25, 1.5, 2.0]:
            cv = np.mean([r[f'pct{pct}_cov'] for r in reps])
            sz = np.mean([r[f'pct{pct}_sz'] for r in reps])
            print(f'  pct={pct:.2f}: coverage={cv:.4f}, avg size={sz:.3f}')
