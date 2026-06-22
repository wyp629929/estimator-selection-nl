#!/usr/bin/env python3
"""Run threshold DGP: f(X) = 2*I(X1>0) + log(1+|X2|)*I(X3>0) - 1, p=10."""
import os, time, json, warnings
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb; import lightgbm as lgb
import torch, torch.nn as nn, torch.optim as optim
warnings.filterwarnings('ignore')

BASE_SEED=2024; N_REPS=500; N=500; P=10; SIGMA=1.0
DEVICE=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results','threshold_results.json')

class DNN(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

class DNT(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Dropout(0.3),nn.Linear(256,128),nn.ReLU(),nn.BatchNorm1d(128),nn.Dropout(0.3),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def train(m,Xt,yt,Xv,yv):
    l=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(Xt),torch.FloatTensor(yt)),64,shuffle=True)
    m=m.to(DEVICE);o=optim.Adam(m.parameters(),lr=1e-3);bL,bS,w=float('inf'),None,0
    Xvt,yvt=torch.FloatTensor(Xv).to(DEVICE),torch.FloatTensor(yv).to(DEVICE)
    for _ in range(200):
        m.train()
        for Xb,yb in l: Xb,yb=Xb.to(DEVICE),yb.to(DEVICE);o.zero_grad();nn.MSELoss()(m(Xb),yb).backward();o.step()
        m.eval()
        with torch.no_grad(): L=nn.MSELoss()(m(Xvt),yvt).item()
        if L<bL-1e-5: bL,bS,w=L,{k:v.cpu().clone() for k,v in m.state_dict().items()},0
        else: w+=1
        if w>=20: break
    if bS: m.load_state_dict(bS)
    return m

def pd(m,X): m.eval(); return m(torch.FloatTensor(X).to(DEVICE)).detach().cpu().numpy()

def dgp():
    X=np.random.randn(N,P)
    f=2.0*(X[:,0]>0)+np.log(1+np.abs(X[:,1]))*(X[:,2]>0)-1.0
    y=f+np.random.randn(N)*SIGMA
    return X,y

print(f'Threshold DGP, {N_REPS} reps, n={N}, {DEVICE}')
res={m:[] for m in['ols','ridge','lasso','rf','xgboost','lightgbm','dnn_original','dnn_tuned']}

for rep in range(N_REPS):
    seed=BASE_SEED+rep
    np.random.seed(seed); torch.manual_seed(seed)
    X,y=dgp()
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=seed)
    ss=StandardScaler();Xtrs=ss.fit_transform(Xtr);Xtes=ss.transform(Xte)
    Xtr2,Xva,ytr2,yva=train_test_split(Xtrs,ytr,test_size=0.2,random_state=seed)

    t0=time.time();m=LinearRegression().fit(Xtr,ytr);res['ols'].append({'mse':mean_squared_error(yte,m.predict(Xte)),'time':time.time()-t0})
    t0=time.time();m=Ridge(alpha=1.0).fit(Xtr,ytr);res['ridge'].append({'mse':mean_squared_error(yte,m.predict(Xte)),'time':time.time()-t0})
    t0=time.time();m=Lasso(alpha=0.01,max_iter=5000).fit(Xtr,ytr);res['lasso'].append({'mse':mean_squared_error(yte,m.predict(Xte)),'time':time.time()-t0})
    t0=time.time();m=RandomForestRegressor(n_estimators=200,max_depth=10,min_samples_leaf=5,random_state=seed).fit(Xtr,ytr);res['rf'].append({'mse':mean_squared_error(yte,m.predict(Xte)),'time':time.time()-t0})
    t0=time.time();m=xgb.XGBRegressor(n_estimators=200,max_depth=6,learning_rate=0.1,random_state=seed,verbosity=0).fit(Xtr,ytr);res['xgboost'].append({'mse':mean_squared_error(yte,m.predict(Xte)),'time':time.time()-t0})
    t0=time.time();m=lgb.LGBMRegressor(n_estimators=200,max_depth=6,learning_rate=0.1,verbose=-1,random_state=seed).fit(Xtr,ytr);res['lightgbm'].append({'mse':mean_squared_error(yte,m.predict(Xte)),'time':time.time()-t0})
    torch.manual_seed(seed);t0=time.time();dn=DNN(Xtr2.shape[1]);dn=train(dn,Xtr2,ytr2,Xva,yva);yp=pd(dn,Xtes);res['dnn_original'].append({'mse':mean_squared_error(yte,yp),'time':time.time()-t0})
    torch.manual_seed(seed+999);t0=time.time();dn2=DNT(Xtr2.shape[1]);dn2=train(dn2,Xtr2,ytr2,Xva,yva);yp2=pd(dn2,Xtes);res['dnn_tuned'].append({'mse':mean_squared_error(yte,yp2),'time':time.time()-t0})
    if rep%50==0: print(f'  rep {rep}/{N_REPS}',flush=True)

sm={}
for m,v in res.items():
    ms=[r['mse']for r in v];ts=[r['time']for r in v]
    sm[m]={'mse_mean':float(np.mean(ms)),'mse_sd':float(np.std(ms)),'time_mean':float(np.mean(ts))}
    print(f'  {m:15s} MSE={sm[m]["mse_mean"]:.4f}±{sm[m]["mse_sd"]:.4f}')

with open(OUT,'w') as f: json.dump({'threshold':sm},f,indent=2)
print(f'Saved to {OUT}')
