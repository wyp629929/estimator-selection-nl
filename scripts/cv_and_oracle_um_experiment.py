#!/usr/bin/env python3
"""Fast CV-MSE + oracle U_M experiment (20 reps, 3 models in oracle)."""
import os, time, json, warnings
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb; import lightgbm as lgb
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')
BASE_SEED = 2024; N_REPS = 20; EPS = 1e-8
ALL_MODELS = ['ols','ridge','lasso','rf','xgboost','lightgbm','dnn']

def sig(X): return np.sin(X[:,0]) + np.log(1+np.abs(X[:,1])) + X[:,2]*X[:,3]
def cq(r,a=0.10):
    n=len(r); l=min(np.ceil((n+1)*(1-a))/n,1.0)
    return np.quantile(r,l,method='higher')

def train(m,X,y,s=0):
    kw = {'random_state':s,'verbosity':0,'verbose':-1}
    if m=='ols': return LinearRegression().fit(X,y)
    if m=='ridge': return Ridge(alpha=1.0).fit(X,y)
    if m=='lasso': return Lasso(alpha=0.01,max_iter=5000).fit(X,y)
    if m=='rf': return RandomForestRegressor(n_estimators=200,max_depth=10,min_samples_leaf=5,random_state=s,n_jobs=2).fit(X,y)
    if m=='xgboost': return xgb.XGBRegressor(n_estimators=200,max_depth=6,learning_rate=0.1,**kw).fit(X,y)
    if m=='lightgbm': return lgb.LGBMRegressor(n_estimators=200,max_depth=6,learning_rate=0.1,**kw).fit(X,y)
    if m=='dnn': return MLPRegressor(hidden_layer_sizes=(128,64,32),max_iter=200,random_state=s).fit(X,y)

cv_ctors = {
    'ols':lambda:LinearRegression(), 'ridge':lambda:Ridge(alpha=1.0),
    'lasso':lambda:Lasso(alpha=0.01,max_iter=5000), 'rf':lambda:RandomForestRegressor(n_estimators=200,max_depth=10,min_samples_leaf=5,n_jobs=2),
    'xgboost':lambda:xgb.XGBRegressor(n_estimators=200,max_depth=6,learning_rate=0.1), 'lightgbm':lambda:lgb.LGBMRegressor(n_estimators=200,max_depth=6,learning_rate=0.1),
}

results = []; x0 = time.time()
for rep in range(N_REPS):
    rng = np.random.RandomState(BASE_SEED+rep)
    n,p=500,10; X=rng.randn(n,p); y=sig(X)+rng.randn(n)
    ts=int(n*0.5); X_tr,X_te=X[:ts],X[-int(n*0.15):]; y_tr=y[:ts]
    f_te=sig(X_te)

    # Conditional MSE via minimal oracle (3 models: OLS, RF, DNN)
    mu = {}
    for m in ['ols','rf']:
        fh=np.zeros((5,X_te.shape[0]))
        for o in range(5):
            yo=sig(X_tr)+rng.randn(ts)
            moi=train(m,X_tr,yo,rep*100+o)
            fh[o]=moi.predict(X_te)
        b=np.mean(fh,0)-f_te; v=np.var(fh,0); mu[m]=b**2+v
    # DNN oracle (3 draws)
    fh=np.zeros((3,X_te.shape[0]))
    for o in range(3):
        yo=sig(X_tr)+rng.randn(ts)
        moi=train('dnn',X_tr,yo,rep*1000+o)
        fh[o]=moi.predict(X_te)
    b=np.mean(fh,0)-f_te; v=np.var(fh,0); mu['dnn']=b**2+v
    # Ridge/Lasso/XGB/LightGBM: approximate as equivalent to OLS/RF
    for m in ['ridge','lasso']: mu[m]=mu['ols'].copy()
    for m in ['xgboost','lightgbm']: mu[m]=mu['rf'].copy()

    # Oracle per-point
    opp=np.argmin(np.column_stack([mu[m] for m in ALL_MODELS]),1)
    global_oracle=ALL_MODELS[np.bincount(opp).argmax()]

    # CP width
    cs=int(ts*0.2); X_tr2,X_ca=X_tr[:-cs],X_tr[-cs:]; y_tr2,y_ca=y_tr[:-cs],y_tr[-cs:]
    cw={}
    for m in ALL_MODELS:
        m2=train(m,X_tr2,y_tr2,rep); q=cq(np.abs(y_ca-m2.predict(X_ca))); cw[m]=2*q
    cp_best=min(ALL_MODELS,key=lambda m:cw[m])

    # CV-MSE (fast models only)
    cv={}
    for m in ['ols','ridge','lasso','rf','xgboost','lightgbm']:
        sc=cross_val_score(cv_ctors[m](),X_tr,y_tr,cv=3,scoring='neg_mean_squared_error')
        cv[m]=-sc.mean()
    # DNN: use simple holdout
    vs=int(ts*0.2)
    dnn_mse=np.mean((train('dnn',X_tr[:-vs],y_tr[:-vs],rep).predict(X_tr[-vs:])-y_tr[-vs:])**2)
    cv['dnn']=dnn_mse
    cv_best=min(ALL_MODELS,key=lambda m:cv[m])

    # Oracle U_M (on test half)
    te_sz=int(n*0.15); nc=te_sz//2
    R_all=np.column_stack([mu[m] for m in ALL_MODELS])
    R_c,R_e=R_all[:nc],R_all[nc:]
    sc_i=np.max(R_c/(np.std(R_c,0,keepdims=True)+EPS),1)
    qu=cq(sc_i)
    Uo=np.maximum(0,R_e+qu*(np.std(R_e,0,keepdims=True)+EPS))
    ob=np.argmin(R_e,1); Ub=np.argmin(Uo,1); dU=np.mean(Ub!=ob)

    # Pairwise
    ne=len(Ub); pc=0; pt=0
    for i in range(ne):
        for a in range(7):
            for b in range(a+1,7):
                pt+=1
                if (Uo[i,a]<Uo[i,b])==(R_e[i,a]<R_e[i,b]): pc+=1
    pwU=pc/max(pt,1)

    results.append(dict(
        dc=int(cp_best!=global_oracle), dv=int(cv_best!=global_oracle), dU=dU, pwU=pwU))

    print(f'  rep {rep+1}/{N_REPS} [{time.time()-x0:.0f}s]',flush=True)

dc=np.mean([r['dc'] for r in results])
dv=np.mean([r['dv'] for r in results])
du=np.mean([r['dU'] for r in results])
se_c=np.std([r['dc'] for r in results])/N_REPS**0.5
se_v=np.std([r['dv'] for r in results])/N_REPS**0.5
se_u=np.std([r['dU'] for r in results])/N_REPS**0.5
pw=np.mean([r['pwU'] for r in results])

print(f'\n=== RESULTS (nonlinear, B={N_REPS}) ===')
print(f'CP width Δ     = {dc:.3f} (SE={se_c:.3f})')
print(f'CV-MSE Δ       = {dv:.3f} (SE={se_v:.3f})')
print(f'Oracle U_M Δ   = {du:.3f} (SE={se_u:.3f})')
print(f'Oracle U_M PR  = {pw:.3f}')
print(f'Random baseline = {6/7:.4f}')
print(f'Paper: CP=0.530, Est U=0.760')
print(f'Δ_cv-Δ_cp = {dv-dc:+.3f}')
print(f'Δ_Uo - 0.760 = {du-0.760:+.3f}')
print(f'Time: {time.time()-x0:.0f}s')

out={'regime':'nonlinear','n_reps':N_REPS,'delta_cp':dc,'delta_cp_se':se_c,
     'delta_cv':dv,'delta_cv_se':se_v,'delta_U_oracle':du,'delta_U_oracle_se':se_u,
     'pairwise_U_oracle':pw}
os.makedirs('/Users/wangyaoping/Desktop/ML_Inference_Paper/results',exist_ok=True)
with open('/Users/wangyaoping/Desktop/ML_Inference_Paper/results/cv_and_oracle_um_results.json','w') as f:
    json.dump(out,f,indent=2)
print('Saved')
