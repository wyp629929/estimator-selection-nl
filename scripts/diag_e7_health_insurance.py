"""E7 diagnostic (post hoc): on health_insurance (y = weekly hours; 34.5% at 40, 29.9% at 0), is the
CP choice of lasso over rf the shrinkage mechanism of Proposition 3(c) rather than a crossing?
Refits one split; reports prediction spread, width at alpha = 0.1, test MSE, and where the cutoff sits."""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import e4_main as E4
from e2_main import MODELS, cp_quantile

split = int(sys.argv[1]) if len(sys.argv) > 1 else 0
f = E4.fit_split('health_insurance', split, torch.device('cpu'))
yc, yt, Pc, Pt = f['yc'], f['yt'], f['Pc'], f['Pt']
for m in ('lasso', 'rf', 'xgboost', 'dnn'):
    j = MODELS.index(m)
    rc = np.abs(yc - Pc[:, j]); q = cp_quantile(rc, 0.1)
    above = rc > q
    print(f"{m:8s} sd(pred)={Pt[:, j].std():6.2f}  width={2 * q:6.2f}  test MSE={np.mean((yt - Pt[:, j]) ** 2):7.2f}  "
          f"share of y=0 among residuals above cutoff={np.mean(yc[above] == 0):.2f}  "
          f"share of y=40 there={np.mean(yc[above] == 40):.2f}  mean |r| below cutoff={rc[~above].mean():5.2f}")
print(f"y: share 0 = {np.mean(yc == 0):.3f}, share 40 = {np.mean(yc == 40):.3f}")
