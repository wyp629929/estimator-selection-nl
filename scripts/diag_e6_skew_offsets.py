"""E6 diagnostic: under skewed noise (linear regime), how different are the candidates' mean
residuals, and do CP widths at alpha = 0.05 / 0.2 track them? Refits replications of noise_skew."""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import e2_main as E

cfg = E.VARIANTS['noise_skew']
for rep in range(int(sys.argv[1]) if len(sys.argv) > 1 else 4):
    regime = 'linear'
    rng = np.random.default_rng([E.BASE_SEED, E.REGIMES.index(regime), rep])
    X, y, _ = E.draw(regime, E.N_TRAIN, rng, cfg)
    models, dnn = E.fit_models(X, y, E.BASE_SEED + rep, torch.device('cpu'))
    Xi, _, fi = E.draw(regime, E.N_INT, rng, cfg)
    Pi = E.predict_all(models, dnn, Xi, 'cpu')
    bias = (fi[:, None] - Pi).mean(axis=0)                       # mean realized error E g_M(X)
    spread = (fi[:, None] - Pi).std(axis=0)                      # sd of g_M(X): what is not a constant shift
    G = ((fi[:, None] - Pi) ** 2).mean(axis=0) + 1
    lin = [0, 1, 2]
    print(f"rep {rep}: mean bias E g_M  " + ' '.join(f"{m}={b:+.3f}" for m, b in zip(E.MODELS, bias)))
    print(f"        sd of g_M        " + ' '.join(f"{m}={s:.3f}" for m, s in zip(E.MODELS, spread)))
    print(f"        G - 1            " + ' '.join(f"{m}={g - 1:.4f}" for m, g in zip(E.MODELS, G)))
    print(f"        linear models: bias range {np.ptp(bias[lin]):.4f}, G range {np.ptp(G[lin]):.5f}")
