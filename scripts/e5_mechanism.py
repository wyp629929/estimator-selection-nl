#!/usr/bin/env python3
"""E5: the mechanism of Proposition 2(b), as a dose-response experiment (Section 4.2 and Appendix D of the manuscript).

Model 1 has error c on a set B of probability p and none elsewhere (g_1 = c 1_B); model 2 has
error b everywhere (g_2 = b). With c = sqrt((b^2 + DELTA) / p), model 2 is better in global risk
by exactly DELTA for every p. Theory: the population CP width ranks model 1 first iff
p < p_crit(alpha, b), and p_crit is close to the closed-form threshold
pbar = 1 - (1 - alpha) / H(q_b). Validation MSE ranks model 2 first as n grows.

Only residuals are drawn (indicator of B and noise), which is the conditional-on-fit setting
of the theory. For each cell: P(CP picks model 1), P(val picks model 1), the exact expected
coverage of each model's split-CP interval (population CDF at the empirical cutoff), and the
mean empirical cutoff.

Usage: python3 scripts/e5_mechanism.py [--quick]
"""
import argparse
import json
import os
import time

import numpy as np
from scipy import optimize, stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, '..', 'results', 'e5', 'e5_mechanism.json')
BASE_SEED = 2024
DELTA = 1.0
NOISES = {'normal': stats.norm(), 't3': stats.t(3, scale=1 / np.sqrt(3))}   # both with variance 1
ALPHAS = [0.05, 0.1, 0.2]
BS = [0.5, 1.0]
MULTS = [0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0]
NS = [250, 1000, 4000, 16000]
REPS = 1000
CHUNK = 100


def Q(F, tau):
    return optimize.brentq(lambda t: F(t) - tau, 0.0, 1e5, xtol=1e-12)


def population(Z, alpha, b):
    tau = 1 - alpha
    h = lambda a, t: Z.cdf(t - a) - Z.cdf(-t - a)
    F2 = lambda t: h(b, t)
    qb = Q(F2, tau)
    pbar = 1 - tau / h(0.0, qb)

    def F1(p):
        c = np.sqrt((b * b + DELTA) / p)
        return lambda t: (1 - p) * h(0.0, t) + p * h(c, t)
    pcrit = optimize.brentq(lambda p: Q(F1(p), tau) - qb, 1e-6, 2 * alpha, xtol=1e-12)
    return F1, F2, qb, pbar, pcrit


def resolvability(F1, F2, q1, q2, tau, n, e=1e-6):
    """Population cutoff gap over its sd, treating the two empirical quantiles as independent."""
    sd = lambda F, q: np.sqrt(tau * (1 - tau) / n) / ((F(q + e) - F(q - e)) / (2 * e))
    return abs(q1 - q2) / np.hypot(sd(F1, q1), sd(F2, q2))


def draw_noise(name, rng, shape):
    return rng.standard_normal(shape) if name == 'normal' else rng.standard_t(3, shape) / np.sqrt(3)


def run_cell(noise, alpha, b, m, n, F1p, F2, p, c, rng, reps):
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    cp1 = val1 = 0
    cov1, cov2, cut1, cut2 = [], [], [], []
    for start in range(0, reps, CHUNK):
        r = min(CHUNK, reps - start)
        eps = draw_noise(noise, rng, (r, n))
        inB = rng.random((r, n)) < p
        r1, r2 = eps + c * inB, eps + b
        q1 = np.partition(np.abs(r1), k - 1, axis=1)[:, k - 1]
        q2 = np.partition(np.abs(r2), k - 1, axis=1)[:, k - 1]
        cp1 += int(np.sum(q1 < q2))
        val1 += int(np.sum(np.mean(r1 ** 2, axis=1) < np.mean(r2 ** 2, axis=1)))
        cov1 += F1p(q1).tolist()
        cov2 += F2(q2).tolist()
        cut1 += q1.tolist(); cut2 += q2.tolist()
    se = lambda a: float(np.std(a, ddof=1) / np.sqrt(len(a)))
    return dict(P_cp1=cp1 / reps, P_val1=val1 / reps,
                cov1=float(np.mean(cov1)), cov1_se=se(cov1), cov2=float(np.mean(cov2)), cov2_se=se(cov2),
                cut1=float(np.mean(cut1)), cut1_se=se(cut1), cut2=float(np.mean(cut2)), cut2_se=se(cut2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='100 reps, n up to 1000')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    reps, ns = (100, NS[:2]) if args.quick else (REPS, NS)
    if args.quick and args.out == OUT:
        args.out = OUT.replace('.json', '_quick.json')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    cells = []
    for i_z, (noise, Z) in enumerate(NOISES.items()):
        for i_a, alpha in enumerate(ALPHAS):
            for i_b, b in enumerate(BS):
                F1, F2, qb, pbar, pcrit = population(Z, alpha, b)
                for i_m, m in enumerate(MULTS):
                    p = m * pcrit
                    c = float(np.sqrt((b * b + DELTA) / p))
                    F1p = F1(p)
                    q1 = Q(F1p, 1 - alpha)
                    for i_n, n in enumerate(ns):
                        rng = np.random.default_rng([BASE_SEED, i_z, i_a, i_b, i_m, i_n])
                        res = run_cell(noise, alpha, b, m, n, F1p, F2, p, c, rng, reps)
                        cells.append(dict(noise=noise, alpha=alpha, b=b, mult=m, p=p, c=c, n=n, reps=reps,
                                          pbar=pbar, pcrit=pcrit, W1_pop=2 * q1, W2_pop=2 * qb,
                                          z=resolvability(F1p, F2, q1, qb, 1 - alpha, n),
                                          P_noB=(1 - p) ** n, **res))
                print(f'{noise} alpha={alpha} b={b} done, {time.time() - t0:.0f}s', flush=True)
    with open(args.out, 'w') as f:
        json.dump(dict(settings=dict(delta=DELTA, alphas=ALPHAS, bs=BS, mults=MULTS, ns=ns, reps=reps,
                                     noises=list(NOISES), base_seed=BASE_SEED, quick=args.quick),
                       cells=cells, seconds=time.time() - t0), f)
    print(f'-> {args.out} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
