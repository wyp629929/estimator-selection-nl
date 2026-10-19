#!/usr/bin/env python3
"""E5 report: tables, the pre-registered predictions (Appendix D of the manuscript), and acceptance checks.

Predictions are reported, never used as a gate. Acceptance checks concern the implementation:
completeness, no NaN, split-CP expected coverage in [1-alpha, 1-alpha+1/(n+1)] +- 4 SE (continuous
scores, so the upper bound applies), and the mean empirical cutoff at n=16000 within 4 SE of the
population cutoff when p < p_crit.

Usage: python3 scripts/e5_report.py [--in results/e5/e5_mechanism.json] [--no-strict]
"""
import argparse
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(BASE_DIR, '..', 'results', 'e5', 'e5_mechanism.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default=DEFAULT_IN)
    ap.add_argument('--no-strict', action='store_true')
    args = ap.parse_args()
    d = json.load(open(args.inp))
    st, cells = d['settings'], d['cells']
    ns = st['ns']

    print('## Table E5: P(selector picks model 1, the globally worse model) by p / p_crit\n')
    print('G(1) - G(2) = %g in every cell. Each entry: CP / val; columns are calibration sizes n.\n' % st['delta'])
    for noise in st['noises']:
        for a in st['alphas']:
            for b in st['bs']:
                C = [c for c in cells if c['noise'] == noise and c['alpha'] == a and c['b'] == b]
                print(f"### {noise}, alpha={a}, b={b}: p_crit={C[0]['pcrit']:.4f} (closed-form pbar={C[0]['pbar']:.4f})\n")
                print('| p / p_crit | p | ' + ' | '.join(f'n={n}' for n in ns) + ' |')
                print('|---' * (len(ns) + 2) + '|')
                for m in st['mults']:
                    row = sorted((c for c in C if c['mult'] == m), key=lambda c: c['n'])
                    print(f"| {m} | {row[0]['p']:.4f} | " + ' | '.join(f"{c['P_cp1']:.2f} / {c['P_val1']:.2f}" for c in row) + ' |')
                print()

    print('## Predictions (informational)\n')
    q_hit = q_tot = 0; misses = []
    for c in cells:
        if c['z'] >= 3:
            q_tot += 1
            ok = c['P_cp1'] >= 0.95 if c['mult'] < 1 else c['P_cp1'] <= 0.05
            q_hit += ok
            if not ok:
                misses.append(c)
    print(f'P2 quantitative (cells with z >= 3): {q_hit}/{q_tot} as predicted '
          f'(P(CP wrong) >= 0.95 below p_crit, <= 0.05 above).')
    for c in misses:
        print(f"   miss: {c['noise']} alpha={c['alpha']} b={c['b']} m={c['mult']} n={c['n']} z={c['z']:.1f} P_cp1={c['P_cp1']:.3f}")
    d_hit = d_tot = 0
    groups = {}
    for c in cells:
        groups.setdefault((c['noise'], c['alpha'], c['b'], c['mult']), []).append(c)
    for key, G in groups.items():
        G = sorted(G, key=lambda c: c['n'])
        d_tot += 1
        d_hit += (G[-1]['P_cp1'] >= G[0]['P_cp1']) if key[3] < 1 else (G[-1]['P_cp1'] <= G[0]['P_cp1'])
    print(f'P2 direction (P(CP wrong) moves towards 1 below p_crit and towards 0 above it, smallest vs largest n): '
          f'{d_hit}/{d_tot}.')
    big = [c for c in cells if c['n'] == max(ns)]
    print(f"P3 val at n={max(ns)}: max P(val wrong) = {max(c['P_val1'] for c in big):.3f}; "
          f"cells with P(val wrong) > 0.05: {sum(c['P_val1'] > 0.05 for c in big)}/{len(big)}.")
    pv, pn = np.array([c['P_val1'] for c in cells]), np.array([c['P_noB'] for c in cells])
    print(f'P3 finite-sample: corr(P(val wrong), (1-p)^n) = {np.corrcoef(pv, pn)[0, 1]:.3f}; '
          f'cells where P(val wrong) < (1-p)^n - 0.05: {int(np.sum(pv < pn - 0.05))}/{len(cells)}.')

    res = []
    exp = {(c['noise'], c['alpha'], c['b'], c['mult'], c['n']) for c in cells}
    res.append(('all cells present exactly once',
                len(cells) == len(exp) == len(st['noises']) * len(st['alphas']) * len(st['bs']) * len(st['mults']) * len(ns),
                f'{len(cells)} cells'))
    res.append(('no NaN/Infinity', 'NaN' not in json.dumps(cells) and 'Infinity' not in json.dumps(cells), ''))
    bad = []
    for c in cells:
        lo, hi = 1 - c['alpha'], 1 - c['alpha'] + 1 / (c['n'] + 1)
        for j in ('1', '2'):
            m, se = c['cov' + j], c['cov' + j + '_se'] + 1e-6
            if not (lo - 4 * se <= m <= hi + 4 * se):
                bad.append((c['noise'], c['alpha'], c['b'], c['mult'], c['n'], j, round(m, 4)))
    res.append(('split-CP expected coverage within [1-alpha, 1-alpha+1/(n+1)] +- 4 SE (both models, all cells)',
                not bad, f'{2 * len(cells)} checks' + (f'; outside: {bad[:5]}' if bad else '')))
    bad = []
    for c in cells:
        if c['n'] == max(ns) and c['mult'] < 1:
            for j, W in (('1', c['W1_pop']), ('2', c['W2_pop'])):
                if abs(c['cut' + j] - W / 2) > 4 * c['cut' + j + '_se'] + 1e-9:
                    bad.append((c['noise'], c['alpha'], c['b'], c['mult'], j, round(c['cut' + j], 4), round(W / 2, 4)))
    res.append((f'mean empirical cutoff within 4 SE of the population cutoff (n={max(ns)}, p < p_crit)', not bad,
                f'outside: {bad[:5]}' if bad else ''))
    print('\nAcceptance checks:')
    failed = 0
    for name, ok, detail in res:
        failed += not ok
        print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}')
    print(f'\n{failed} check(s) failed.')
    if failed and not args.no_strict:
        sys.exit(1)


if __name__ == '__main__':
    main()
