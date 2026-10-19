#!/usr/bin/env python3
"""E6 report: the three noise variants side by side, and the predictions of Appendix D of the manuscript.

Acceptance checks (coverage, Lemma 1, completeness) are those of e2_report.py, run on each variant
directory separately. This script only tabulates and checks the pre-registered predictions; it never
fails on a prediction.

Usage: python3 scripts/e6_report.py [--dir results/e6]
"""
import argparse
import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(BASE_DIR, '..', 'results', 'e6')
NOISES = ['normal', 't3', 'skew']
REGIMES = ['linear', 'nonlinear', 'threshold']
CPS = ['cp_a0.05', 'cp', 'cp_a0.2']          # alpha = 0.05, 0.1, 0.2
ALPHA_OF = {'cp_a0.05': 0.05, 'cp': 0.1, 'cp_a0.2': 0.2}


def ms(a):
    a = np.asarray(a, dtype=float)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=DEFAULT_DIR)
    args = ap.parse_args()
    rows = {}
    for z in NOISES:
        for r in REGIMES:
            p = os.path.join(args.dir, f'noise_{z}', f'{r}.jsonl')
            rows[z, r] = [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
    ncal = sorted({n for R in rows.values() for x in R for n in x['by_ncal']}, key=int)
    get = lambda z, r, n, s, k: [x['by_ncal'][n]['selectors'][s][k] for x in rows[z, r]]

    print('## Table G1: global target. G regret and pairwise agreement with the G ordering (mean over replications)\n')
    print('| noise | regime | n | val: regret / pairwise | ' + ' | '.join(f'CP α={ALPHA_OF[s]}: regret / pairwise' for s in CPS) + ' |')
    print('|---' * (4 + len(CPS)) + '|')
    for z in NOISES:
        for r in REGIMES:
            if not rows[z, r]:
                continue
            for n in ncal:
                cell = lambda s: f"{np.mean(get(z, r, n, s, 'G_regret')):.4f} / {np.mean(get(z, r, n, s, 'pairwise')):.3f}"
                print(f'| {z} | {r} | {n} | {cell("val")} | ' + ' | '.join(cell(s) for s in CPS) + ' |')

    print('\n## Table G2: local target. Local regret (mean ± SE)\n')
    print('| noise | regime | n | val | CRS-kNN | CRS-forest |')
    print('|---|---|---|---|---|---|')
    for z in NOISES:
        for r in REGIMES:
            if not rows[z, r]:
                continue
            for n in ncal:
                f = lambda s: '%.3f ± %.3f' % ms(get(z, r, n, s, 'regret'))
                print(f'| {z} | {r} | {n} | {f("val")} | {f("knn")} | {f("crs")} |')

    print('\n## Predictions (informational)\n')

    def paired(z, r, n, a, b):           # mean and SE of a - b, paired by replication
        return ms(np.array(get(z, r, n, a, 'regret')) - np.array(get(z, r, n, b, 'regret')))

    def gregret_ok(z):
        cells = [(r, n) for r in REGIMES for n in ncal if rows[z, r]]
        bad = [(r, n) for r, n in cells if np.mean(get(z, r, n, 'val', 'G_regret')) > np.mean(get(z, r, n, 'cp', 'G_regret'))]
        return len(cells) - len(bad), len(cells), bad

    def crs_ok(z):
        out = []
        for n in [m for m in ncal if int(m) >= 1000]:
            if rows[z, 'nonlinear']:
                d, se = paired(z, 'nonlinear', n, 'val', 'crs')
                out.append((f'nonlinear n={n}: val - CRS = {d:.3f} (SE {se:.3f}), predicted > 2 SE', d > 2 * se))
            if rows[z, 'linear']:
                d, se = paired(z, 'linear', n, 'crs', 'val')
                out.append((f'linear n={n}: CRS - val = {d:.4f} (SE {se:.4f}), predicted |.| <= 2 SE', abs(d) <= 2 * se))
        return out

    for z, tag in (('t3', 'T'), ('skew', 'S')):
        if not any(rows[z, r] for r in REGIMES):
            print(f'{z}: no results yet')
            continue
        hit, tot, bad = gregret_ok(z)
        print(f'{tag}{"1" if tag == "T" else "4"} ({z}) val G regret <= CP (alpha=0.1) G regret: {hit}/{tot}' + (f'; not: {bad}' if bad else ''))
        print(f'{tag}{"2" if tag == "T" else "3"} ({z}) CRS vs val:')
        for msg, ok in crs_ok(z):
            print(f'   [{"yes" if ok else "NO "}] {msg}')
    if rows['t3', 'nonlinear']:
        k = [np.mean(get('t3', 'nonlinear', n, 'knn', 'regret')) for n in ncal]
        print(f'T3 (t3) CRS-kNN local regret in nonlinear by n {ncal}: {np.round(k, 3).tolist()}; '
              f'strictly decreasing: {all(a > b for a, b in zip(k, k[1:]))}')
    if rows['normal', 'linear'] and rows['skew', 'linear']:
        print('S1 (skew vs normal, linear): change in CP pairwise agreement, skew minus normal (SE); predicted < 0 at '
              'alpha 0.05 and 0.2, and below the change at alpha 0.1:')
        for n in ncal:
            d = {}
            for s in CPS:
                (a, sa), (b, sb) = ms(get('skew', 'linear', n, s, 'pairwise')), ms(get('normal', 'linear', n, s, 'pairwise'))
                d[s] = (a - b, np.hypot(sa, sb))
            ok = d['cp_a0.05'][0] < 0 and d['cp_a0.2'][0] < 0 and d['cp_a0.05'][0] < d['cp'][0] and d['cp_a0.2'][0] < d['cp'][0]
            print(f'   n={n}: ' + '  '.join(f'α={ALPHA_OF[s]}: {d[s][0]:+.3f} ({d[s][1]:.3f})' for s in CPS) + f'  -> {"as predicted" if ok else "NOT as predicted"}')
        print('S2 (skew vs normal) val pairwise agreement within 2 SE:')
        for r in REGIMES:
            for n in ncal:
                (a, sa), (b, sb) = ms(get('skew', r, n, 'val', 'pairwise')), ms(get('normal', r, n, 'val', 'pairwise'))
                print(f'   [{"yes" if abs(a - b) <= 2 * np.hypot(sa, sb) else "NO "}] {r} n={n}: {a - b:+.3f} (SE {np.hypot(sa, sb):.3f})')


if __name__ == '__main__':
    main()
