#!/usr/bin/env python3
"""E7 report: the pre-registered predictions of Appendix D of the manuscript, on the 20 CTR23 datasets.

Acceptance checks are those of e4b_report.py (E7b) and e4_report.py (E7a), run on the E7 directories.
This script only tabulates and checks predictions; it never fails on a prediction.

Usage: python3 scripts/e7_report.py [--e4b results/e7/e4b] [--e4 results/e7/e4]
"""
import argparse
import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
E7 = os.path.join(BASE_DIR, '..', 'results', 'e7')
ALPHAS = ['0.05', '0.1', '0.2']
COST = 1.01          # "costly" disagreement: CP pick's test MSE more than 1% above the val pick's


def load(d):
    meta = json.load(open(os.path.join(d, 'meta.json')))
    rows = {}
    for k in meta['settings']['datasets']:
        p = os.path.join(d, f'{k}.jsonl')
        if os.path.exists(p):
            rows[k] = sorted((json.loads(l) for l in open(p) if l.strip()), key=lambda x: x['split'])
    return meta, rows


def ms(a):
    a = np.asarray(a, dtype=float)
    return a.mean(), (a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e4b', default=os.path.join(E7, 'e4b'))
    ap.add_argument('--e4', default=os.path.join(E7, 'e4'))
    args = ap.parse_args()
    _, rows = load(args.e4b)
    full = lambda x: str(max(int(n) for n in x['by_n']))

    print('## Table H1: CP width vs validation MSE on the CTR23 datasets (full calibration set)\n')
    print('| dataset | n_cal | ' + ' | '.join(f'α={a}: P(CP=val), CP/val risk (mean, max)' for a in ALPHAS) + ' |')
    print('|---' * (2 + len(ALPHAS)) + '|')
    ratio = {}
    for k, R in rows.items():
        cells = []
        for a in ALPHAS:
            e = [x['by_n'][full(x)] for x in R]
            r = np.array([z['cp'][a]['mse'] / z['val']['mse'] for z in e])
            ratio[k, a] = r
            cells.append(f"{np.mean([z['cp'][a]['choice'] == z['val']['choice'] for z in e]):.2f}, {r.mean():.3f} ({r.max():.3f})")
        print(f"| {k} | {R[0]['n_cal']} | " + ' | '.join(cells) + ' |')

    print('\n## Predictions (informational)\n')
    costly = [(k, a, x) for k, R in rows.items() for a in ALPHAS for x in R
              if x['by_n'][full(x)]['cp'][a]['choice'] != x['by_n'][full(x)]['val']['choice']
              and x['by_n'][full(x)]['cp'][a]['mse'] > COST * x['by_n'][full(x)]['val']['mse']]
    wins = np.array([x['by_n'][full(x)]['cp'][a]['win_share'] for _, a, x in costly], dtype=float)
    print(f'M1 (original, reported) costly disagreements (CP pick != val pick, CP test MSE > {COST} x val), full n: {len(costly)} '
          f'(dataset, alpha, split) cells in {len({(k, a) for k, a, _ in costly})} (dataset, alpha) groups.')
    if len(costly):
        share = float(np.mean(wins > 0.5))
        verdict = ('only reported (fewer than 10 cells)' if len(costly) < 10 else
                   ('as predicted' if share >= 0.8 else 'NOT as predicted'))
        print(f'   CP pick wins on most test points (win share > 0.5) in {share:.2f} of them '
              f'(median win share {np.median(wins):.3f}); predicted >= 0.80 -> {verdict}')
        for k, a in sorted({(k, a) for k, a, _ in costly}):
            w = [x['by_n'][full(x)]['cp'][a]['win_share'] for kk, aa, x in costly if (kk, aa) == (k, a)]
            print(f'   {k} alpha={a}: {len(w)} cells, win shares {np.round(w, 3).tolist()}')

    # M1' (primary): costly cells selected on test half A, bulk/tail decomposition on half B.
    cells = [(k, a, x['by_n'][full(x)]['cp'][a]['m1prime']) for k, R in rows.items() for a in ALPHAS for x in R
             if x['by_n'][full(x)]['cp'][a].get('m1prime') is not None]
    sel = [(k, a, m) for k, a, m in cells if m['mse_ratio_A'] > COST]
    print(f"M1' (primary) costly disagreements selected on test half A: {len(sel)} cells in "
          f"{len({(k, a) for k, a, _ in sel})} (dataset, alpha) groups.")
    if sel:
        bulk = np.array([m['bulk_d'] for *_, m in sel]); tail = np.array([m['tail_d'] for *_, m in sel])
        share = float(np.mean(bulk <= 0))
        verdict = ('only reported (fewer than 10 cells)' if len(sel) < 10 else
                   ('as predicted' if share >= 0.8 else 'NOT as predicted'))
        print(f"   on half B: bulk mean d <= 0 in {share:.2f} of them, tail mean d > 0 in {np.mean(tail > 0):.2f}; "
              f"predicted bulk share >= 0.80 -> {verdict}")
        for k, a in sorted({(k, a) for k, a, _ in sel}):
            mm = [m for kk, aa, m in sel if (kk, aa) == (k, a)]
            print(f"   {k} alpha={a}: {len(mm)} cells, bulk d {np.round([m['bulk_d'] for m in mm], 4).tolist()}, "
                  f"tail d {np.round([m['tail_d'] for m in mm], 4).tolist()}")
    print('(The M1 lines above are the original statistic; it is reported, M1\' is the primary check.)')

    print('M2 persistence: P(CP = val) at full n minus at n=1000, in (dataset, alpha) groups with a costly disagreement; '
          'predicted rise <= 0.1:')
    groups = sorted({(k, a) for k, a, _ in costly})
    ok = 0
    for k, a in groups:
        R = rows[k]
        p = lambda n: np.mean([x['by_n'][n]['cp'][a]['choice'] == x['by_n'][n]['val']['choice'] for x in R if n in x['by_n']])
        d = p(full(R[0])) - p('1000')
        ok += d <= 0.1
        print(f'   [{"yes" if d <= 0.1 else "NO "}] {k} alpha={a}: {p("1000"):.2f} -> {p(full(R[0])):.2f}')
    if groups:
        print(f'   {ok}/{len(groups)} as predicted')

    m = {a: np.mean([ratio[k, a].mean() for k in rows]) for a in ALPHAS}
    print(f"M3 mean over datasets of the mean CP/val risk ratio: " + '  '.join(f'α={a}: {m[a]:.4f}' for a in ALPHAS)
          + f"  -> predicted α=0.2 >= α=0.05: {'as predicted' if m['0.2'] >= m['0.05'] else 'NOT as predicted'}")
    over = [k for k in rows if ratio[k, '0.1'].mean() > COST]
    worst = max(rows, key=lambda k: ratio[k, '0.1'].mean())
    print(f"Descriptive: alpha=0.1, datasets with mean CP/val ratio > {COST}: {len(over)}/{len(rows)} {over}; "
          f"largest: {worst} {ratio[worst, '0.1'].mean():.3f}")

    if os.path.exists(os.path.join(args.e4, 'meta.json')):
        _, rows4 = load(args.e4)
        print('\nE7a (local selection, alpha = 0.1), risk relative to val (mean ± SE over splits):')
        l1 = l2 = 0
        for k, R in rows4.items():
            val = np.array([x['selectors']['val']['mse'] for x in R])
            c, cse = ms(np.array([x['selectors']['crs']['mse'] for x in R]) / val)
            kn, kse = ms(np.array([x['selectors']['knn']['mse'] for x in R]) / val)
            l1 += c <= 1.01
            l2 += 1 - c > 2 * cse
            print(f'   {k}: CRS-forest {c:.3f} ± {cse:.3f}   CRS-kNN {kn:.3f} ± {kse:.3f}')
        print(f'L1 CRS-forest/val <= 1.01 in {l1}/{len(rows4)} datasets (predicted: all).')
        print(f'L2 CRS-forest better than val by more than 2 SE in {l2}/{len(rows4)} datasets (no directional prediction).')


if __name__ == '__main__':
    main()
