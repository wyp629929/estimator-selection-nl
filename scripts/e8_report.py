#!/usr/bin/env python3
"""E8 report: the predictions of Appendix D of the manuscript, pre-registered before the data were
downloaded, on the 31 datasets of the Grinsztajn et al. (2022) benchmark.

Acceptance checks are those of e4b_report.py (E8b) and e4_report.py (E8a), run on the E8 directories.
This script only tabulates and checks predictions; it never fails on a prediction.

  Q1  every dataset and level: mean CP/val held-out risk ratio >= 0.97
  Q2  binary datasets, every level with mean ratio > 1.05 and the CP pick's e > alpha in most splits:
      pattern S in at least 80% of the splits with CP != val
  Q3  regression datasets that are not nearly binary, with at least 3 costly cells (M1' ratio on test
      half A > 1.05): crossings (no worse on the bulk of test half B) in at least half of those cells
  Q4  nearly binary regression datasets (two most frequent values hold >= 50% of y), every level
      with mean ratio > 1.05: pattern S in at least 80% of the splits with CP != val
  Q5  forest conditional selector: mean ratio to val <= 1.01 on at least 85% of the datasets
  Q6  forest conditional selector better than val by more than 2 SE on at least 20% of the
      regression datasets
Pattern S (Table 3 of the paper): the CP pick has more concentrated test predictions, a larger test
risk, and no smaller mean |r| over the calibration points within its own cutoff than the val pick.

Usage: python3 scripts/e8_report.py [--e4b results/e8/e4b] [--e4 results/e8/e4]
"""
import argparse
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from e2_main import MODELS  # noqa: E402
from e4_main import DATA_DIR, E8_BINARY, E8_REGRESSION  # noqa: E402

E8 = os.path.join(BASE_DIR, '..', 'results', 'e8')
ALPHAS = ['0.05', '0.1', '0.2']


def load(d):
    meta = json.load(open(os.path.join(d, 'meta.json')))
    rows = {}
    for k in meta['settings']['datasets']:
        p = os.path.join(d, f'{k}.jsonl')
        if os.path.exists(p):
            rows[k] = sorted((json.loads(l) for l in open(p) if l.strip()), key=lambda x: x['split'])
    return rows


def full(x):
    return x['by_n'][str(max(int(n) for n in x['by_n']))]


def nearly_binary(k):
    y = np.load(os.path.join(DATA_DIR, f'{k}.npz'))['y']
    _, cnt = np.unique(y, return_counts=True)
    return np.sort(cnt)[-2:].sum() / len(y) >= 0.5


def pattern_s(x, a):
    """None if CP and val agree in this split, else whether pattern S holds."""
    e, i = full(x), MODELS.index
    c, v = i(e['cp'][a]['choice']), i(e['val']['choice'])
    if c == v:
        return None
    body = e['cp'][a]['body']
    return bool(x['sd_test'][c] < x['sd_test'][v] and x['model_mse'][c] > x['model_mse'][v] and body[c] >= body[v])


def verdict(ok, tested=True):
    return 'not tested (no qualifying cell)' if not tested else ('borne out' if ok else 'NOT borne out')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--e4b', default=os.path.join(E8, 'e4b'))
    ap.add_argument('--e4', default=os.path.join(E8, 'e4'))
    args = ap.parse_args()
    rb, ra = load(args.e4b), load(args.e4)
    ratio = {(k, a): np.array([full(x)['cp'][a]['mse'] / full(x)['val']['mse'] for x in R])
             for k, R in rb.items() for a in ALPHAS}

    print('## Table E8b: CP width vs validation MSE (full calibration set): mean (max) risk ratio, P(CP = val)\n')
    print('| dataset | type | n_cal | ' + ' | '.join(f'α={a}' for a in ALPHAS) + ' |')
    print('|---' * (3 + len(ALPHAS)) + '|')
    for k, R in rb.items():
        kind = 'binary' if k in E8_BINARY else ('nearly binary' if nearly_binary(k) else 'regression')
        cells = [f"{ratio[k, a].mean():.3f} ({ratio[k, a].max():.3f}), "
                 f"{np.mean([full(x)['cp'][a]['choice'] == full(x)['val']['choice'] for x in R]):.2f}" for a in ALPHAS]
        print(f"| {k} | {kind} | {R[0]['n_cal']} | " + ' | '.join(cells) + ' |')

    print('\n## Pre-registered predictions\n')
    # Q1
    worst = min(((ratio[k, a].mean(), k, a) for k in rb for a in ALPHAS))
    print(f'Q1  min mean ratio over {len(rb)} datasets x {len(ALPHAS)} levels = {worst[0]:.4f} ({worst[1]}, α={worst[2]}): '
          f'{verdict(worst[0] >= 0.97)}')

    # Q2 and Q4
    def shrink_cells(names, need_e):
        out = []
        for k in names:
            if k not in rb:
                continue
            for a in ALPHAS:
                if ratio[k, a].mean() <= 1.05:
                    continue
                if need_e:
                    above = [full(x)['e'][MODELS.index(full(x)['cp'][a]['choice'])] > float(a) for x in rb[k]]
                    if np.mean(above) <= 0.5:
                        continue
                s = [pattern_s(x, a) for x in rb[k]]
                s = [z for z in s if z is not None]
                out.append((k, a, sum(s), len(s)))
        return out
    for tag, cells in (('Q2', shrink_cells(E8_BINARY, True)),
                       ('Q4', shrink_cells([k for k in E8_REGRESSION if k in rb and nearly_binary(k)], False))):
        for k, a, s, n in cells:
            print(f'{tag}  {k} α={a}: mean ratio {ratio[k, a].mean():.3f}; S in {s}/{n} splits with CP != val')
        ok = all(n > 0 and s / n >= 0.8 for *_, s, n in cells)
        print(f'{tag}  {len(cells)} qualifying cells: {verdict(ok, bool(cells))}')

    # Q3
    q3 = []
    for k in E8_REGRESSION:
        if k not in rb or nearly_binary(k):
            continue
        m = [full(x)['cp'][a]['m1prime'] for x in rb[k] for a in ALPHAS if full(x)['cp'][a].get('m1prime')]
        costly = [z for z in m if z['mse_ratio_A'] > 1.05]
        if len(costly) >= 3:
            cr = sum(z['bulk_d'] <= 0 for z in costly)
            q3.append((k, cr, len(costly)))
            print(f'Q3  {k}: crossings in {cr}/{len(costly)} costly cells')
    print(f'Q3  {len(q3)} qualifying datasets: {verdict(all(cr >= n / 2 for _, cr, n in q3), bool(q3))}')

    # Q5 and Q6
    rel = {k: np.array([x['selectors']['crs']['mse'] / x['selectors']['val']['mse'] for x in R]) for k, R in ra.items()}
    n_ok = sum(r.mean() <= 1.01 for r in rel.values())
    print(f'Q5  forest mean ratio <= 1.01 on {n_ok}/{len(rel)} datasets: {verdict(len(rel) > 0 and n_ok / len(rel) >= 0.85, bool(rel))}')
    reg = [k for k in E8_REGRESSION if k in rel]
    gain = [k for k in reg if 1 - rel[k].mean() > 2 * rel[k].std(ddof=1) / np.sqrt(len(rel[k]))]
    print(f'Q6  forest better than val by > 2 SE on {len(gain)}/{len(reg)} regression datasets {gain}: '
          f'{verdict(len(reg) > 0 and len(gain) / len(reg) >= 0.2, bool(reg))}')


if __name__ == '__main__':
    main()
