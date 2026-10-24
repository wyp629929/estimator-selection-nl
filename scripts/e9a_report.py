#!/usr/bin/env python3
"""E9a report: the predictions of Appendix D of the manuscript (written before the run).

  P1  per dataset and level, the mean CP/val ratio with predictions clipped to [0, 1] is within 0.02
      of the unclipped one; datasets with runaway splits (some candidate's unclipped test MSE more
      than 10 times the best one's) may instead have a clipped ratio no larger than the unclipped one
  P2  on the cells that qualified for E8's Q2, plus adult and default at alpha = 0.05: with clipping,
      mean ratio > 1.05 and pattern S in at least 80% of the splits with CP != val
  P3  on creditcard, mammography and homecredit, at every level where all candidates have e < alpha
      (clipped): the CP pick's mean share of positives above its cutoff is at least 0.9

Usage: python3 scripts/e9a_report.py [results directory, default results/e9a]
"""
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from e2_main import MODELS  # noqa: E402

OUT = os.path.join(BASE_DIR, '..', 'results', 'e9a')
ALPHAS = ['0.05', '0.1', '0.2']
Q2_E8 = [('albert', a) for a in ALPHAS] + [('covertype', '0.05'), ('electricity', '0.05')] + \
        [('eye_movements', a) for a in ALPHAS] + [('road_safety', '0.05')] + [('higgs', a) for a in ALPHAS] + \
        [('magic_telescope', '0.05'), ('credit', '0.05'), ('credit', '0.1'), ('jannis', '0.05'), ('jannis', '0.1')]
P2_CELLS = Q2_E8 + [('adult', '0.05'), ('default', '0.05')]


def load(d=OUT):
    rows = {}
    for f in sorted(os.listdir(d)):
        if f.endswith('.jsonl'):
            R = [json.loads(l) for l in open(os.path.join(d, f)) if l.strip()]
            rows[f[:-6]] = sorted(R, key=lambda x: x['split'])
    return rows


def ratio(x, kind, a):
    s = x[kind]
    i = MODELS.index
    return s['model_mse'][i(s['cp'][a]['choice'])] / s['model_mse'][i(s['val'])]


def pattern_s(x, kind, a):
    s, i = x[kind], MODELS.index
    c, v = i(s['cp'][a]['choice']), i(s['val'])
    if c == v:
        return None
    b = s['cp'][a]['body']
    return s['sd_test'][c] < s['sd_test'][v] and s['model_mse'][c] > s['model_mse'][v] and b[c] >= b[v]


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else OUT)
    print('## E9a: mean CP/val ratio without and with clipping to [0, 1] (same refit), share of test predictions outside [0, 1]\n')
    print('| dataset | splits | outside [0,1] | runaway splits | ' + ' | '.join(f'α={a} raw → clipped' for a in ALPHAS) + ' |')
    print('|---' * (4 + len(ALPHAS)) + '|')
    p1_ok, p1_fail = True, []
    for k, R in rows.items():
        runaway = sum(max(x['raw']['model_mse']) > 10 * min(x['raw']['model_mse']) for x in R)
        cells = []
        for a in ALPHAS:
            r0, r1 = np.mean([ratio(x, 'raw', a) for x in R]), np.mean([ratio(x, 'clipped', a) for x in R])
            ok = abs(r1 - r0) <= 0.02 or (runaway > 0 and r1 <= r0)
            if not ok:
                p1_ok = False
                p1_fail.append((k, a, round(r0, 3), round(r1, 3)))
            fmt = lambda r: f'{r:.3f}' if r < 1e3 else f'{r:.2e}'
            cells.append(f'{fmt(r0)} → {fmt(r1)}')
        print(f"| {k} | {len(R)} | {np.mean([x['outside'] for x in R]):.3f} | {runaway} | " + ' | '.join(cells) + ' |')

    print('\n## Predictions\n')
    print(f"P1  {'borne out' if p1_ok else 'NOT borne out'}" + (f'; failing cells: {p1_fail}' if p1_fail else ''))
    p2 = []
    for k, a in P2_CELLS:
        if k not in rows:
            continue
        R = rows[k]
        r1 = np.mean([ratio(x, 'clipped', a) for x in R])
        s = [z for z in (pattern_s(x, 'clipped', a) for x in R) if z is not None]
        ok = r1 > 1.05 and len(s) > 0 and sum(s) / len(s) >= 0.8
        p2.append(ok)
        print(f'P2  {k} α={a}: clipped mean ratio {r1:.3f}; S in {sum(s)}/{len(s)} splits with CP != val: {"ok" if ok else "FAILS"}')
    print(f"P2  {sum(p2)}/{len(p2)} cells: {'borne out' if all(p2) and p2 else 'NOT borne out'}")
    p3 = []
    for k in ('creditcard', 'mammography', 'homecredit'):
        if k not in rows:
            continue
        R = rows[k]
        for a in ALPHAS:
            if not all(max(x['clipped']['e']) < float(a) for x in R):
                continue
            share = np.mean([x['clipped']['cp'][a]['pos_above'][MODELS.index(x['clipped']['cp'][a]['choice'])] for x in R])
            p3.append(share >= 0.9)
            print(f'P3  {k} α={a}: share of positives above the CP pick\'s cutoff {share:.3f}')
    print(f"P3  {sum(p3)}/{len(p3)} cells: {'borne out' if all(p3) and p3 else ('not tested' if not p3 else 'NOT borne out')}")


if __name__ == '__main__':
    main()
