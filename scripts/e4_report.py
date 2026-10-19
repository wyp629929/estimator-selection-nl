#!/usr/bin/env python3
"""E4 report: held-out risk of each selector on real data, and acceptance checks.

Risk is reported relative to the validation-MSE rule on the same split (ratio < 1 means
lower held-out risk than the best global rule chosen from the calibration data).

Usage:
  python3 scripts/e4_report.py
  python3 scripts/e4_report.py --dir results/e4_quick --no-strict
"""
import argparse
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(BASE_DIR, '..', 'results', 'e4')


def ms(a):
    a = np.asarray(a, dtype=float)
    return a.mean(), (a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=DEFAULT_DIR)
    ap.add_argument('--no-strict', action='store_true')
    ap.add_argument('--prev', default=None, help='earlier E4 run: shared splits must give identical model test MSE')
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.dir, 'meta.json')))
    st = meta['settings']
    sels = st['global_selectors'] + st['local_selectors']
    rows = {}
    for k in st['datasets']:
        p = os.path.join(args.dir, f'{k}.jsonl')
        rows[k] = sorted((json.loads(l) for l in open(p) if l.strip()), key=lambda x: x['split']) if os.path.exists(p) else []

    L = ['## Table E1: test MSE of each estimator (mean over splits)\n',
         '| dataset | n train / cal / test | ' + ' | '.join(st['models']) + ' |', '|---' * (len(st['models']) + 2) + '|']
    for k, R in rows.items():
        if R:
            m = np.mean([x['model_mse'] for x in R], axis=0)
            L.append(f'| {k} | {R[0]["n_train"]} / {R[0]["n_cal"]} / {R[0]["n_test"]} | ' + ' | '.join(f'{v:.4g}' for v in m) + ' |')
    L += ['\n## Table E2: held-out risk relative to the validation-MSE rule (mean ± SE over splits)\n',
          '| dataset | ' + ' | '.join(sels) + ' | best single model (infeasible) | P(kNN level = const) | P(forest level = const) |',
          '|---' * (len(sels) + 4) + '|']
    summary = {}
    for k, R in rows.items():
        if not R:
            continue
        val = np.array([x['selectors']['val']['mse'] for x in R])
        cells, summary[k] = [], {}
        for s in sels:
            m, se = ms(np.array([x['selectors'][s]['mse'] for x in R]) / val)
            cells.append(f'{m:.3f} ± {se:.3f}')
            summary[k][s] = (m, se)
        best = ms(np.array([min(x['model_mse']) for x in R]) / val)
        pc = {s: float(np.mean([x['selectors'][s]['level'] == 'const' for x in R])) for s in ('knn', 'crs')}
        L.append(f'| {k} | ' + ' | '.join(cells) + f' | {best[0]:.3f} ± {best[1]:.3f} | {pc["knn"]:.2f} | {pc["crs"]:.2f} |')
        summary[k]['best_single'] = best
        summary[k]['const_level_rate'] = pc
    print('\n'.join(L))
    with open(os.path.join(args.dir, 'summary.json'), 'w') as f:
        json.dump(dict(meta=meta, summary=summary), f)

    res = []
    for k in st['datasets']:
        got = [x['split'] for x in rows[k]]
        res.append((f'{k}: splits 0..{st["splits"][k] - 1} all present exactly once',
                    sorted(got) == list(range(st['splits'][k])), f'{len(got)} rows'))
    flat = json.dumps(rows)
    res.append(('no NaN/Infinity in any output', 'NaN' not in flat and 'Infinity' not in flat, ''))
    for k, R in rows.items():
        if len(R) < 2:
            continue
        n = R[0]['n_cal']
        lo, hi = 1 - st['cp_alpha'], 1 - st['cp_alpha'] + 1 / (n + 1)
        cov = np.array([x['cp_coverage'] for x in R])
        m, se = cov.mean(axis=0), cov.std(axis=0, ddof=1) / np.sqrt(len(R)) + 1e-4
        res.append((f'{k}: split-CP test coverage >= {lo:.3f} - 4 SE for all models',
                    bool(np.all(m >= lo - 4 * se)), f'{m.min():.3f}–{m.max():.3f}'))
        # The upper bound needs continuous scores. E7 rule (results that record cp_distinct_share): check a
        # model only if at least 99% of its calibration |r| values are distinct in every split.
        # Older results: use only (split, model) pairs with a unique cutoff value.
        if 'cp_distinct_share' in R[0]:
            cont = np.array([min(x['cp_distinct_share'][j] for x in R) >= 0.99 for j in range(cov.shape[1])])
            free = np.tile(cont, (len(R), 1))
            rule = f"models with discrete residuals excluded: {[st['models'][j] for j in range(len(cont)) if not cont[j]]}"
        else:
            free = np.array([x['cp_n_at_cut'] for x in R]) == 1
            rule = f'{int((~free).sum())} (split, model) pairs with ties excluded'
        bad, used = [], 0
        for j in range(cov.shape[1]):
            c = cov[free[:, j], j]
            if len(c) >= 2:
                used += 1
                if c.mean() > hi + 4 * (c.std(ddof=1) / np.sqrt(len(c)) + 1e-4):
                    bad.append(st['models'][j])
        res.append((f'{k}: split-CP test coverage <= {hi:.3f} + 4 SE where scores are continuous', not bad,
                    f'{used} models checked, ' + rule + (f'; above bound: {bad}' if bad else '')))
        um, use = ms([x['U_joint_coverage'] for x in R])
        res.append((f'{k}: U_M joint coverage >= {lo:.2f} - 4 SE', um >= lo - 4 * use, f'{um:.3f} (SE {use:.3f})'))
    if args.prev:
        worst, cnt = 0.0, 0
        for k, R in rows.items():
            p = os.path.join(args.prev, f'{k}.jsonl')
            if not os.path.exists(p):
                continue
            prev = {x['split']: x for x in map(json.loads, filter(str.strip, open(p)))}
            for x in R:
                if x['split'] in prev:
                    cnt += 1
                    worst = max(worst, float(np.max(np.abs(np.array(x['model_mse']) - np.array(prev[x['split']]['model_mse'])))),
                                abs(x['selectors']['val']['mse'] - prev[x['split']]['selectors']['val']['mse']))
        res.append(('fitted models and the val selector match the previous run (max |diff| < 1e-9)',
                    cnt > 0 and worst < 1e-9, f'{cnt} splits compared, max |diff| = {worst:.2e}'))
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
