#!/usr/bin/env python3
"""E4b report: what the split-CP width selects on real data, and acceptance checks.

Acceptance checks concern validity only (completeness, no NaN, CP coverage, same fitted models
as E4, and two deterministic identities). Coverage upper bounds are checked only where the
cutoff is tie-free.

Usage:
  python3 scripts/e4b_report.py --e4 results/e4
  python3 scripts/e4b_report.py --dir results/e4b_quick --no-strict
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(BASE_DIR, '..', 'results', 'e4b')


def load(d, names):
    out = {}
    for k in names:
        p = os.path.join(d, f'{k}.jsonl')
        out[k] = sorted((json.loads(l) for l in open(p) if l.strip()), key=lambda x: x['split']) if os.path.exists(p) else []
    return out


def ms(a):
    a = np.asarray(a, dtype=float)
    return a.mean(), (a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=DEFAULT_DIR)
    ap.add_argument('--e4', default=None, help='E4 results directory: fitted models must give identical test MSE')
    ap.add_argument('--no-strict', action='store_true')
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.dir, 'meta.json')))
    st = meta['settings']
    models, alphas = st['models'], st['alphas']
    rows = load(args.dir, st['datasets'])

    L = ['## Table F: what the split-CP width selects (per dataset, CP level alpha, calibration size n)\n',
         'e = share of calibration residuals with |r| >= 1/2 (Proposition 3); regime (b) when e < alpha, (c) when e > alpha.\n',
         '| dataset | prevalence | alpha | n | mean # positives | positives invisible to the width (share) | CP pick: most frequent | e of CP pick (mean) | P(e of CP pick < alpha) | P(CP = val) | P(CP = test-best) | P(val = test-best) | CP / val risk | Kendall(width, risk) |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k, R in rows.items():
        if not R:
            continue
        prev = np.mean([x['prevalence_cal'] for x in R]) if R[0]['binary'] else None
        best = [models[int(np.argmin(x['model_mse']))] for x in R]
        sizes = sorted({int(n) for x in R for n in x['by_n']}, key=int)
        for a in alphas:
            for n in sizes:
                Rn = [x for x in R if str(n) in x['by_n']]
                e = [x['by_n'][str(n)] for x in Rn]
                cps = [z['cp'][str(a)] for z in e]
                cnt = collections.Counter(c['choice'] for c in cps)
                top, freq = cnt.most_common(1)[0]
                agree = np.mean([c['choice'] == z['val']['choice'] for c, z in zip(cps, e)])
                cp_best = np.mean([c['choice'] == b for c, b in zip(cps, [best[R.index(x)] for x in Rn])])
                val_best = np.mean([z['val']['choice'] == best[R.index(x)] for z, x in zip(e, Rn)])
                ratio = ms([c['mse'] / z['val']['mse'] for c, z in zip(cps, e)])
                tau = np.nanmean([c['kendall_width_vs_risk'] for c in cps])
                if prev is not None:
                    shares = [np.mean(c['pos_invisible']) for c in cps if c['pos_invisible'] is not None]
                    inv = '—' if not shares else f'{np.mean(shares):.2f}'      # '—': no positives in these subsamples
                    e_pick = [z['e'][models.index(c['choice'])] for c, z in zip(cps, e)]
                    cols = (f'{prev:.3f}', f'{np.mean([z["n_pos"] for z in e]):.1f}', inv,
                            f'{np.mean(e_pick):.3f}', f'{np.mean(np.array(e_pick) < a):.2f}')
                else:
                    cols = ('continuous', '—', '—', '—', '—')
                L.append(f'| {k} | {cols[0]} | {a} | {n} | {cols[1]} | {cols[2]} | {top} ({freq}/{len(cps)}) | {cols[3]} | '
                         f'{cols[4]} | {agree:.2f} | {cp_best:.2f} | {val_best:.2f} | {ratio[0]:.4f} ± {ratio[1]:.4f} | {tau:.2f} |')
    print('\n'.join(L))

    res = []
    for k, R in rows.items():
        got = [x['split'] for x in R]
        res.append((f'{k}: splits 0..{st["splits"][k] - 1} all present exactly once',
                    sorted(got) == list(range(st['splits'][k])), f'{len(got)} rows'))
    flat = json.dumps(rows)
    res.append(('no NaN/Infinity in any output', 'NaN' not in flat and 'Infinity' not in flat, ''))
    for k, R in rows.items():
        if len(R) < 2:
            continue
        for n in sorted({int(n) for x in R for n in x['by_n']}):
            Rn = [x for x in R if str(n) in x['by_n']]
            if len(Rn) < 2:
                continue
            for a in alphas:
                cov = np.array([x['by_n'][str(n)]['cp'][str(a)]['coverage'] for x in Rn])
                lo, hi = 1 - a, 1 - a + 1 / (n + 1)
                m, se = cov.mean(axis=0), cov.std(axis=0, ddof=1) / np.sqrt(len(Rn)) + 1e-4
                res.append((f'{k} n={n} alpha={a}: split-CP coverage >= {lo:.3f} - 4 SE for all models',
                            bool(np.all(m >= lo - 4 * se)), f'{m.min():.3f}–{m.max():.3f}'))
                # The upper bound needs continuous scores. E7 rule (results that record distinct_share): check a
                # model on a dataset only if, in every split and calibration size, at least 99% of its calibration
                # |r| values are distinct. Older results: use only (split, model) pairs with a unique cutoff value.
                if 'distinct_share' in Rn[0]['by_n'][str(n)]:
                    cont = np.array([min(e['distinct_share'][j] for x in R for e in x['by_n'].values()) >= 0.99
                                     for j in range(cov.shape[1])])
                    free = np.tile(cont, (len(Rn), 1))
                    rule = f'models with discrete residuals excluded: {[models[j] for j in range(len(cont)) if not cont[j]]}'
                else:
                    free = np.array([x['by_n'][str(n)]['cp'][str(a)]['n_at_cut'] for x in Rn]) == 1
                    rule = f'{int((~free).sum())} (split, model) pairs with ties excluded'
                bad, used = [], 0
                for j in range(cov.shape[1]):
                    c = cov[free[:, j], j]
                    if len(c) >= 2:
                        used += 1
                        if c.mean() > hi + 4 * (c.std(ddof=1) / np.sqrt(len(c)) + 1e-4):
                            bad.append(models[j])
                res.append((f'{k} n={n} alpha={a}: split-CP coverage <= {hi:.3f} + 4 SE where scores are continuous', not bad,
                            f'{used} models checked, ' + rule + (f'; above bound: {bad}' if bad else '')))
    # deterministic bound: positives above the cutoff <= residuals above the k-th order statistic = n - k
    viol, cnt = 0, 0
    for k_, R in rows.items():
        for x in R:
            for n, e in x['by_n'].items():
                if e.get('n_pos') is None:
                    continue
                for a in alphas:
                    kk = min(int(np.ceil((int(n) + 1) * (1 - a))), int(n))
                    for share in e['cp'][str(a)]['pos_invisible'] or []:
                        cnt += 1
                        viol += round(share * e['n_pos']) > int(n) - kk
    res.append(('positives above the cutoff never exceed n - k (implementation check)', cnt == 0 or viol == 0,
                f'{cnt} (split, n, alpha, model) cases, {viol} violations'))
    # Proposition 3, finite-sample form: the cutoff s_(k) is below 1/2 exactly when at most n - k residuals are >= 1/2.
    viol, cnt = 0, 0
    for k_, R in rows.items():
        for x in R:
            for n, e in x['by_n'].items():
                if e.get('e') is None:
                    continue
                for a in alphas:
                    kk = min(int(np.ceil((int(n) + 1) * (1 - a))), int(n))
                    for ej, w in zip(e['e'], e['cp'][str(a)]['widths']):
                        cnt += 1
                        viol += (w / 2 < 0.5) != (round(ej * int(n)) <= int(n) - kk)
    res.append(('cutoff < 1/2 exactly when #{|r| >= 1/2} <= n - k (Proposition 3, finite-sample form)',
                viol == 0, f'{cnt} (split, n, alpha, model) cases, {viol} violations'
                + (' (not applicable: no binary outcomes)' if cnt == 0 else '')))
    if args.e4:
        worst, cnt = 0.0, 0
        for k, R in rows.items():
            p = os.path.join(args.e4, f'{k}.jsonl')
            if not os.path.exists(p):
                continue
            e4 = {x['split']: x for x in map(json.loads, open(p))}
            for x in R:
                if x['split'] in e4:
                    cnt += 1
                    worst = max(worst, float(np.max(np.abs(np.array(x['model_mse']) - np.array(e4[x['split']]['model_mse'])))))
        res.append(('same fitted models as E4 (test MSE identical, max |diff| < 1e-9)', cnt > 0 and worst < 1e-9,
                    f'{cnt} splits compared, max |diff| = {worst:.2e}'))
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
