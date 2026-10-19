#!/usr/bin/env python3
"""E2 report: summary tables and acceptance checks for the JSONL output of e2_main.py.

Usage:
  python3 scripts/e2_report.py                                  # results/e2
  python3 scripts/e2_report.py --dir results/e2_quick --no-strict
  python3 scripts/e2_report.py --repro results/e2_repro          # also compare a re-run subset
  python3 scripts/e2_report.py --prev results/e2_v1              # selectors shared with an earlier run must be identical
  python3 scripts/e2_report.py --dir results/e3/corr              # E3 variant (regime subset read from meta.json)
  python3 scripts/e2_report.py --prev results/e2 --prev-selectors val rhat G   # default dir results/e2
Writes <dir>/summary.json. Exit code 1 if any acceptance check fails (unless --no-strict).
"""
import argparse
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(BASE_DIR, '..', 'results', 'e2')
REGIMES = ['linear', 'semiparametric', 'nonlinear', 'threshold', 'highdim', 'heteroscedastic']


def load(d, regimes=REGIMES):
    rows = {}
    for r in regimes:
        p = os.path.join(d, f'{r}.jsonl')
        if os.path.exists(p):
            with open(p) as f:
                rows[r] = sorted((json.loads(l) for l in f if l.strip()), key=lambda x: x['rep'])
    return rows


def ms(a):
    a = np.asarray(a, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float('nan')


def fmt(a, nd=3):
    m, s = ms(a)
    return f'{m:.{nd}f} ± {s:.{nd}f}'


def summarize(rows, st):
    SELECTORS = st['global_selectors'] + st['local_selectors']
    models, ncal, eg, el = st['models'], [str(n) for n in st['n_cal']], st['eps_global'], st['eps_local']
    out = {}
    for r, R in rows.items():
        o = {'reps': len(R),
             'G_mean': np.mean([x['G'] for x in R], axis=0).tolist(),
             'G_se': (np.std([x['G'] for x in R], axis=0, ddof=1) / np.sqrt(len(R))).tolist(),
             'oracle_choice_freq': np.bincount([int(np.argmin(x['G'])) for x in R], minlength=len(models)).tolist(),
             'G_gap12_lt': {str(e): float(np.mean([x['G_gap12'] < e for x in R])) for e in eg[1:]},
             'local_gap_lt': np.mean([x['local_gap_lt'] for x in R], axis=0).tolist(),
             'floor': np.mean([x['floor'] for x in R], axis=0).tolist(), 'by_ncal': {}}
        for n in ncal:
            b = {}
            for s in SELECTORS:
                v = [x['by_ncal'][n]['selectors'][s] for x in R]
                d = {'loc_regret': ms([z['regret'] for z in v]),
                     'loc_delta': [ms([z['delta'][i] for z in v]) for i in range(len(el))]}
                if s in ('cp', 'val'):
                    d.update(G_regret=ms([z['G_regret'] for z in v]),
                             G_delta=[ms([z['G_delta'][i] for z in v]) for i in range(len(eg))],
                             pairwise=ms([z['pairwise'] for z in v]),
                             choice_freq={m: float(np.mean([z['choice'] == m for z in v])) for m in models})
                b[s] = d
            b['cp_coverage'] = np.mean([x['by_ncal'][n]['cp_coverage'] for x in R], axis=0).tolist()
            b['U_joint_coverage'] = ms([x['by_ncal'][n]['U_joint_coverage'] for x in R])
            o['by_ncal'][n] = b
        out[r] = o
    return out


def tables(rows, st):
    SELECTORS = st['global_selectors'] + st['local_selectors']
    models, ncal, eg, el = st['models'], [str(n) for n in st['n_cal']], st['eps_global'], st['eps_local']
    L = ['## Table A: global risk G(M) incl. noise variance (mean ± SE over replications)\n',
         '| regime | ' + ' | '.join(models) + ' | P(oracle gap < %g) | P(gap < %g) |' % (eg[1], eg[2]),
         '|---' * (len(models) + 3) + '|']
    for r, R in rows.items():
        G = np.array([x['G'] for x in R])
        cells = [f'{G[:, j].mean():.3f}' for j in range(len(models))]
        L.append(f'| {r} | ' + ' | '.join(cells) +
                 f' | {np.mean([x["G_gap12"] < eg[1] for x in R]):.2f} | {np.mean([x["G_gap12"] < eg[2] for x in R]):.2f} |')

    L += ['\n## Table B: global target (selector vs argmin G)\n',
          '| regime | n_cal | selector | G regret | Δ_glob exact | Δ_glob ε=%g | Δ_glob ε=%g | pairwise |' % (eg[1], eg[2]),
          '|---|---|---|---|---|---|---|---|']
    for r, R in rows.items():
        for n in ncal:
            for s in ('cp', 'val'):
                v = [x['by_ncal'][n]['selectors'][s] for x in R]
                L.append(f'| {r} | {n} | {s} | {fmt([z["G_regret"] for z in v], 4)} | '
                         + ' | '.join(f'{np.mean([z["G_delta"][i] for z in v]):.3f}' for i in range(len(eg)))
                         + f' | {np.mean([z["pairwise"] for z in v]):.3f} |')

    L += ['\n## Table C: local target (selector vs argmin D(·|x)); floor = Lemma 1 bound for global rules\n',
          '| regime | n_cal | floor ε=0 / %g / %g | ' % (el[1], el[2]) + ' | '.join(f'{s}: regret, Δ ε=0, Δ ε=%g' % el[2] for s in SELECTORS) + ' |',
          '|---' * (len(SELECTORS) + 3) + '|']
    for r, R in rows.items():
        fl = np.mean([x['floor'] for x in R], axis=0)
        for n in ncal:
            cells = []
            for s in SELECTORS:
                v = [x['by_ncal'][n]['selectors'][s] for x in R]
                cells.append(f'{np.mean([z["regret"] for z in v]):.3f}, {np.mean([z["delta"][0] for z in v]):.3f}, '
                             f'{np.mean([z["delta"][2] for z in v]):.3f}')
            L.append(f'| {r} | {n} | {fl[0]:.3f} / {fl[1]:.3f} / {fl[2]:.3f} | ' + ' | '.join(cells) + ' |')
    if 'crs' in SELECTORS and 'level' in rows[next(iter(rows))][0]['by_ncal'][ncal[0]]['selectors']['crs']:
        L += ['\n## Table D: CRS vs the best global rule; local regret, and how often CV picks the constant level\n',
              '| regime | n_cal | heterogeneity (floor ε=%g) | val | CRS-kNN | CRS-forest | P(kNN level = const) | P(forest level = const) |' % el[2],
              '|---|---|---|---|---|---|---|---|']
        for r, R in rows.items():
            fl = np.mean([x['floor'][2] for x in R])
            for n in ncal:
                g = lambda k: fmt([x['by_ncal'][n]['selectors'][k]['regret'] for x in R])
                pc = lambda k: np.mean([x['by_ncal'][n]['selectors'][k]['level'] == 'const' for x in R])
                L.append(f'| {r} | {n} | {fl:.3f} | {g("val")} | {g("knn")} | {g("crs")} | {pc("knn"):.2f} | {pc("crs"):.2f} |')
    return '\n'.join(L)


def checks(rows, st, repro_rows, prev_rows=None, prev_selectors=None):
    res = []
    reps, ncal, el = st['reps'], st['n_cal'], st['eps_local']
    regimes_expected = st.get('regimes', REGIMES)
    for r in regimes_expected:
        R = rows.get(r, [])
        got = [x['rep'] for x in R]
        res.append((f'{r}: replications 0..{reps - 1} all present exactly once',
                    sorted(got) == list(range(reps)), f'{len(got)} rows, {len(set(got))} unique'))
    flat = json.dumps(rows)
    res.append(('no NaN/Infinity in any output', 'NaN' not in flat and 'Infinity' not in flat, ''))
    for r, R in rows.items():
        for n in ncal:
            cov = np.array([x['by_ncal'][str(n)]['cp_coverage'] for x in R])        # (reps, m)
            lo, hi = 1 - st['cp_alpha'], 1 - st['cp_alpha'] + 1 / (n + 1)
            m, se = cov.mean(axis=0), cov.std(axis=0, ddof=1) / np.sqrt(len(R)) + 1e-4
            ok = bool(np.all((m >= lo - 4 * se) & (m <= hi + 4 * se)))       # 126 tests in all: 4 SE keeps false alarms < 1%
            res.append((f'{r} n={n}: split-CP test coverage within [{lo:.3f}, {hi:.3f}] ± 4 SE for all models', ok,
                        f'{m.min():.3f}–{m.max():.3f}'))
            u = [x['by_ncal'][str(n)]['U_joint_coverage'] for x in R]
            um, use = ms(u)
            res.append((f'{r} n={n}: U_M joint coverage >= {lo:.2f} - 4 SE', um >= lo - 4 * use, f'{um:.3f} (SE {use:.3f})'))
            viol = max(x['floor'][i] - x['by_ncal'][str(n)]['selectors'][s]['delta'][i]
                       for x in R for s in ('cp', 'val') for i in range(len(el)))
            res.append((f'{r} n={n}: Lemma 1 holds per replication (global rules never below the floor)',
                        viol <= 1e-12, f'max violation {viol:.2e}'))
    if repro_rows is not None:
        diffs, cnt = [], 0
        for r, R2 in repro_rows.items():
            main = {x['rep']: x for x in rows.get(r, [])}
            for x in R2:
                if x['rep'] in main:
                    cnt += 1
                    diffs.append(float(np.max(np.abs(np.array(x['G']) - np.array(main[x['rep']]['G'])))))
        res.append(('re-run replications reproduce G(M) (max |diff| < 1e-6)', cnt > 0 and max(diffs) < 1e-6,
                    f'{cnt} replications compared, max |diff| = {max(diffs) if diffs else float("nan"):.2e}'))
    if prev_rows is not None:
        worst, cnt, shared = 0.0, 0, set()
        G_checked = not prev_selectors or 'G' in prev_selectors
        for r, Rp in prev_rows.items():
            cur = {x['rep']: x for x in rows.get(r, [])}
            for xp in Rp:
                x = cur.get(xp['rep'])
                if x is None:
                    continue
                cnt += 1
                if G_checked:
                    worst = max(worst, float(np.max(np.abs(np.array(x['G']) - np.array(xp['G'])))))
                for n, bp in xp['by_ncal'].items():
                    if n not in x['by_ncal']:
                        continue
                    for sname, vp in bp['selectors'].items():
                        v = x['by_ncal'][n]['selectors'].get(sname)
                        if v is None or (prev_selectors and sname not in prev_selectors):
                            continue
                        shared.add(sname)
                        worst = max(worst, abs(v['regret'] - vp['regret']),
                                    float(np.max(np.abs(np.array(v['delta']) - np.array(vp['delta'])))))
        res.append((f'selectors shared with the previous run are unchanged (max |diff| < 1e-9; {sorted(shared)})',
                    cnt > 0 and worst < 1e-9, f'{cnt} replications compared, max |diff| = {worst:.2e}'))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=DEFAULT_DIR)
    ap.add_argument('--repro', default=None, help='directory with a re-run subset to compare against')
    ap.add_argument('--prev', default=None, help='directory of an earlier run; shared selectors must match exactly')
    ap.add_argument('--prev-selectors', nargs='+', default=None,
                    help="restrict the --prev comparison to these selectors (add 'G' to also compare G)")
    ap.add_argument('--no-strict', action='store_true')
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.dir, 'meta.json')))
    st = meta['settings']
    regs = st.get('regimes', REGIMES)
    rows = load(args.dir, regs)
    repro = load(args.repro, regs) if args.repro else None
    prev = load(args.prev, regs) if args.prev else None

    print(tables(rows, st))
    with open(os.path.join(args.dir, 'summary.json'), 'w') as f:
        json.dump(dict(meta=meta, summary=summarize(rows, st)), f)
    print(f'\nSummary -> {os.path.relpath(os.path.join(args.dir, "summary.json"))}')

    print('\nAcceptance checks:')
    failed = 0
    for name, ok, detail in checks(rows, st, repro, prev, args.prev_selectors):
        failed += not ok
        print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}')
    print(f'\n{failed} check(s) failed.')
    if failed and not args.no_strict:
        sys.exit(1)


if __name__ == '__main__':
    main()
