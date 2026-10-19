#!/usr/bin/env python3
"""E1 report: figure, summary table and acceptance checks for results/e1_mirror.json.

Usage:
  python3 scripts/e1_report.py                         # full results
  python3 scripts/e1_report.py --in results/e1_mirror_quick.json --no-strict
Exit code 1 if any acceptance check fails (unless --no-strict).
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(BASE_DIR, '..', 'results', 'e1', 'e1_mirror.json')
FIG_DIR = os.path.join(BASE_DIR, '..', 'figures')

LABELS = {'cp': 'CP width (residual-only, global)', 'val': 'Validation MSE (residual-only, global)',
          'bnd': 'Boundary rule (residual-only, $x$-dependent)', 'knn': '$k$-NN risk regression (conditional)',
          'bnd_orc': 'Boundary rule, true orientation (infeasible)'}
COLORS = {'cp': '#4C72B0', 'val': '#55A868', 'bnd': '#DD8452', 'knn': '#C44E52', 'bnd_orc': '#8C8C8C'}
RESIDUAL_ONLY = ['cp', 'val', 'bnd']


def checks(res, settings):
    """Acceptance checks. Returns list of (name, passed, detail)."""
    out = []
    eps, sels, ns = settings['eps'], settings['selectors'], settings['n_cal']
    reps = settings['reps']
    tol_ks = 0.05 + 3 * np.sqrt(0.05 * 0.95 / reps)            # 3-sigma binomial band
    for c, s in res.items():
        ks = s['ks_reject_rate_0.05']
        out.append((f'{c}: KS reject rate <= {tol_ks:.3f} (residual laws equal across worlds, Thm 4(i))',
                    max(ks) <= tol_ks, f'{[round(v, 3) for v in ks]}'))
        eq = s['equivariance_maxdiff']
        out.append((f'{c}: OLS/RF reflection equivariance max |diff| < 0.1', max(eq) < 0.1, f'{[round(v, 4) for v in eq]}'))
        for e, ep in enumerate(eps):
            bound = s['theorem3_bound'][e]
            for sel in RESIDUAL_ONLY:
                k = sels.index(sel)
                worst = min(s['delta_avg_worlds_mean'][i][k][e] + 3 * s['delta_avg_worlds_se'][i][k][e] for i in range(len(ns)))
                out.append((f'{c}: {sel} avg Delta_eps={ep} >= Thm 4 bound {bound:.3f} (up to 3 SE) at every n',
                            worst >= bound, f'min(mean+3SE) = {worst:.3f}'))
        k = sels.index('knn')
        r0, r1 = s['regret_avg_worlds_mean'][0][k], s['regret_avg_worlds_mean'][-1][k]
        out.append((f'{c}: kNN regret decreases from smallest to largest n', r1 < r0, f'{r0:.3f} -> {r1:.3f}'))
    return out


def table(res, settings):
    sels, ns, eps = settings['selectors'], settings['n_cal'], settings['eps']
    e5 = eps.index(0.5) if 0.5 in eps else len(eps) - 1
    lines = []
    for c, s in res.items():
        lines.append(f'\n### {c}  (kappa_eps/2 = {[round(v, 3) for v in s["theorem3_bound"]]} for eps={eps}; '
                     f'Lemma 1 floor = {[round(v, 3) for v in s["lemma1_floor_mean"]]}; '
                     f'global MSE excl. sigma^2 [ols, rf] = {[round(v, 3) for v in s["global_mse_mean"][0]]})\n')
        lines.append('| n | selector | regret (avg worlds) | Delta exact P / PT | Delta_eps=%g (avg) |' % eps[e5])
        lines.append('|---|---|---|---|---|')
        for i, n in enumerate(ns):
            for k, sel in enumerate(sels):
                lines.append(f'| {n} | {sel} | {s["regret_avg_worlds_mean"][i][k]:.3f} ± {s["regret_avg_worlds_se"][i][k]:.3f} '
                             f'| {s["delta_mean"][0][i][k][0]:.3f} / {s["delta_mean"][1][i][k][0]:.3f} '
                             f'| {s["delta_avg_worlds_mean"][i][k][e5]:.3f} ± {s["delta_avg_worlds_se"][i][k][e5]:.3f} |')
    return '\n'.join(lines)


def figure(res, settings, config, path_stem):
    s = res[config]
    sels, ns, eps = settings['selectors'], np.array(settings['n_cal']), settings['eps']
    e5 = eps.index(0.5) if 0.5 in eps else len(eps) - 1
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax = axes[0]
    for k, sel in enumerate(sels):
        m = np.array([s['regret_avg_worlds_mean'][i][k] for i in range(len(ns))])
        se = np.array([s['regret_avg_worlds_se'][i][k] for i in range(len(ns))])
        ls = '--' if sel == 'bnd_orc' else '-'
        ax.errorbar(ns, m, yerr=2 * se, color=COLORS[sel], ls=ls, marker='o', ms=3, capsize=2, label=LABELS[sel])
    ax.set_xscale('log'); ax.set_xlabel('calibration size $n$'); ax.set_ylabel('local regret (average over $P$, $P^T$)')
    ax.set_title('(a) Local regret')

    ax = axes[1]
    for k, sel in enumerate(sels):
        m = np.array([s['delta_avg_worlds_mean'][i][k][e5] for i in range(len(ns))])
        ls = '--' if sel == 'bnd_orc' else '-'
        ax.plot(ns, m, color=COLORS[sel], ls=ls, marker='o', ms=3)
    ax.axhline(s['theorem3_bound'][e5], color='black', lw=1, ls=':', label='Theorem 4 bound (residual-only)')
    ax.set_xscale('log'); ax.set_xlabel('calibration size $n$'); ax.set_ylabel(r'$\Delta_\epsilon$ (average over $P$, $P^T$)')
    ax.set_title(r'(b) Misalignment beyond $\epsilon=%g$' % eps[e5]); ax.legend(fontsize=8, loc='center right')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ['pdf', 'png']:
        fig.savefig(f'{path_stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default=DEFAULT_IN)
    ap.add_argument('--fig-config', default=None, help='config shown in the figure (default: d5 if present)')
    ap.add_argument('--no-strict', action='store_true')
    args = ap.parse_args()
    data = json.load(open(args.inp))
    res, settings = data['results'], data['settings']

    print(table(res, settings))
    cfg = args.fig_config or ('d5' if 'd5' in res else next(iter(res)))
    if os.path.abspath(args.inp) == os.path.abspath(DEFAULT_IN):
        os.makedirs(FIG_DIR, exist_ok=True)
        stem = os.path.join(FIG_DIR, 'e1_mirror')
    else:                                   # non-default input: keep the figure next to the JSON
        stem = os.path.splitext(os.path.abspath(args.inp))[0]
    figure(res, settings, cfg, stem)
    print(f'\nFigure ({cfg}) -> {os.path.relpath(stem)}.pdf / .png')

    print('\nAcceptance checks:')
    failed = 0
    for name, ok, detail in checks(res, settings):
        failed += not ok
        print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}')
    print(f'\n{failed} check(s) failed.')
    if failed and not args.no_strict:
        sys.exit(1)


if __name__ == '__main__':
    main()
