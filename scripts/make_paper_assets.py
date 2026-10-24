#!/usr/bin/env python3
"""Tables and figures of the revised paper, computed directly from results/ (no hand-copied numbers).

Writes paper_assets/tables/*.tex and paper_assets/figures/*.pdf.
Usage: python3 scripts/make_paper_assets.py
"""
import collections
import glob
import json
import os
import shutil

import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(BASE, 'results')
OUT = os.path.join(BASE, 'paper_assets')
TAB, FIG = os.path.join(OUT, 'tables'), os.path.join(OUT, 'figures')
REGIMES = ['linear', 'semiparametric', 'nonlinear', 'threshold', 'highdim', 'heteroscedastic']
M = ['ols', 'ridge', 'lasso', 'rf', 'xgboost', 'lightgbm', 'dnn']
PRETTY = {'ols': 'OLS', 'ridge': 'ridge', 'lasso': 'lasso', 'rf': 'RF', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'dnn': 'DNN'}
DS = {'brazilian_houses': 'brazilian\\_houses', 'kings_county': 'kings\\_county', 'miami_housing': 'miami\\_housing',
      'health_insurance': 'health\\_insurance', 'physiochemical_protein': 'physiochem.\\_protein',
      'naval_propulsion_plant': 'naval\\_propulsion', 'video_transcoding': 'video\\_transcoding',
      'grid_stability': 'grid\\_stability', 'cpu_activity': 'cpu\\_activity', 'fps_benchmark': 'fps\\_benchmark',
      'california_housing': 'california\\_housing', 'white_wine': 'white\\_wine', 'wave_energy': 'wave\\_energy',
      'cps88wages': 'cps88wages', 'pumadyn32nh': 'pumadyn32nh'}
ds = lambda k: DS.get(k, k.replace('_', '\\_'))


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def ms(a):
    a = np.asarray(a, float)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def ms_split(a, rho):
    """Mean and standard error over repeated random splits, corrected for their overlap
    (Nadeau and Bengio 2003): variance (1/J + rho) s^2 with rho = n_test / (n_train + n_cal)."""
    a = np.asarray(a, float)
    return a.mean(), a.std(ddof=1) * np.sqrt(1 / len(a) + rho)


def write(name, lines):
    with open(os.path.join(TAB, name), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', name)


# ── Table 1: global target, E2 ───────────────────────────────────────────────
def table_global():
    L = [r'\begin{tabular}{llcccc}', r'\toprule',
         r' & & \multicolumn{2}{c}{Global regret} & \multicolumn{2}{c}{$\Delta_{\mathrm{glob}}$, tolerance 0.01} \\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}', r'Regime & $n$ & CP width & Val.\ MSE & CP width & Val.\ MSE \\', r'\midrule']
    for r in REGIMES:
        R = jl(os.path.join(RES, 'e2', f'{r}.jsonl'))
        for n in ('250', '4000'):
            g = lambda s, k: np.mean([x['by_ncal'][n]['selectors'][s][k] for x in R])
            d = lambda s: np.mean([x['by_ncal'][n]['selectors'][s]['G_delta'][1] for x in R])
            L.append(f"{r if n == '250' else ''} & {n} & {g('cp', 'G_regret'):.4f} & {g('val', 'G_regret'):.4f} & {d('cp'):.3f} & {d('val'):.3f} \\\\")
        if r != REGIMES[-1]:
            L.append(r'\addlinespace[2pt]')
    L += [r'\bottomrule', r'\end{tabular}']
    write('tab1_global.tex', L)


# ── Table 2: local target, E2 ────────────────────────────────────────────────
def table_local():
    sels = [('val', 'Val.\\ MSE'), ('crs', 'Cond.\\ forest'), ('knn', 'Cond.\\ $k$-NN'), ('lw', 'Norm.\\ CP'), ('U', '$U_M$')]
    L = [r'\begin{tabular}{llccccc}', r'\toprule',
         r'Regime & $n$ & ' + ' & '.join(s for _, s in sels) + r' \\', r'\midrule']
    maxse = 0.0
    for r in REGIMES:
        R = jl(os.path.join(RES, 'e2', f'{r}.jsonl'))
        for n in ('250', '4000'):
            cells = []
            for s, _ in sels:
                m, se = ms([x['by_ncal'][n]['selectors'][s]['regret'] for x in R])
                maxse = max(maxse, se)
                cells.append(f'{m:.3f}')
            L.append(f"{r if n == '250' else ''} & {n} & " + ' & '.join(cells) + r' \\')
        if r != REGIMES[-1]:
            L.append(r'\addlinespace[2pt]')
    L += [r'\bottomrule', r'\end{tabular}']
    write('tab2_local.tex', L)
    print(f'   table 2: largest standard error {maxse:.4f}')


# ── Table 3: what CP width selects on real data ─────────────────────────────
def full(x):
    return str(max(int(n) for n in x['by_n']))


ALPHAS = ('0.05', '0.1', '0.2')


def ratios(R, a):
    return np.array([x['by_n'][full(x)]['cp'][a]['mse'] / x['by_n'][full(x)]['val']['mse'] for x in R])


def table_real_cp():
    """Main Table 3: every binary data set, plus the regression data sets with a costly width, at all three levels."""
    blocks = [
        ('Binary outcomes', [('creditcard', 'e4b', ('inv', '0.1')), ('mammography', 'e4b', ('inv', '0.1')),
                             ('homecredit', 'e4b', ('inv', '0.1')), ('bank', 'e4b', ('inv', '0.2')),
                             ('default', 'e4b', ('shrink', '0.05')), ('adult', 'e4b', ('shrink', '0.05')),
                             ('pima', 'e4b', None)]),
        ('Nearly binary outcome', [('health_insurance', 'e7/e4b', ('shrink', '0.1'))]),
        ('Continuous outcomes', [('brazilian_houses', 'e7/e4b', ('m1p', None)), ('cpu_activity', 'e7/e4b', ('m1p', None)),
                                 ('fifa', 'e7/e4b', ('m1p', None)), ('supercond', 'e4b', None)]),
    ]
    L = [r'\begin{tabular}{lcccccl}', r'\toprule',
         r' & & & \multicolumn{3}{c}{CP/val risk ratio} & \\', r'\cmidrule(lr){4-6}',
         r'Data set & Prevalence & $e$ (range) & $\alpha=0.05$ & $0.1$ & $0.2$ & Evidence \\']
    named = set()
    for title, rows in blocks:
        L += [r'\midrule', f'\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\']
        for k, d, ev in rows:
            named.add(k)
            R = jl(os.path.join(RES, d, f'{k}.jsonl'))
            e = [x['by_n'][full(x)] for x in R]
            binary = R[0].get('binary') and e[0].get('e') is not None
            prev = f"{R[0]['prevalence_cal']:.3f}" if binary else '---'
            er = (f"{min(min(z['e']) for z in e):.3f}--{max(max(z['e']) for z in e):.3f}" if binary else '---')
            cells = [f'{ratios(R, a).mean():.3f}' for a in ALPHAS]
            if ev is None:
                evs = '---'
            elif ev[0] == 'inv':
                a = ev[1]
                evs = f"(b) {np.mean([np.mean(z['cp'][a]['pos_invisible']) for z in e]):.2f} [{a}]"
            elif ev[0] == 'shrink':
                a = ev[1]
                dd = json.load(open(os.path.join(RES, 'diag', f'shrinkage_{k}_a{float(a)}.json')))
                dis = [r for r in dd if r['cp'] != r['val']]
                ok = sum(r['sd_c'] < r['sd_v'] and r['mse_c'] > r['mse_v'] and r['body_c'] >= r['body_v'] for r in dis)
                evs = f"S {ok}/{len(dis)} [{a}]"
            else:
                cells_m = [z['cp'][aa]['m1prime'] for z in e for aa in ALPHAS if z['cp'][aa].get('m1prime')]
                sel = [m for m in cells_m if m['mse_ratio_A'] > 1.01]
                evs = f"C {sum(m['bulk_d'] <= 0 for m in sel)}/{len(sel)}"
            L.append(f"{ds(k)} & {prev} & {er} & " + ' & '.join(cells) + f" & {evs} \\\\")
    other = [p for p in sorted(glob.glob(os.path.join(RES, 'e7', 'e4b', '*.jsonl'))) if os.path.basename(p)[:-6] not in named]
    med = {a: np.median([ratios(jl(p), a).mean() for p in other]) for a in ALPHAS}
    L += [r'\midrule',
          f"\\multicolumn{{3}}{{l}}{{Other {len(other)} CTR23 data sets (median)}} & " + ' & '.join(f'{med[a]:.3f}' for a in ALPHAS) + r' & \\',
          r'\bottomrule', r'\end{tabular}']
    write('tab3_real_cp.tex', L)


def cp_all_header():
    return [r'\begin{tabular}{lc' + 'cc' * 3 + '}', r'\toprule',
            r' & & \multicolumn{2}{c}{$\alpha=0.05$} & \multicolumn{2}{c}{$\alpha=0.1$} & \multicolumn{2}{c}{$\alpha=0.2$} \\',
            r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
            r'Data set & $n_{\mathrm{cal}}$ & ratio & $P(\text{CP}=\text{val})$ & ratio & $P(\text{CP}=\text{val})$ & ratio & $P(\text{CP}=\text{val})$ \\', r'\midrule']


def num(x):
    """Three decimals; values of 1000 or more (a few splits with runaway predictions) in powers of ten."""
    if x < 1000:
        return f'{x:.3f}'
    m, e = f'{x:.1e}'.split('e')
    return f'${m}\\times10^{{{int(e)}}}$'


def cp_all_rows(d):
    L = []
    meta = json.load(open(os.path.join(RES, d, 'meta.json')))
    for k in meta['settings']['datasets']:
        R = jl(os.path.join(RES, d, f'{k}.jsonl'))
        cells = []
        for a in ALPHAS:
            r = ratios(R, a)
            agree = np.mean([x['by_n'][full(x)]['cp'][a]['choice'] == x['by_n'][full(x)]['val']['choice'] for x in R])
            cells += [f'{num(r.mean())} ({np.median(r):.3f})', f'{agree:.2f}']
        L.append(f"{ds(k)} & {R[0]['n_cal']} & " + ' & '.join(cells) + r' \\')
    return L


def table_e8_cp_all():
    """Appendix Table B4: E8, the 31 datasets of the Grinsztajn et al. (2022) benchmark, all three levels.
    Means can be dominated by one split (miniboone); medians are given beside them."""
    rows = cp_all_rows('e8/e4b')
    n_bin = 13                                            # 13 binary datasets first, then 18 regression
    L = cp_all_header() + rows[:n_bin] + [r'\midrule'] + rows[n_bin:] + [r'\bottomrule', r'\end{tabular}']
    write('tabA_e8_cp_all.tex', L)


def table_e9a():
    """Appendix Table B6: E9a, the 20 binary data sets with every candidate's predictions clipped to [0, 1]
    (same refit): mean CP/val risk ratio unclipped -> clipped, and the share of test predictions outside [0, 1]."""
    i = M.index
    ratio = lambda x, kind, a: x[kind]['model_mse'][i(x[kind]['cp'][a]['choice'])] / x[kind]['model_mse'][i(x[kind]['val'])]
    new = json.load(open(os.path.join(RES, 'e8', 'e4b', 'meta.json')))['settings']['datasets'][:13]
    L = [r'\begin{tabular}{lcccc}', r'\toprule',
         r'Data set & Outside $[0,1]$ & $\alpha=0.05$ & $\alpha=0.1$ & $\alpha=0.2$ \\', r'\midrule']
    for k in ['pima', 'homecredit', 'creditcard', 'mammography', 'bank', 'default', 'adult'] + [None] + new:
        if k is None:
            L.append(r'\midrule')
            continue
        R = jl(os.path.join(RES, 'e9a', f'{k}.jsonl'))
        cells = [f"{num(np.mean([ratio(x, 'raw', a) for x in R]))} $\\to$ {num(np.mean([ratio(x, 'clipped', a) for x in R]))}" for a in ALPHAS]
        L.append(f"{ds(k)} & {np.mean([x['outside'] for x in R]):.3f} & " + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    write('tabA_e9a_clipped.tex', L)


def table_real_cp_all():
    """Appendix Table B3: all 28 data sets, all three levels."""
    L = [r'\begin{tabular}{lc' + 'cc' * 3 + '}', r'\toprule',
         r' & & \multicolumn{2}{c}{$\alpha=0.05$} & \multicolumn{2}{c}{$\alpha=0.1$} & \multicolumn{2}{c}{$\alpha=0.2$} \\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
         r'Data set & $n_{\mathrm{cal}}$ & ratio & $P(\text{CP}=\text{val})$ & ratio & $P(\text{CP}=\text{val})$ & ratio & $P(\text{CP}=\text{val})$ \\', r'\midrule']
    for d in ('e4b', 'e7/e4b'):
        meta = json.load(open(os.path.join(RES, d, 'meta.json')))
        for k in meta['settings']['datasets']:
            R = jl(os.path.join(RES, d, f'{k}.jsonl'))
            cells = []
            for a in ALPHAS:
                r = ratios(R, a)
                agree = np.mean([x['by_n'][full(x)]['cp'][a]['choice'] == x['by_n'][full(x)]['val']['choice'] for x in R])
                cells += [f'{r.mean():.3f} ({np.median(r):.3f})', f'{agree:.2f}']
            L.append(f"{ds(k)} & {R[0]['n_cal']} & " + ' & '.join(cells) + r' \\')
        if d == 'e4b':
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}']
    write('tabA_real_cp_all.tex', L)


# ── Table 4: local selection on real data ───────────────────────────────────
def local_rows(dirs=('e4', 'e7/e4')):
    out = []
    for d in dirs:
        meta = json.load(open(os.path.join(RES, d, 'meta.json')))
        for k in meta['settings']['datasets']:
            R = jl(os.path.join(RES, d, f'{k}.jsonl'))
            val = np.array([x['selectors']['val']['mse'] for x in R])
            c = np.array([x['selectors']['crs']['mse'] for x in R]) / val
            kn = np.array([x['selectors']['knn']['mse'] for x in R]) / val
            best = np.array([min(x['model_mse']) for x in R]) / val
            cp = np.array([x['selectors']['cp']['mse'] for x in R]) / val
            const = np.mean([x['selectors']['crs']['level'] == 'const' for x in R])
            rho = R[0]['n_test'] / (R[0]['n_train'] + R[0]['n_cal'])
            se = lambda a: ms_split(a, rho)
            out.append(dict(k=k, n=R[0]['n_cal'], crs=se(c), med=np.median(c), knn=se(kn), best=se(best), cp=se(cp), const=const))
    return out


def table_real_local():
    rows = local_rows()
    sig = [r for r in rows if 1 - r['crs'][0] > 2 * r['crs'][1]]
    worse = [r for r in rows if r['crs'][0] - 1 > 0.01]
    L = [r'\begin{tabular}{lccccc}', r'\toprule',
         r'Data set & $n_{\mathrm{cal}}$ & Cond.\ forest & Cond.\ $k$-NN & Best single & $P(\text{const.})$ \\', r'\midrule',
         r'\multicolumn{6}{l}{\emph{Conditional selection better than validation MSE by more than 2 SE}} \\']
    fmt = lambda t: f'{t[0]:.3f} ({t[1]:.3f})'
    for r in sorted(sig, key=lambda r: r['crs'][0]):
        L.append(f"{ds(r['k'])} & {r['n']} & {fmt(r['crs'])} & {fmt(r['knn'])} & {fmt(r['best'])} & {r['const']:.2f} \\\\")
    L.append(r'\multicolumn{6}{l}{\emph{Mean ratio above 1.01}} \\')
    for r in sorted(worse, key=lambda r: r['crs'][0]):
        L.append(f"{ds(r['k'])} & {r['n']} & {fmt(r['crs'])} & {fmt(r['knn'])} & {fmt(r['best'])} & {r['const']:.2f} \\\\")
    rest = [r for r in rows if r not in sig and r not in worse]
    L += [r'\midrule',
          f"\\multicolumn{{6}}{{l}}{{Other {len(rest)} data sets: forest ratios between {min(r['crs'][0] for r in rest):.3f} and "
          f"{max(r['crs'][0] for r in rest):.3f}.}} \\\\", r'\bottomrule', r'\end{tabular}']
    write('tab4_real_local.tex', L)
    # appendix: all data sets
    A = [r'\begin{tabular}{lcccccc}', r'\toprule',
         r'Data set & $n_{\mathrm{cal}}$ & CP width & Cond.\ forest & (median) & Cond.\ $k$-NN & Best single \\', r'\midrule']
    for r in rows:
        A.append(f"{ds(r['k'])} & {r['n']} & {fmt(r['cp'])} & {fmt(r['crs'])} & {r['med']:.3f} & {fmt(r['knn'])} & {fmt(r['best'])} \\\\")
    A += [r'\bottomrule', r'\end{tabular}']
    write('tabA_real_local_all.tex', A)
    print(f'   table 4: {len(sig)} significant gains, {len(worse)} above 1.01, {len(rest)} others')


def table_e8_local_all():
    """Appendix Table B5: E8 local selection, same columns as Table B3."""
    fmt = lambda t: f'{num(t[0])} ({num(t[1])})'
    A = [r'\begin{tabular}{lcccccc}', r'\toprule',
         r'Data set & $n_{\mathrm{cal}}$ & CP width & Cond.\ forest & (median) & Cond.\ $k$-NN & Best single \\', r'\midrule']
    for i, r in enumerate(local_rows(('e8/e4',))):
        if i == 13:                                       # 13 binary datasets first, then 18 regression
            A.append(r'\midrule')
        A.append(f"{ds(r['k'])} & {r['n']} & {fmt(r['cp'])} & {fmt(r['crs'])} & {r['med']:.3f} & {fmt(r['knn'])} & {fmt(r['best'])} \\\\")
    A += [r'\bottomrule', r'\end{tabular}']
    write('tabA_e8_local_all.tex', A)


# ── Appendix: robustness variants (E3, E6) ──────────────────────────────────
def table_robust():
    L = [r'\begin{tabular}{llcccc}', r'\toprule',
         r' & & \multicolumn{2}{c}{Global regret} & \multicolumn{2}{c}{Local regret} \\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}', r'Variant & Regime & CP width & Val.\ MSE & Val.\ MSE & Cond.\ forest \\', r'\midrule']
    variants = [('correlated $X$', 'e3/corr'), ('tuned estimators', 'e3/tuned'), ('$p=50$ / $400$', 'e3/highp'),
                ('Gaussian noise', 'e6/noise_normal'), ('$t_3$ noise', 'e6/noise_t3'), ('skewed noise', 'e6/noise_skew')]
    for name, d in variants:
        first = True
        for p in sorted(glob.glob(os.path.join(RES, d, '*.jsonl'))):
            R = jl(p); r = os.path.basename(p)[:-6]
            g = lambda s: np.mean([x['by_ncal']['4000']['selectors'][s]['G_regret'] for x in R])
            l = lambda s: np.mean([x['by_ncal']['4000']['selectors'][s]['regret'] for x in R])
            L.append(f"{name if first else ''} & {r} & {g('cp'):.4f} & {g('val'):.4f} & {l('val'):.3f} & {l('crs'):.3f} \\\\")
            first = False
        L.append(r'\addlinespace[2pt]')
    L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
    write('tabA_robust.tex', L)


# ── Figure 1: E5 dose-response ──────────────────────────────────────────────
def figure_e5():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = json.load(open(os.path.join(RES, 'e5', 'e5_mechanism.json')))
    C = [c for c in d['cells'] if c['noise'] == 'normal' and c['alpha'] == 0.1 and c['b'] == 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    colors = {250: '#9ecae1', 1000: '#4292c6', 4000: '#2171b5', 16000: '#08306b'}
    for ax, key, title in ((axes[0], 'P_cp1', '(a) CP width'), (axes[1], 'P_val1', '(b) Validation MSE')):
        for n in sorted({c['n'] for c in C}):
            cc = sorted((c for c in C if c['n'] == n), key=lambda c: c['mult'])
            for side, lab in ((lambda m: m < 1, f'$n={n}$'), (lambda m: m > 1, None)):   # no data at p = p_crit
                part = [c for c in cc if side(c['mult'])]
                ax.plot([c['mult'] for c in part], [c[key] for c in part], 'o-', color=colors[n], label=lab, ms=4)
        ax.axvline(1.0, ls=':', color='k', lw=1)
        ax.set_xscale('log'); ax.set_xticks([0.25, 0.5, 1, 2, 3]); ax.set_xticklabels(['0.25', '0.5', '1', '2', '3'])
        ax.set_xlabel(r'$p\,/\,p_{\mathrm{crit}}$'); ax.set_title(title)
    axes[0].set_ylabel('P(selects the globally worse model)')
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig1_e5_mechanism.pdf'))
    print('wrote fig1_e5_mechanism.pdf')


if __name__ == '__main__':
    os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    table_global(); table_local(); table_real_cp(); table_real_cp_all(); table_real_local(); table_robust(); figure_e5()
    table_e8_cp_all(); table_e8_local_all(); table_e9a()
    shutil.copy(os.path.join(BASE, 'figures', 'e1_mirror.pdf'), os.path.join(FIG, 'fig2_e1_barrier.pdf'))
    print('copied fig2_e1_barrier.pdf')
