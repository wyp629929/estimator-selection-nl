#!/usr/bin/env python3
"""Generate Figure 1: Misalignment probability Δ across regimes."""
import matplotlib.pyplot as plt
import numpy as np

# Data (from ranking_decision_gap.json, 200 reps)
regimes = ['Linear', 'Mild\nnonlin.', 'Nonlinear', 'Threshold', 'High-dim', 'Heterosc.']
x = np.arange(len(regimes))

width_mean = [0.600, 0.605, 0.530, 0.490, 0.345, 0.605]
width_sd   = [0.490, 0.489, 0.499, 0.500, 0.475, 0.489]
u_mean     = [0.784, 0.786, 0.760, 0.713, 0.757, 0.783]
u_sd       = [0.062, 0.058, 0.061, 0.069, 0.059, 0.062]
baseline   = 0.857

bar_width = 0.30
fig, ax = plt.subplots(figsize=(6.5, 4.5))

bars1 = ax.bar(x - bar_width/2, width_mean, bar_width, yerr=width_sd,
               capsize=2, error_kw={'linewidth': 0.8, 'color': '#4a4a4a'},
               label='CP Width (global)', color='#4C72B0', edgecolor='#2a3f6e', linewidth=0.5)
bars2 = ax.bar(x + bar_width/2, u_mean, bar_width, yerr=u_sd,
               capsize=2, error_kw={'linewidth': 0.8, 'color': '#4a4a4a'},
               label='Regret Bound $U_M$ (per-point)', color='#DD8452', edgecolor='#a65d2e', linewidth=0.5)

ax.axhline(y=baseline, linestyle='--', color='black', linewidth=1.0, label=f'Random baseline ({baseline})')

ax.set_xlabel('Regime', fontsize=12)
ax.set_ylabel('Misalignment Probability $\\Delta$', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(regimes, fontsize=10)
ax.set_ylim(0, 1.25)
ax.legend(fontsize=9, loc='upper right')

fig.tight_layout()
fig.savefig('/Users/wangyaoping/Desktop/ML_Inference_Paper/figures/figure1.png', dpi=300)
fig.savefig('/Users/wangyaoping/Desktop/ML_Inference_Paper/figures/figure1.pdf')
print('Figure 1 saved to figures/figure1.png and figures/figure1.pdf')
