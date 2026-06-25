#!/usr/bin/env python3
"""Figure A: Theory validation — Example 1 lower bound vs empirical Δ."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

r_vals = np.linspace(0, 3, 300)
# Δ ≥ min{P(|X| > r), P(|X| < r)} for X ~ N(0,1), x0=0
delta_lower = np.minimum(2 * (1 - stats.norm.cdf(r_vals)),  # P(|X| > r)
                         2 * stats.norm.cdf(r_vals) - 1)     # P(|X| < r)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(r_vals, delta_lower, 'k-', linewidth=2, label='Thm 1 lower bound $\\Delta \\geq \\min\\{P(|X|>r), P(|X|<r)\\}$')

# Empirical Δ from threshold regime (Width)
# From Table 2, threshold Width Δ = 0.490
# Example 1 with moderate crossing radius predicts Δ ≥ 0.5 at worst case
# The empirical value 0.490 is within the predicted range
ax.axhline(y=0.490, color='C0', linestyle='--', linewidth=1.5, label='Threshold Width $\\Delta = 0.490$')
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Maximal lower bound (0.5)')

ax.fill_between(r_vals, delta_lower, 1.0, alpha=0.08, color='red', label='Forbidden region ($\\Delta <$ bound)')
ax.fill_between(r_vals, 0, delta_lower, alpha=0.08, color='green', label='Achievable region')

ax.set_xlabel('Crossing radius $r = |b_0/k|$', fontsize=11)
ax.set_ylabel('Misalignment probability $\\Delta$', fontsize=11)
ax.set_title('Theoretical Lower Bound vs. Empirical Misalignment', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 3)

fig.tight_layout()
fig.savefig('/Users/wangyaoping/Desktop/ML_Inference_Paper/figures/figure4_theory_bound.png', dpi=200)
fig.savefig('/Users/wangyaoping/Desktop/ML_Inference_Paper/figures/figure4_theory_bound.pdf')
print('Figure A (theory bound) saved')
