#!/usr/bin/env python3
"""Generate Figure 2: Schematic of bias crossing in the threshold regime."""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 300)
# Bias of a linear model on threshold DGP: large misspecification on x1>0 side
bias_linear = 0.5 + 0.8 * (x > 0) * x
# Bias of a tree model: small bias near discontinuities, larger elsewhere
bias_tree = 0.3 * np.abs(x) + 0.2 * np.exp(-(x - 0.5)**2 / 0.1)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(x, bias_linear**2, 'b-', linewidth=2, label='Linear model $\\mathrm{bias}^2(x)$')
ax.plot(x, bias_tree**2, 'orange', linewidth=2, label='Tree model $\\mathrm{bias}^2(x)$')
ax.axvline(0, color='gray', linestyle=':', alpha=0.7)
ax.text(0.05, 1.5, 'Crossing\nregion', fontsize=8, color='gray')
ax.set_xlabel('$x_1$ (threshold variable)', fontsize=10)
ax.set_ylabel('Squared bias', fontsize=10)
ax.set_title('Bias Crossing in the Threshold Regime', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 2.5)
fig.tight_layout()
fig.savefig('/Users/wangyaoping/Desktop/ML_Inference_Paper/figures/figure2_bias_crossing.pdf')
fig.savefig('/Users/wangyaoping/Desktop/ML_Inference_Paper/figures/figure2_bias_crossing.png', dpi=150)
print('Figure saved')
