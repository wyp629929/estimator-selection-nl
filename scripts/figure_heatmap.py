#!/usr/bin/env python3
"""Figure B: Surrogate comparison heatmap."""
import matplotlib.pyplot as plt
import numpy as np

surrogates = ['Split CP Width', 'U$_M$ (regret)', 'CQR', 'Val-MSE', 'CV+']
regimes = ['Linear', 'Mild\nnonlin.', 'Nonlinear', 'Threshold', 'High-dim', 'Heterosc.']

data = np.array([
    [0.600, 0.605, 0.530, 0.490, 0.345, 0.605],  # Width
    [0.784, 0.786, 0.760, 0.713, 0.757, 0.783],  # U_M
    [0.885, 0.875, 1.000, 1.000, 1.000, 0.855],  # CQR
    [0.445, 0.485, 0.715, 0.155, 0.145, 0.485],  # Val-MSE
    [0.482, 0.427, 0.834, 0.123, 0.043, 0.454],  # CV+
])

fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(data, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')

ax.set_xticks(range(len(regimes)))
ax.set_xticklabels(regimes, fontsize=9)
ax.set_yticks(range(len(surrogates)))
ax.set_yticklabels(surrogates, fontsize=9)

for i in range(len(surrogates)):
    for j in range(len(regimes)):
        val = data[i, j]
        color = 'white' if val > 0.65 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8, color=color)

ax.set_title('Misalignment Probability $\\Delta$ Across Surrogates and Regimes', fontsize=11)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('$\\Delta$ (misalignment probability)', fontsize=9)

fig.tight_layout()
fig.savefig('/Users/wangyaoping/Desktop/ML_Inference_Paper/figures/figure3_heatmap.png', dpi=200)
fig.savefig('/Users/wangyaoping/Desktop/ML_Inference_Paper/figures/figure3_heatmap.pdf')
print('Figure B (heatmap) saved')
