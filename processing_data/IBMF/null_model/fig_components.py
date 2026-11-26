"""fig_Q.pdf: empirical distribution of the components (pooled over the 5 vectors) vs moment-matched Beta,
with the point masses at 0 and 1 shown separately (height = mass / bin width)."""
import glob, re, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import beta as Bd
from null_model import beta_params
from config import VECTORS_DIR, FIG_DIR, BLUE, ORANGE, INK, MUTED, style
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from panel_utils import save_panels   # one image per panel, for LaTeX subfigures
data = {}
for f in glob.glob(os.path.join(VECTORS_DIR, '*.txt')):
    mu = float(re.search(r'mu_([0-9.]+)_', f).group(1)); N = int(re.search(r'_N_(\d+)_', f).group(1))
    data.setdefault((mu, N), []).append(np.loadtxt(f, comments='#')[:, 1])
keys = [(0.36, 1024), (0.37, 1024), (0.38, 1024), (0.38, 8192)]
style(); fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6)); eps = 1e-6; edges = np.linspace(0, 1, 41); w = edges[1]
for ax, k, lab in zip(axs.flat, keys, 'abcd'):
    x = np.concatenate(data[k]); n = len(x)
    ab = np.array([beta_params(v.mean(), v.var(ddof=1))[:2] for v in data[k]]); a, b = ab.mean(0)
    p0, p1 = (x < eps).mean(), (x > 1 - eps).mean(); inner = x[(x >= eps) & (x <= 1 - eps)]
    h, _ = np.histogram(inner, bins=edges)
    ax.bar(edges[:-1] + w / 2, h / n / w, width=w * 0.92, color=MUTED, alpha=0.55, lw=0, label='data, interior values')
    t = np.linspace(1e-4, 1 - 1e-4, 800); ax.plot(t, Bd.pdf(t, a, b), color=BLUE, lw=2, label='moment-matched Beta density')
    for xa, pa, pb in [(0, p0, Bd.cdf(w, a, b)), (1, p1, Bd.sf(1 - w, a, b))]:
        ax.plot([xa, xa], [1e-3, pa / w], color=ORANGE, lw=3, solid_capstyle='butt')
        ax.text(xa + (0.03 if xa == 0 else -0.03), pa / w * 1.15, f'atom {100*pa:.0f}%\nBeta bin {100*pb:.0f}%',
                ha='left' if xa == 0 else 'right', va='bottom', fontsize=7, color=INK)
    ax.set_yscale('log'); ax.set_ylim(3e-2, 40); ax.set_xlim(-0.02, 1.02)
    ax.set_title(f'({lab}) ' + fr'$\mu={k[0]:.2f}$, $N={k[1]}$:  $\hat\alpha$={a:.2f}, $\hat\beta$={b:.2f}', loc='left', fontsize=9)
    ax.set_xlabel('$m$'); ax.set_ylabel('density')
axs[0, 0].plot([], [], color=ORANGE, lw=3, label='point mass at 0 or 1 (mass / bin width)')
h_, l_ = axs[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, fontsize=8, loc='lower center', ncol=3)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(FIG_DIR, 'fig_Q.pdf')); save_panels(fig, os.path.join(FIG_DIR, 'fig_Q.pdf'))
