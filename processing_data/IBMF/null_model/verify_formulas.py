"""Checks the analytic formulas against the exact P(d^2) (numerical inversion) and Monte Carlo,
and draws fig_distance.pdf (closed form of I(D) vs sampling; P(d^2) exact/CLT/Edgeworth/MC)."""
import os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from null_model import beta_ED2n, exact_density_d2, I_closed_form
from config import FIG_DIR, BLUE, ORANGE, INK, MUTED, style
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from panel_utils import save_panels   # one image per panel, for LaTeX subfigures
rng = np.random.default_rng(1)

def cum(a, b):
    M2, M4, M6 = (beta_ED2n(a, b, n) for n in (1, 2, 3))
    return M2, M4 - M2**2, M6 - 3 * M4 * M2 + 2 * M2**3

print("alpha beta N | var exact/formula | mode(d2) exact/formula | mode(d) exact/formula | <d> exact/formula | <d> MC")
for a, b in [(2.5, 3.2), (0.7, 1.5)]:
    mu, k2, k3 = cum(a, b)
    for N in [5, 20, 100]:
        s = np.linspace(1e-4, min(1, 4 * mu), 1500); P = exact_density_d2(a, b, N, s)
        norm = np.trapezoid(P, s); mean = np.trapezoid(s * P, s) / norm
        var = np.trapezoid((s - mean)**2 * P, s) / norm
        v = k2 / N
        X = rng.beta(a, b, (200000, N)); Y = rng.beta(a, b, (200000, N)); d2 = ((X - Y)**2).mean(1)
        print(f"{a} {b} {N:3d} | {var:.3e}/{v:.3e} | {s[np.argmax(P)]:.4f}/{mu - k3/(2*N*k2):.4f} | "
              f"{np.sqrt(s)[np.argmax(2*np.sqrt(s)*P)]:.4f}/{np.sqrt(mu + k2/(2*N*mu) - k3/(2*N*k2)):.4f} | "
              f"{np.trapezoid(np.sqrt(s)*P, s)/norm:.4f}/{np.sqrt(mu)*(1 - v/(8*mu**2)):.4f} | {np.sqrt(d2).mean():.4f}")

style(); fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for (a, b), c in zip([(2.5, 3.2), (0.7, 1.5)], [BLUE, ORANGE]):
    tau = np.linspace(0.005, 0.995, 200); I = [I_closed_form(t, a, b) for t in tau]
    D = np.abs(rng.beta(a, b, 10**6) - rng.beta(a, b, 10**6))
    ax[0].hist(D, bins=100, range=(0, 1), density=True, color=c, alpha=0.25, lw=0)
    ax[0].plot(tau, 2 * np.array(I), color=c, lw=2, label=fr'$\alpha={a},\ \beta={b}$')
ax[0].set_xlabel(r'$D=|m-\hat m|$'); ax[0].set_ylabel(r'density of $|D|$, $2I(D)$'); ax[0].legend(frameon=False)
ax[0].set_title('(a) closed form vs. sampling', loc='left', fontsize=9)
a, b, N = 2.5, 3.2, 10
mu, k2, k3 = cum(a, b); v = k2 / N; g = k3 / k2**1.5 / np.sqrt(N)
s = np.linspace(1e-4, 0.25, 700); P = exact_density_d2(a, b, N, s)
z = (s - mu) / np.sqrt(v); G = np.exp(-z**2 / 2) / np.sqrt(2 * np.pi * v)
X = rng.beta(a, b, (400000, N)); Y = rng.beta(a, b, (400000, N)); d2 = ((X - Y)**2).mean(1)
ax[1].hist(d2, bins=120, range=(0, 0.25), density=True, color=MUTED, alpha=0.3, lw=0, label='Monte Carlo')
ax[1].plot(s, P, color=INK, lw=2, label='exact (numerical inversion)')
ax[1].plot(s, G, color=BLUE, lw=1.5, ls='--', label='Gaussian (CLT)')
ax[1].plot(s, G * (1 + g / 6 * (z**3 - 3 * z)), color=ORANGE, lw=1.5, label='Edgeworth, 1st order')
ax[1].axvline(mu, color=MUTED, lw=1, ls=':')
ax[1].set_xlabel(r'$d^2$'); ax[1].set_ylabel(r'$P(d^2)$'); ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title(fr'(b) $P(d^2)$, $\alpha={a},\beta={b},N={N}$', loc='left', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig_distance.pdf')); save_panels(fig, os.path.join(FIG_DIR, 'fig_distance.pdf'))
