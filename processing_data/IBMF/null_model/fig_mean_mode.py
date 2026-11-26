"""fig_data.pdf (Figure 3 of the notes): null-model <d> and d_mode vs mu, with empirical histogram values
(dotted, diamonds), observed pairwise distances of the 5 vectors (grey), and mode - mean (symlog).
Needs results/vectors_by_mu.json (analyze_vectors.py) and results/histograms_empirical.json (empirical_histograms.py)."""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from config import RESULTS_DIR, FIG_DIR, BLUE, ORANGE, AQUA, INK, MUTED, style
agg = json.load(open(os.path.join(RESULTS_DIR, 'vectors_by_mu.json')))
emp = json.load(open(os.path.join(RESULTS_DIR, 'histograms_empirical.json')))
style(); fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3)); off = 0.0012
A = [a for a in agg if a['N'] == 1024]; x = np.array([a['mu'] for a in A])
for a in agg:
    ax[0].scatter([a['mu'] + (off if a['N'] == 8192 else 0)] * len(a['pairs']), a['pairs'], s=9, color=MUTED, alpha=0.5, lw=0, zorder=1)
ax[0].errorbar(x, [a['md'] for a in A], [a['mdsd'] for a in A], color=BLUE, marker='o', ms=4, lw=1.8, capsize=2, zorder=3)
ax[0].errorbar(x, [a['mo'] for a in A], [a['mosd'] for a in A], color=ORANGE, marker='s', ms=3.5, lw=1.4, capsize=2, zorder=4)
for a in agg:
    if a['N'] == 8192:
        ax[0].errorbar(a['mu'] + off, a['md'], a['mdsd'], color=BLUE, marker='o', mfc='white', ms=4, capsize=2, zorder=3)
        ax[0].errorbar(a['mu'] + off, a['mo'], a['mosd'], color=ORANGE, marker='s', mfc='white', ms=3.5, capsize=2, zorder=4)
for N, mfc, o in [(1024, None, 0), (8192, 'white', off)]:
    E = sorted([e for e in emp if e['N'] == N], key=lambda e: e['mu']); xe = np.array([e['mu'] for e in E]) + o
    kw = dict(mfc=mfc) if mfc else {}
    ax[0].plot(xe, [e['mean'] for e in E], color=BLUE, ls=':', lw=1.6, marker='D', ms=4, zorder=5, **kw)
    ax[0].errorbar(xe, [e['mode'] for e in E], yerr=[[e['mode'] - e['mode_lo'] for e in E], [e['mode_hi'] - e['mode'] for e in E]],
                   color=ORANGE, ls=':', lw=1.6, marker='D', ms=4, capsize=2, zorder=5, **kw)
H = [Line2D([], [], color=BLUE, marker='o', lw=1.8, label=r'$\langle d\rangle$, null model'),
     Line2D([], [], color=ORANGE, marker='s', lw=1.4, label=r'$d_{\rm mode}$, null model'),
     Line2D([], [], color=BLUE, marker='D', ls=':', lw=1.6, label=r'$\langle d\rangle$, empirical histogram'),
     Line2D([], [], color=ORANGE, marker='D', ls=':', lw=1.6, label=r'$d_{\rm mode}$, empirical histogram'),
     Line2D([], [], color=MUTED, marker='o', ls='', ms=4, alpha=0.6, label='pairwise $d$, 5 vectors'),
     Line2D([], [], color=INK, marker='o', mfc='white', ls='', ms=4, label='open markers: $N=8192$')]
ax[0].legend(handles=H, frameon=False, fontsize=6.5, loc='upper left')
ax[0].set_xlabel(r'$\mu$'); ax[0].set_ylabel('distance $d$'); ax[0].set_ylim(-0.03, 1.02)
ax[0].set_title('(a) mean and mode of $d$', loc='left', fontsize=9)
ax[1].errorbar(x, [a['diff'] for a in A], [a['diffsd'] for a in A], color=INK, marker='o', ms=4, lw=1.5, capsize=2)
for a in agg:
    if a['N'] == 8192: ax[1].plot(a['mu'] + off, a['diff'], color=INK, marker='o', mfc='white', ms=4)
for N, mfc, o in [(1024, None, 0), (8192, 'white', off)]:
    E = sorted([e for e in emp if e['N'] == N], key=lambda e: e['mu']); xe = np.array([e['mu'] for e in E]) + o
    d = np.array([e['mode'] - e['mean'] for e in E]); lo = np.array([e['mode_lo'] - e['mean'] for e in E]); hi = np.array([e['mode_hi'] - e['mean'] for e in E])
    kw = dict(mfc=mfc) if mfc else {}
    ax[1].errorbar(xe, d, yerr=[d - lo, hi - d], color=AQUA, ls=':', lw=1.6, marker='D', ms=4, capsize=2, **kw)
ax[1].set_yscale('symlog', linthresh=1e-5); ax[1].axhline(0, color=MUTED, lw=0.8); ax[1].set_ylim(-1e-4, 1)
ax[1].legend(handles=[Line2D([], [], color=INK, marker='o', lw=1.5, label='null model'),
                      Line2D([], [], color=AQUA, marker='D', ls=':', lw=1.6, label='empirical histogram')],
             frameon=False, fontsize=7, loc='upper left')
ax[1].set_xlabel(r'$\mu$'); ax[1].set_ylabel(r'$d_{\rm mode}-\langle d\rangle$ (symlog)')
ax[1].set_title('(b) mode minus mean', loc='left', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig_data.pdf'))
