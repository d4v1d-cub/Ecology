"""Comparison of the fixed points reached by the IBMF iteration and by the numerical integration of the Lotka-Volterra
equations (LV_ODE_countFP --method ibmf / --method ode) on the same graph and from the same initial conditions.

Inputs: two directories with the outputs of LV_ODE_countFP (binary file of the fixed points and per-run log).
Output: printed statistics, a JSON file with them, and a figure:
  (a) phi_ODE vs phi_IBMF for the fixed points found by both methods (phi = fraction of the n_init runs),
  (b) run-weighted histograms of the mean abundance m_bar of the fixed point reached, one curve per method,
  (c) stability of the fixed points: largest field of an extinct species vs smallest real part of the eigenvalues of
      D_S M_SS (S = surviving species, D_S = diag(m_S), M = I + interaction matrix restricted to S).
A fixed point m is stable for dn_i/dt = n_i (1 - n_i - sum_j a_ij n_j) if every extinct species has field
1 - sum_j a_ij m_j < 0 and every eigenvalue of D_S M_SS has positive real part.

Usage:
  python compare_ODE_IBMF.py --ibmf DIR --ode DIR --graph-exe print_graph_IBMF.out --N 1024 --mu 0.37 --sigma 0.0
      --eps 0.0 --seedgraph 14 --out PREFIX
"""
import argparse, glob, json, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def read_bin(path):
    with open(path, 'rb') as fh:
        if fh.read(8) != b'IBMFFP01':
            raise ValueError(f"{path}: not a fixed-point file")
        N, nfp, attempts = np.frombuffer(fh.read(24), '<i8')
        counts = np.frombuffer(fh.read(4 * nfp), '<i4').astype(int)
        M = np.frombuffer(fh.read(4 * nfp * N), '<f4').reshape(nfp, N).astype(float)
    return M, counts, int(attempts)


def read_dir(d):
    M, counts, attempts = read_bin(glob.glob(f"{d}/*_fixedpoints.bin")[0])
    log = np.loadtxt(glob.glob(f"{d}/*_runs.txt")[0], comments='#', ndmin=2)
    return dict(M=M, counts=counts, attempts=attempts, log=log)


def load_graph(exe, N, mu, sigma, eps, seed):
    out = subprocess.run([exe, str(N), f"{mu}", f"{sigma}", f"{eps}", str(seed)], capture_output=True, text=True,
                         check=True).stdout
    R = np.array([l.split() for l in out.strip().split('\n')], float)
    nb = R[:, 1::2].astype(int)
    a = R[:, 2::2]
    rows = np.repeat(np.arange(N), nb.shape[1])
    A = csr_matrix((a.ravel(), (rows, nb.ravel())), shape=(N, N))     # A[i, j] = a_ij (effect of j on i)
    return A


def match(Ma, Mb, tol=1e-2):
    """index in Mb of the fixed point equal to each row of Ma (max |diff| < tol), -1 if none"""
    idx = -np.ones(len(Ma), int)
    for k, m in enumerate(Ma):
        if len(Mb) == 0:
            break
        dmax = np.abs(Mb - m).max(1)
        j = int(dmax.argmin())
        if dmax[j] < tol:
            idx[k] = j
    return idx


def stability(A, m, symmetric, ext_tol=1e-9):
    """largest field among extinct species, smallest real part of the eigenvalues of D_S M_SS, residual"""
    N = len(m)
    f = 1.0 - A @ m
    ext = m <= ext_tol
    surv = ~ext
    residual = max(np.abs(m[surv] - f[surv]).max(initial=0.0), np.maximum(f[ext], 0).max(initial=0.0))
    max_field_ext = f[ext].max() if ext.any() else -np.inf
    S = np.nonzero(surv)[0]
    MSS = (A[S][:, S]).toarray() + np.eye(len(S))
    if symmetric:      # D^1/2 M D^1/2 is symmetric and similar to D M: same eigenvalues
        dh = np.sqrt(m[S])
        lam = np.linalg.eigvalsh(dh[:, None] * MSS * dh[None, :]).min()
    else:
        lam = np.linalg.eigvals(m[S][:, None] * MSS).real.min()
    return max_field_ext, lam, residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ibmf', required=True); ap.add_argument('--ode', required=True)
    ap.add_argument('--graph-exe', required=True)
    ap.add_argument('--N', type=int, required=True); ap.add_argument('--mu', type=float, required=True)
    ap.add_argument('--sigma', type=float, required=True); ap.add_argument('--eps', type=float, default=0.0)
    ap.add_argument('--seedgraph', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--tol_fp', type=float, default=1e-2)
    args = ap.parse_args()

    I, O = read_dir(args.ibmf), read_dir(args.ode)
    A = load_graph(args.graph_exe, args.N, args.mu, args.sigma, args.eps, args.seedgraph)
    n_init = I['attempts']
    res = dict(N=args.N, mu=args.mu, sigma=args.sigma, seedgraph=args.seedgraph, n_init=n_init)
    for name, D in (('ibmf', I), ('ode', O)):
        res[f'{name}_converged'] = float(D['log'][:, 1].mean())
        res[f'{name}_n_fp'] = int(len(D['counts']))
    # fixed points in common
    io = match(I['M'], O['M'], args.tol_fp)          # IBMF fixed point -> ODE fixed point
    oi = match(O['M'], I['M'], args.tol_fp)
    res['n_common'] = int((io >= 0).sum())
    phi_I = I['counts'] / n_init
    phi_O = O['counts'] / n_init
    res['phi_ibmf_in_common'] = float(phi_I[io >= 0].sum())   # fraction of IBMF runs ending at a common fixed point
    res['phi_ode_in_common'] = float(phi_O[oi >= 0].sum())
    common = np.nonzero(io >= 0)[0]
    pI, pO = phi_I[common], phi_O[io[common]]
    res['corr_phi_common_log'] = float(np.corrcoef(np.log(pI), np.log(pO))[0, 1]) if len(common) > 2 else None
    # run by run: same fixed point from the same initial condition
    li, lo = I['log'], O['log']
    both = (li[:, 7] >= 0) & (lo[:, 7] >= 0)
    same = np.zeros(len(li), bool)
    for k in np.nonzero(both)[0]:
        same[k] = np.abs(I['M'][int(li[k, 7])] - O['M'][int(lo[k, 7])]).max() < args.tol_fp
    res['run_by_run_same'] = float(same[both].mean())
    # m_bar of the fixed point of each run
    mb = {}
    for name, D in (('ibmf', I), ('ode', O)):
        ok = D['log'][:, 7] >= 0
        mb[name] = D['M'][D['log'][ok, 7].astype(int)].mean(1)
    # stability
    stab = {}
    symmetric = args.sigma == 0
    for name, D in (('ibmf', I), ('ode', O)):
        s = np.array([stability(A, m, symmetric) for m in D['M']])
        stab[name] = s
        res[f'{name}_max_field_extinct'] = float(s[:, 0].max())
        res[f'{name}_min_eig'] = float(s[:, 1].min())
        res[f'{name}_n_unstable'] = int(((s[:, 0] > 0) | (s[:, 1] <= 0)).sum())
        res[f'{name}_max_residual'] = float(s[:, 2].max())
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{args.out}.json", 'w'), indent=1)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "axes.linewidth": 1.2})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), dpi=130)
    ax = axs[0]
    ax.loglog(pI, pO, 'o', ms=4, label=f"{len(common)} fixed points in common")
    lim = [0.5 / n_init, 1]
    ax.plot(lim, lim, 'k:', lw=1)
    ax.set_xlabel(r"$\phi_{\rm IBMF}$", fontsize=15); ax.set_ylabel(r"$\phi_{\rm ODE}$", fontsize=15)
    ax.set_title(rf"(a) $\sigma={args.sigma:.2f}$, $N={args.N}$, $\mu={args.mu:.3f}$", fontsize=13)
    ax.legend(fontsize=10, frameon=False, loc='upper left')
    ax = axs[1]
    bins = np.linspace(min(mb['ibmf'].min(), mb['ode'].min()), max(mb['ibmf'].max(), mb['ode'].max()), 41)
    for name, lab in (('ibmf', 'IBMF'), ('ode', 'ODE')):
        ax.hist(mb[name], bins=bins, histtype='step', lw=1.6, density=True, label=lab)
    ax.set_xlabel(r"$\bar m$", fontsize=15); ax.set_ylabel("density of runs", fontsize=13)
    ax.set_title(r"(b) mean abundance of the fixed point reached", fontsize=13)
    ax.legend(fontsize=10, frameon=False)
    ax = axs[2]
    for name, lab, mk in (('ibmf', 'IBMF', 'o'), ('ode', 'ODE', 's')):
        ax.plot(stab[name][:, 1], stab[name][:, 0], mk, ms=4, mfc='none', label=lab)
    ax.axvline(0, color='gray', lw=0.8, ls=':'); ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(r"min Re eig$(D_SM_{SS})$", fontsize=13); ax.set_ylabel("max field of extinct species", fontsize=13)
    ax.set_title("(c) stability of the fixed points", fontsize=13)
    ax.legend(fontsize=10, frameon=False)
    for ax in axs:
        ax.tick_params(direction='in', top=True, right=True, which='both')
    fig.tight_layout()
    fig.savefig(f"{args.out}.png")


if __name__ == '__main__':
    sys.exit(main())
