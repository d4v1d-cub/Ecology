"""Per-task analysis of the cluster comparison between IBMF and the numerical integration of the Lotka-Volterra
equations (outputs of LV_ODE_countFP --method ibmf / --method ode for one graph and one value of mu).

Writes <out>.json (scalar statistics) and <out>.npz (arrays):
  - for each method and each checkpoint n_init = 256, 512, ...: number of distinct fixed points n_fixed and number of
    converged runs n_conv (from the summary files), as in Figs. 1-2 of the notes;
  - fixed points in common (max |diff| < tol_fp), phi of the common fixed points for both methods;
  - fraction of initial conditions for which both methods reach the same fixed point;
  - mean abundance of the fixed point reached by each converged run, for both methods;
  - stability of every fixed point: largest field of an extinct species and smallest real part of the eigenvalues of
    D_S M_SS (see compare_ODE_IBMF.py).

Usage:
  python analyse_ODE_IBMF_task.py --ibmf DIR --ode DIR --graph-exe print_graph_IBMF.out --N 1024 --mu 0.37
      --sigma 0.0 --eps 0.0 --seedgraph 1 --out PREFIX
"""
import argparse, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_ODE_IBMF import read_dir, load_graph, match, stability


def checkpoint_counts(d):
    """{n_init: (n_fixed, n_conv)} from the summary files of a run directory"""
    out = {}
    for f in glob.glob(f"{d}/*_summary.txt"):
        m = re.search(r"_ninitcond_(\d+)_", os.path.basename(f))
        if m is None:
            continue
        ninit = int(m.group(1))
        rows = [l.split() for l in open(f) if l.strip() and not l.startswith('#')]
        frac = np.array([float(r[1]) for r in rows]) if rows else np.zeros(0)
        out[ninit] = (len(rows), int(round(frac.sum() * ninit)))
    return dict(sorted(out.items()))


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
    arrays = {}
    for name, D, d in (('ibmf', I, args.ibmf), ('ode', O, args.ode)):
        res[f'{name}_converged'] = float(D['log'][:, 1].mean())
        res[f'{name}_n_fp'] = int(len(D['counts']))
        cc = checkpoint_counts(d)
        arrays[f'{name}_ninit'] = np.array(list(cc.keys()))
        arrays[f'{name}_nfixed'] = np.array([v[0] for v in cc.values()])
        arrays[f'{name}_nconv'] = np.array([v[1] for v in cc.values()])
        ok = D['log'][:, 7] >= 0
        arrays[f'{name}_mbar_runs'] = D['M'][D['log'][ok, 7].astype(int)].mean(1) if ok.any() else np.zeros(0)
        arrays[f'{name}_phi'] = D['counts'] / n_init
        if len(D['M']):
            s = np.array([stability(A, m, args.sigma == 0) for m in D['M']])
        else:
            s = np.zeros((0, 3))
        arrays[f'{name}_stability'] = s       # columns: max field extinct, min Re eig(D_S M_SS), residual
        res[f'{name}_max_field_extinct'] = float(s[:, 0].max()) if len(s) else None
        res[f'{name}_min_eig'] = float(s[:, 1].min()) if len(s) else None
    io = match(I['M'], O['M'], args.tol_fp)
    oi = match(O['M'], I['M'], args.tol_fp)
    common = np.nonzero(io >= 0)[0]
    arrays['phi_ibmf_common'] = I['counts'][common] / n_init
    arrays['phi_ode_common'] = O['counts'][io[common]] / n_init
    res['n_common'] = int(len(common))
    res['phi_ibmf_in_common'] = float(arrays['ibmf_phi'][io >= 0].sum())
    res['phi_ode_in_common'] = float(arrays['ode_phi'][oi >= 0].sum())
    li, lo = I['log'], O['log']
    both = (li[:, 7] >= 0) & (lo[:, 7] >= 0)
    same = [np.abs(I['M'][int(li[k, 7])] - O['M'][int(lo[k, 7])]).max() < args.tol_fp for k in np.nonzero(both)[0]]
    res['run_by_run_same'] = float(np.mean(same)) if same else None
    json.dump(res, open(f"{args.out}.json", 'w'), indent=1)
    np.savez_compressed(f"{args.out}.npz", **arrays)
    print(json.dumps(res))


if __name__ == '__main__':
    sys.exit(main())
