"""Per-graph analysis of the fixed points written by IBMF_LV_sequential_countFP --print_fp_bin.

Input: one file  <base>_fixedpoints.bin(.gz)  (all distinct fixed points of one graph, with the number of converged
runs that reached each of them). The ensemble is that of the pairs of distinct converged runs, as in the distance
histograms: a fixed point reached w_k times counts w_k times, and pairs of runs that reached the same fixed point
have d = 0 and X_i = 0.

Output (npz, small): sums over the pairs of runs needed to average over graphs,
  - moments of d and d^2, and a histogram of d,
  - same-site sums  sum_i <X_i>,  sum_i <X_i^2>  (X_i = (m_i^a - m_i^b)^2),
  - different-site sums by graph distance r:  sum_{dist(i,j)=r} <X_i X_j>  and  sum_{dist(i,j)=r} <X_i><X_j>,
  - number of pairs of sites at each distance,
  - overlap C_ab, mean s^2, fraction of extinct species p0,
  - frozen sites (exactly 0, exactly 1, other values within several tolerances), also among k random runs.
All pair averages <.> are over ordered pairs of distinct runs, P = W (W - 1) with W the number of converged runs.

The sums over pairs are computed from sums over fixed points (cost linear in the number of fixed points):
  sum_{k,l} w_k w_l (u_k-u_l)^2 (v_k-v_l)^2 = 2W S[w u^2 v^2] + 2 S[w u^2] S[w v^2] - 4 S[w u^2 v] S[w v]
                                             - 4 S[w u v^2] S[w u] + 4 S[w u v]^2,
with u = m_i, v = m_j and S the sum over fixed points.

Usage:
  python analyse_IBMF_fixedpoints_bin.py --bin FILE --graph-exe print_graph_IBMF.out --out FILE.npz
"""
import argparse, gzip, re, subprocess, time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

D_BINS = np.linspace(0.0, 1.5, 3001)                # histogram of d
FROZEN_TOLS = np.array([1e-6, 1e-5, 1e-4, 1e-3])     # tolerances for sites frozen at other values
K_LIST = np.array([2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2048])
N_SUBSETS = 20


def read_fixedpoints(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as fh:
        magic = fh.read(8)
        if magic != b'IBMFFP01':
            raise ValueError(f"{path}: not a fixed-point file")
        N, nfp, attempts = np.frombuffer(fh.read(24), dtype='<i8')
        counts = np.frombuffer(fh.read(4 * nfp), dtype='<i4').astype(np.int64)
        M = np.frombuffer(fh.read(4 * nfp * N), dtype='<f4').reshape(nfp, N)
    return int(N), int(attempts), counts, M


def parse_name(path):
    m = re.search(r"eps_([0-9.]+)_mu_([0-9.]+)_sigma_([0-9.]+)_N_(\d+)_c_(\d+)_seedgraph_(\d+)", path)
    return dict(eps=m.group(1), mu=m.group(2), sigma=m.group(3), N=int(m.group(4)), c=int(m.group(5)),
                seed_graph=int(m.group(6)))


def load_graph(exe, N, mu, sigma, eps, seed_graph):
    out = subprocess.run([exe, str(N), mu, sigma, eps, str(seed_graph)], capture_output=True, text=True,
                         check=True).stdout
    R = np.array([l.split() for l in out.strip().split('\n')], float)
    nb = R[:, 1::2].astype(int); a = R[:, 2::2]
    c = nb.shape[1]
    G = csr_matrix((np.ones(c * N), (np.repeat(np.arange(N), c), nb.ravel())), shape=(N, N))
    return G, nb, a


def pair_sums_distances(M, w):
    """Moments of d, d^2 and histogram of d over ordered pairs of distinct runs."""
    nfp, N = M.shape
    q = (M * M).sum(1)
    Gm = M @ M.T
    d2 = np.maximum((q[:, None] + q[None, :] - 2 * Gm) / N, 0.0)
    np.fill_diagonal(d2, 0.0)
    ww = np.outer(w, w).astype(float)
    np.fill_diagonal(ww, w * (w - 1.0))           # pairs of distinct runs at the same fixed point: d = 0
    d = np.sqrt(d2)
    hist, _ = np.histogram(d.ravel(), bins=D_BINS, weights=ww.ravel())
    # overlap measured from the means of each fixed point
    mbar = M.mean(1)
    Mc = M - mbar[:, None]
    C = (Mc @ Mc.T) / N
    s2 = (Mc * Mc).sum(1) / N
    return dict(P=ww.sum(), sum_d=(ww * d).sum(), sum_d2=(ww * d2).sum(), sum_d4=(ww * d2 * d2).sum(),
                sum_d3=(ww * d2 * d).sum(), hist_d=hist,
                sum_Cab=(ww * C).sum(),
                sum_s2=(w * s2).sum(), W=w.sum())


def site_and_shell_sums(M, w, G, chunk=512, rmax=64):
    """Same-site sums and different-site sums by graph distance (ordered pairs of distinct runs)."""
    nfp, N = M.shape
    W = w.sum(); P = W * (W - 1.0)
    u = M.astype(np.float64)
    s1 = w @ u; s2 = w @ (u * u); s3 = w @ (u ** 3); s4 = w @ (u ** 4)
    EX = (2 * W * s2 - 2 * s1 * s1) / P                                   # <X_i>
    EX2 = (2 * W * s4 - 8 * s3 * s1 + 6 * s2 * s2) / P                    # <X_i^2>
    wu = w[:, None] * u; wu2 = w[:, None] * u * u
    XX = np.zeros(rmax + 1); XmXm = np.zeros(rmax + 1); npairs = np.zeros(rmax + 1)
    for i0 in range(0, N, chunk):
        idx = np.arange(i0, min(N, i0 + chunk))
        D = dijkstra(G, directed=False, unweighted=True, indices=idx)
        D[~np.isfinite(D)] = rmax
        D = D.astype(np.int64)
        A22 = wu2[:, idx].T @ (u * u)            # S[w u_i^2 u_j^2]
        A21 = wu2[:, idx].T @ u                  # S[w u_i^2 u_j]
        A12 = wu[:, idx].T @ (u * u)             # S[w u_i u_j^2]
        A11 = wu[:, idx].T @ u                   # S[w u_i u_j]
        EXX = (2 * W * A22 + 2 * s2[idx, None] * s2[None, :] - 4 * A21 * s1[None, :] - 4 * A12 * s1[idx, None]
               + 4 * A11 * A11) / P              # <X_i X_j>
        XX += np.bincount(D.ravel(), weights=EXX.ravel(), minlength=rmax + 1)[:rmax + 1]
        XmXm += np.bincount(D.ravel(), weights=np.outer(EX[idx], EX).ravel(), minlength=rmax + 1)[:rmax + 1]
        npairs += np.bincount(D.ravel(), minlength=rmax + 1)[:rmax + 1]
    last = int(np.nonzero(npairs)[0].max())
    return dict(sum_EX=EX.sum(), sum_EX2=EX2.sum(), sum_EX_sq=(EX * EX).sum(),
                XX=XX[:last + 1], XmXm=XmXm[:last + 1], npairs=npairs[:last + 1])


def frozen_statistics(M, w, seed=0):
    nfp, N = M.shape
    zero = (M == 0).all(0); one = (M == 1).all(0)
    spread = M.max(0) - M.min(0)
    other = np.array([((spread < t) & ~zero & ~one).mean() for t in FROZEN_TOLS])
    runs = np.repeat(np.arange(nfp), w)
    rng = np.random.default_rng(seed)
    fk0 = np.full(len(K_LIST), np.nan); fk1 = np.full(len(K_LIST), np.nan)
    for j, k in enumerate(K_LIST):
        if k > len(runs):
            continue
        z, o = [], []
        for _ in range(N_SUBSETS):
            sel = np.unique(runs[rng.choice(len(runs), k, replace=False)])
            z.append((M[sel] == 0).all(0).mean()); o.append((M[sel] == 1).all(0).mean())
        fk0[j], fk1[j] = np.mean(z), np.mean(o)
    p0 = (w * (M == 0).mean(1)).sum() / w.sum()
    p1 = (w * (M == 1).mean(1)).sum() / w.sum()
    return dict(frozen0=zero.mean(), frozen1=one.mean(), frozen_other=other, frozen_tols=FROZEN_TOLS,
                frozen0_vs_k=fk0, frozen1_vs_k=fk1, k_list=K_LIST, p0=p0, p1=p1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--graph-exe', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    t0 = time.time()
    meta = parse_name(args.bin)
    N, attempts, counts, M = read_fixedpoints(args.bin)
    assert N == meta['N']
    w = counts.astype(np.float64)
    G, nb, a = load_graph(args.graph_exe, N, meta['mu'], meta['sigma'], meta['eps'], meta['seed_graph'])
    residual = float(np.abs(M - np.maximum(0, 1 - (a[None] * M[:, nb]).sum(2))).max()) if len(M) else np.nan
    out = dict(N=N, attempts=attempts, nfp=len(M), counts=counts, residual=residual,
               sigma=float(meta['sigma']), mu=float(meta['mu']), seed_graph=meta['seed_graph'], d_bins=D_BINS)
    if w.sum() >= 2:
        out.update(pair_sums_distances(M.astype(np.float64), w))
        out.update(site_and_shell_sums(M, w, G))
        out.update(frozen_statistics(M, counts))
    out['time'] = time.time() - t0
    np.savez_compressed(args.out, **out)
    print(f"{args.bin}: N={N} nfp={len(M)} runs={int(w.sum())}/{attempts} residual={residual:.1e} "
          f"time={out['time']:.0f}s")


if __name__ == '__main__':
    main()
