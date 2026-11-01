"""Extract the fixed-point summaries of IBMF_LV_sequential_countFP from a .tar.gz archive, without unpacking it.

The archive is read as a stream: only the members named "..._summary.txt" with the requested sigma (and seedgraph)
are decompressed and parsed. Each summary file has the header
    # number count n_ave n_sq n[0] n[1]
with one line per distinct fixed point: index, fraction of the runs (initial conditions) that reached it, average
abundance, average squared abundance, and the abundances of species 0 and 1.

Output: a gzipped CSV with one row per fixed point and columns
    eps sigma mu N c seedgraph ninitcond fp fraction n_ave n_sq n0 n1 nfp
(ninitcond is the number of initial conditions in the file name, which includes the intermediate checkpoints; nfp is
the number of distinct fixed points in that file). Used to draw bifurcation diagrams (n0 vs mu) for each pair
(N, ninitcond) of a given graph.

Usage:
  python extract_bifurcation_summaries.py --tar ARCHIVE.tar.gz --sigma 0.000 --seedgraph 3 --out OUT.csv.gz
  (--seedgraph can be repeated, or omitted to keep all graphs; --max-members limits the scan, for debugging)
"""
import argparse, csv, gzip, os, re, sys, tarfile, time

NAME_RE = re.compile(r"IBMF_T0_seq_gr_inside_RRG_eps_(?P<eps>[0-9.]+)_mu_(?P<mu>[0-9.]+)_sigma_(?P<sigma>[0-9.]+)"
                     r"_N_(?P<N>\d+)_c_(?P<c>\d+)_seedgraph_(?P<seedgraph>\d+)_Lotka_Volterra_final_av0_[0-9.]+_dn_[0-9.]+"
                     r"_ninitcond_(?P<ninitcond>\d+)_tol_[0-9.e+-]+_maxiter_\d+_damping_[0-9.]+_summary\.txt$")
COLUMNS = ['eps', 'sigma', 'mu', 'N', 'c', 'seedgraph', 'ninitcond', 'fp', 'fraction', 'n_ave', 'n_sq', 'n0', 'n1',
           'nfp']


def parse_summary(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        v = line.split()
        if len(v) < 6:
            continue
        rows.append((int(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]), float(v[5])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tar', required=True, help='.tar.gz archive')
    ap.add_argument('--sigma', default='0.000')
    ap.add_argument('--seedgraph', type=int, action='append', help='graph(s) to keep; all if omitted')
    ap.add_argument('--out', required=True, help='output .csv.gz')
    ap.add_argument('--max-members', type=int, default=None, help='stop after this many archive members (debug)')
    args = ap.parse_args()

    t0 = time.time()
    keep_graphs = set(args.seedgraph) if args.seedgraph else None
    nmembers = nfiles = nrows = 0
    pairs = {}
    with gzip.open(args.out, 'wt', newline='') as fout, tarfile.open(args.tar, mode='r|gz') as tf:
        writer = csv.writer(fout)
        writer.writerow(COLUMNS)
        for member in tf:
            nmembers += 1
            if args.max_members is not None and nmembers > args.max_members:
                break
            if not member.isfile():
                continue
            m = NAME_RE.search(os.path.basename(member.name))
            if m is None or m.group('sigma') != args.sigma:
                continue
            seedgraph = int(m.group('seedgraph'))
            if keep_graphs is not None and seedgraph not in keep_graphs:
                continue
            text = tf.extractfile(member).read().decode()
            rows = parse_summary(text)
            for fp, frac, n_ave, n_sq, n0, n1 in rows:
                writer.writerow([m.group('eps'), m.group('sigma'), m.group('mu'), m.group('N'), m.group('c'), seedgraph,
                                 m.group('ninitcond'), fp, frac, n_ave, n_sq, n0, n1, len(rows)])
            nfiles += 1
            nrows += len(rows)
            key = (seedgraph, int(m.group('N')), int(m.group('ninitcond')))
            pairs[key] = pairs.get(key, 0) + 1
            if nfiles % 1000 == 0:
                print(f"  {nfiles} summary files, {nmembers} members scanned, {time.time() - t0:.0f} s", flush=True)

    print(f"{args.tar}: {nmembers} members scanned, {nfiles} summary files kept, {nrows} fixed points, "
          f"{time.time() - t0:.0f} s")
    print("seedgraph  N  ninitcond  number_of_mu_values")
    for (g, N, ninit), n in sorted(pairs.items()):
        print(f"{g} {N} {ninit} {n}")


if __name__ == '__main__':
    sys.exit(main())
