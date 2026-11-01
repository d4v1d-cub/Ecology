"""Extract the fixed-point summaries of IBMF_LV_sequential_countFP from a .tar.gz archive (read as a stream, without
unpacking it) or from a directory of summary files.

Only the files named "..._summary.txt" with the requested sigma (and seedgraph) are read. Each summary file has the
header
    # number count n_ave n_sq n[0] n[1] [n_ext n_one d_ref h_ref]
with one line per distinct fixed point: index, fraction of the runs (initial conditions) that reached it, average
abundance, average squared abundance, the abundances of species 0 and 1 and, in the files written by the current
version of the program, the number of extinct species, the number of species with abundance 1, the distance to the
most frequent fixed point and the number of species whose extinction differs from it (empty for older files).

Output: a gzipped CSV with one row per fixed point and columns
    eps sigma mu N c seedgraph ninitcond fp fraction n_ave n_sq n0 n1 n_ext n_one d_ref h_ref nfp
(ninitcond is the number of initial conditions in the file name, which includes the intermediate checkpoints; nfp is
the number of distinct fixed points in that file). Used to draw bifurcation diagrams (n0 vs mu) for each pair
(N, ninitcond) of a given graph.

Usage:
  python extract_bifurcation_summaries.py --tar ARCHIVE.tar.gz --sigma 0.000 --seedgraph 3 --out OUT.csv.gz
  python extract_bifurcation_summaries.py --dir DIRECTORY --sigma 0.150 --out OUT.csv.gz   (searched recursively)
  (--seedgraph can be repeated, or omitted to keep all graphs; --max-members limits the scan, for debugging)
"""
import argparse, csv, gzip, os, re, sys, tarfile, time

NAME_RE = re.compile(r"IBMF_T0_seq_gr_inside_RRG_eps_(?P<eps>[0-9.]+)_mu_(?P<mu>[0-9.]+)_sigma_(?P<sigma>[0-9.]+)"
                     r"_N_(?P<N>\d+)_c_(?P<c>\d+)_seedgraph_(?P<seedgraph>\d+)_Lotka_Volterra_final_av0_[0-9.]+_dn_[0-9.]+"
                     r"_ninitcond_(?P<ninitcond>\d+)_tol_[0-9.e+-]+_maxiter_\d+_damping_[0-9.]+_summary\.txt$")
COLUMNS = ['eps', 'sigma', 'mu', 'N', 'c', 'seedgraph', 'ninitcond', 'fp', 'fraction', 'n_ave', 'n_sq', 'n0', 'n1',
           'n_ext', 'n_one', 'd_ref', 'h_ref', 'nfp']


def parse_summary(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        v = line.split()
        if len(v) < 6:
            continue
        extra = v[6:10] if len(v) >= 10 else [''] * 4
        rows.append([int(v[0])] + [float(x) for x in v[1:6]] + extra)
    return rows


def summary_files(args):
    """Yields (file name, text) for the summary files of the archive (as a stream) or of the directory."""
    if args.tar is not None:
        with tarfile.open(args.tar, mode='r|gz') as tf:
            for member in tf:
                yield member.name, (lambda m=member: tf.extractfile(m).read().decode()) if member.isfile() else None
    else:
        for root, _, names in os.walk(args.dir):
            for name in sorted(names):
                if name.endswith('_summary.txt'):
                    path = os.path.join(root, name)
                    yield path, (lambda p=path: open(p).read())


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--tar', help='.tar.gz archive')
    src.add_argument('--dir', help='directory with summary files (searched recursively)')
    ap.add_argument('--sigma', default='0.000')
    ap.add_argument('--seedgraph', type=int, action='append', help='graph(s) to keep; all if omitted')
    ap.add_argument('--out', required=True, help='output .csv.gz')
    ap.add_argument('--max-members', type=int, default=None, help='stop after this many files or members (debug)')
    args = ap.parse_args()

    t0 = time.time()
    keep_graphs = set(args.seedgraph) if args.seedgraph else None
    nmembers = nfiles = nrows = 0
    pairs = {}
    with gzip.open(args.out, 'wt', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(COLUMNS)
        for name, read_text in summary_files(args):
            nmembers += 1
            if args.max_members is not None and nmembers > args.max_members:
                break
            if read_text is None:
                continue
            m = NAME_RE.search(os.path.basename(name))
            if m is None or m.group('sigma') != args.sigma:
                continue
            seedgraph = int(m.group('seedgraph'))
            if keep_graphs is not None and seedgraph not in keep_graphs:
                continue
            rows = parse_summary(read_text())
            for row in rows:
                writer.writerow([m.group('eps'), m.group('sigma'), m.group('mu'), m.group('N'), m.group('c'), seedgraph,
                                 m.group('ninitcond')] + row + [len(rows)])
            nfiles += 1
            nrows += len(rows)
            key = (seedgraph, int(m.group('N')), int(m.group('ninitcond')))
            pairs[key] = pairs.get(key, 0) + 1
            if nfiles % 1000 == 0:
                print(f"  {nfiles} summary files, {nmembers} members scanned, {time.time() - t0:.0f} s", flush=True)

    print(f"{args.tar or args.dir}: {nmembers} files scanned, {nfiles} summary files kept, {nrows} fixed points, "
          f"{time.time() - t0:.0f} s")
    print("seedgraph  N  ninitcond  number_of_mu_values")
    for (g, N, ninit), n in sorted(pairs.items()):
        print(f"{g} {N} {ninit} {n}")


if __name__ == '__main__':
    sys.exit(main())
