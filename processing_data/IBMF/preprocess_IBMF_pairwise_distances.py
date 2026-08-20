#!/usr/bin/env python
r"""
Pool pairwise-fixed-point-distance files produced by IBMF_LV_sequential_countFP.cpp
(the --pairwise-dist output) across random-graph seeds, and turn each pool into a
weighted histogram (the P(q) distribution).

Input files are expected to be found anywhere under --input-dir (recursively, e.g. in
per-mu subfolders named "mu_<mu>") and to follow the naming convention:

    IBMF_T0_seq_gr_inside_RRG_eps_<eps>_mu_<mu>_sigma_<sigma>_N_<N>_c_<c>
    _seedgraph_<seedgraph>_Lotka_Volterra_final_av0_<av0>_dn_<dn>_ninitcond_<ninitcond>
    _tol_<tol>_maxiter_<maxiter>_damping_<damping>_pairwise_dist.txt

and contain two whitespace-separated columns (distance, multiplicity), one comment
header line starting with "#".

All files that share the same (eps, sigma, mu, N) are pooled into a single weighted
histogram, regardless of seedgraph (and regardless of any other parameter in the name --
a warning is printed if those other parameters differ within a group, since that usually
signals runs that should not be pooled together). Each file is only ever read for its
two columns, and the histogram is accumulated incrementally, so memory use does not grow
with the number or size of the input files.

One output file per (eps, sigma, mu, N) group is written to --output-dir, with columns:
    bin_left  bin_right  bin_center  count  density
where density is normalized so that sum(density * bin_width) = 1.
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

FILENAME_RE = re.compile(
    r"eps_(?P<eps>[0-9.eE+-]+)_mu_(?P<mu>[0-9.eE+-]+)_sigma_(?P<sigma>[0-9.eE+-]+)"
    r"_N_(?P<N>\d+)_c_(?P<c>[0-9.eE+-]+)_seedgraph_(?P<seedgraph>\d+)"
    r"_Lotka_Volterra_final_av0_(?P<av0>[0-9.eE+-]+)_dn_(?P<dn>[0-9.eE+-]+)"
    r"_ninitcond_(?P<ninitcond>\d+)_tol_(?P<tol>[0-9.eE+-]+)_maxiter_(?P<maxiter>\d+)"
    r"_damping_(?P<damping>[0-9.eE+-]+)_pairwise_dist\.txt$"
)

# Fields that identify a physical parameter point; files sharing these are pooled.
GROUP_KEYS = ("eps", "sigma", "mu", "N")
# Other fields worth checking for consistency within a group (excludes seedgraph,
# which is expected to vary).
CONSISTENCY_KEYS = ("c", "av0", "dn", "ninitcond", "tol", "maxiter", "damping")


def find_input_files(input_dir):
    matched = []
    for root, _dirs, filenames in os.walk(input_dir):
        for filename in filenames:
            if FILENAME_RE.search(filename) is not None:
                matched.append(os.path.join(root, filename))
    return sorted(matched)


def parse_filename(path):
    m = FILENAME_RE.search(os.path.basename(path))
    if m is None:
        return None
    return m.groupdict()


def group_files(paths):
    groups = defaultdict(list)
    for path in paths:
        params = parse_filename(path)
        if params is None:
            print(f"  [skip] filename does not match expected pattern: {path}", file=sys.stderr)
            continue
        key = tuple(params[k] for k in GROUP_KEYS)
        groups[key].append((path, params))
    return groups


def check_consistency(key, entries):
    reference = entries[0][1]
    for path, params in entries[1:]:
        for field in CONSISTENCY_KEYS:
            if params[field] != reference[field]:
                print(
                    f"  [warn] group eps={key[0]} sigma={key[1]} mu={key[2]} N={key[3]}: "
                    f"'{field}' differs across pooled files "
                    f"({reference[field]!r} vs {params[field]!r} in {os.path.basename(path)}); "
                    f"pooling them anyway.",
                    file=sys.stderr,
                )


def scan_min_max(paths):
    global_min = np.inf
    global_max = -np.inf
    for path in paths:
        col = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                           usecols=[0], names=["distance"])["distance"].to_numpy()
        if col.size == 0:
            continue
        global_min = min(global_min, float(col.min()))
        global_max = max(global_max, float(col.max()))
    return global_min, global_max


def accumulate_histogram(paths, edges):
    nbins = len(edges) - 1
    counts = np.zeros(nbins, dtype=np.float64)
    total_weight = 0
    for path in paths:
        df = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                          names=["distance", "multiplicity"])
        distances = df["distance"].to_numpy(dtype=np.float64)
        multiplicities = df["multiplicity"].to_numpy(dtype=np.int64)
        if distances.size == 0:
            continue
        local_counts, _ = np.histogram(distances, bins=edges, weights=multiplicities)
        counts += local_counts
        total_weight += int(multiplicities.sum())
    return counts, total_weight


def build_output_filename(key, reference_params, nbins):
    eps, sigma, mu, N = key
    c = reference_params["c"]
    av0 = reference_params["av0"]
    dn = reference_params["dn"]
    ninitcond = reference_params["ninitcond"]
    tol = reference_params["tol"]
    maxiter = reference_params["maxiter"]
    damping = reference_params["damping"]
    return (
        f"IBMF_T0_seq_gr_inside_RRG_eps_{eps}_mu_{mu}_sigma_{sigma}_N_{N}_c_{c}"
        f"_Lotka_Volterra_final_av0_{av0}_dn_{dn}_ninitcond_{ninitcond}_tol_{tol}"
        f"_maxiter_{maxiter}_damping_{damping}_pairwise_dist_histogram_nbins_{nbins}.txt"
    )


def write_histogram(out_path, key, entries, edges, counts, total_weight):
    eps, sigma, mu, N = key
    seedgraphs = sorted({params["seedgraph"] for _, params in entries}, key=int)
    bin_width = edges[1] - edges[0]
    if total_weight > 0:
        density = counts / (total_weight * bin_width)
    else:
        density = np.zeros_like(counts)

    header_lines = [
        f"eps={eps} sigma={sigma} mu={mu} N={N}",
        f"pooled {len(entries)} file(s), seedgraph values: {','.join(seedgraphs)}",
        f"total_weight={total_weight} nbins={len(counts)} "
        f"min_dist={edges[0]:.10f} max_dist={edges[-1]:.10f}",
        "bin_left bin_right bin_center count density",
    ]
    header = "\n".join(header_lines)

    bin_left = edges[:-1]
    bin_right = edges[1:]
    bin_center = 0.5 * (bin_left + bin_right)
    data = np.column_stack([bin_left, bin_right, bin_center, counts, density])
    np.savetxt(out_path, data, fmt=["%.10f", "%.10f", "%.10f", "%d", "%.10e"],
               delimiter="\t", header=header, comments="# ")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", required=True,
                         help="Root directory to search recursively for "
                              "*_pairwise_dist.txt files (e.g. the PhaseDiagram folder "
                              "containing the mu_<mu> subfolders).")
    parser.add_argument("--output-dir", required=True,
                         help="Directory to write one pooled histogram file per "
                              "(eps, sigma, mu, N) group into.")
    parser.add_argument("--bins", type=int, required=True,
                         help="Number of histogram bins.")
    parser.add_argument("--min-dist", type=float, default=None,
                         help="Lower edge of the histogram range. If omitted, it is "
                              "determined automatically per group from the pooled data "
                              "(requires an extra pass over the group's files).")
    parser.add_argument("--max-dist", type=float, default=None,
                         help="Upper edge of the histogram range. If omitted, it is "
                              "determined automatically per group from the pooled data "
                              "(requires an extra pass over the group's files).")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Searching {args.input_dir} for pairwise-distance files ...")
    files = find_input_files(args.input_dir)
    if not files:
        print("No matching pairwise-distance files found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(files)} file(s).")

    groups = group_files(files)
    if not groups:
        print("No files could be parsed successfully.", file=sys.stderr)
        sys.exit(1)
    print(f"Grouped into {len(groups)} (eps, sigma, mu, N) set(s).")

    os.makedirs(args.output_dir, exist_ok=True)

    for key, entries in sorted(groups.items()):
        eps, sigma, mu, N = key
        check_consistency(key, entries)
        paths = [path for path, _ in entries]

        if args.min_dist is not None and args.max_dist is not None:
            min_dist, max_dist = args.min_dist, args.max_dist
        else:
            print(f"  Scanning range for eps={eps} sigma={sigma} mu={mu} N={N} "
                  f"({len(paths)} file(s)) ...")
            auto_min, auto_max = scan_min_max(paths)
            min_dist = args.min_dist if args.min_dist is not None else auto_min
            max_dist = args.max_dist if args.max_dist is not None else auto_max

        if not np.isfinite(min_dist) or not np.isfinite(max_dist) or min_dist >= max_dist:
            print(f"  [skip] could not determine a valid range for eps={eps} sigma={sigma} "
                  f"mu={mu} N={N} (min={min_dist}, max={max_dist})", file=sys.stderr)
            continue

        edges = np.linspace(min_dist, max_dist, args.bins + 1)

        print(f"  Pooling {len(paths)} file(s) for eps={eps} sigma={sigma} mu={mu} N={N} ...")
        counts, total_weight = accumulate_histogram(paths, edges)

        reference_params = entries[0][1]
        out_name = build_output_filename(key, reference_params, args.bins)
        out_path = os.path.join(args.output_dir, out_name)
        write_histogram(out_path, key, entries, edges, counts, total_weight)
        print(f"  Wrote {out_path} (total_weight={total_weight})")

    print("Done.")


if __name__ == "__main__":
    main()
