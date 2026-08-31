#!/usr/bin/env python3
"""
Aggregate Equilibrium_Points files produced by Ecosystem_Dynamics_on_Graphs_in_Temperature.cpp
and build the empirical probability density of the equilibrium abundance of each species.

Expected input layout, rooted at each of --input-dirs:
    <root>/epsilon_<eps>_<ia_label>_lambda_<lambda>_h_<h>_tmax_<tmax>_deltatsave_<deltatsave>/
        N_<N>_c_<c>/
            Equilibrium_Points/
                Lotka-Volterra_mu_<mu>_sigma_<sigma>_T_<T>_Extraction_<n>_Measure_<j>_Equilibrium_Points.txt

Every file matching a given (epsilon, ia_label, lambda, h, tmax, deltatsave, N, c, mu, sigma, T)
combination is pooled together, across all Extractions and Measures (and across all --input-dirs),
and one output file per combination is written to --output-dir.

Every species, in every output file, is histogrammed on the SAME shared abundance grid (given
by --nmin/--nmax/--dn), rather than each species picking its own range -- this is what lets a
later averaging/comparison step (preprocess_Langevin_distributions.py) stack per-species and
per-file histograms directly, with no interpolation.

Pass shell globs directly to --input-dirs (e.g. "genericname_*"): the shell expands them into a
list of paths before this script ever sees them, so no special handling is needed here.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LEVEL1_RE = re.compile(
    r'^epsilon_(?P<epsilon>[^_]+)_(?P<ia_label>.+)_lambda_(?P<lambda_>[^_]+)'
    r'_h_(?P<h>[^_]+)_tmax_(?P<tmax>[^_]+)_deltatsave_(?P<deltatsave>[^_]+)$'
)
LEVEL2_RE = re.compile(r'^N_(?P<N>\d+)_c_(?P<c>[^_]+)$')
FILE_RE = re.compile(
    r'^Lotka-Volterra_mu_(?P<mu>[^_]+)_sigma_(?P<sigma>[^_]+)_T_(?P<T>[^_]+)'
    r'_Extraction_(?P<extraction>\d+)_Measure_(?P<measure>\d+)_Equilibrium_Points\.txt$'
)

GROUP_KEY_ORDER = ['epsilon', 'ia_label', 'lambda', 'h', 'tmax', 'deltatsave',
                    'N', 'c', 'mu', 'sigma', 'T']


def discover_equilibrium_files(input_dirs, verbose=False):
    """Yield (group_key, group_labels, file_path) for every Equilibrium_Points file found."""
    for root in input_dirs:
        root = Path(root)
        if not root.is_dir():
            print(f"WARNING: skipping '{root}': not a directory", file=sys.stderr)
            continue
        for level1 in sorted(root.glob('epsilon_*')):
            if not level1.is_dir():
                continue
            m1 = LEVEL1_RE.match(level1.name)
            if not m1:
                if verbose:
                    print(f"WARNING: skipping '{level1}': name does not match the expected pattern", file=sys.stderr)
                continue
            for level2 in sorted(level1.glob('N_*_c_*')):
                if not level2.is_dir():
                    continue
                m2 = LEVEL2_RE.match(level2.name)
                if not m2:
                    if verbose:
                        print(f"WARNING: skipping '{level2}': name does not match the expected pattern", file=sys.stderr)
                    continue
                eq_dir = level2 / 'Equilibrium_Points'
                if not eq_dir.is_dir():
                    continue
                for fpath in sorted(eq_dir.glob('Lotka-Volterra_*_Equilibrium_Points.txt')):
                    m3 = FILE_RE.match(fpath.name)
                    if not m3:
                        if verbose:
                            print(f"WARNING: skipping '{fpath}': name does not match the expected pattern", file=sys.stderr)
                        continue
                    group_labels = {
                        'epsilon': m1['epsilon'], 'ia_label': m1['ia_label'], 'lambda': m1['lambda_'],
                        'h': m1['h'], 'tmax': m1['tmax'], 'deltatsave': m1['deltatsave'],
                        'N': m2['N'], 'c': m2['c'],
                        'mu': m3['mu'], 'sigma': m3['sigma'], 'T': m3['T'],
                    }
                    group_key = tuple(group_labels[k] for k in GROUP_KEY_ORDER)
                    yield group_key, group_labels, fpath


def read_equilibrium_file(fpath, only_converged, verbose):
    """Return dict: species_index -> list of equilibrium abundances found in this file."""
    per_species = defaultdict(list)
    with open(fpath, 'r') as fin:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 4:
                parts = line.split()
            if len(parts) != 4:
                if verbose:
                    print(f"WARNING: skipping malformed line {lineno} in '{fpath}'", file=sys.stderr)
                continue
            idx_str, conv_flag, abundance_str, _std_str = parts
            try:
                idx = int(idx_str)
                abundance = float(abundance_str)
            except ValueError:
                if verbose:
                    print(f"WARNING: skipping unparsable line {lineno} in '{fpath}'", file=sys.stderr)
                continue
            if only_converged and conv_flag.strip() != 'T':
                continue
            per_species[idx].append(abundance)
    return per_species


def aggregate(input_dirs, only_converged, verbose):
    """Pool every Equilibrium_Points file into per-group, per-species abundance lists."""
    groups = defaultdict(lambda: defaultdict(list))  # group_key -> species_idx -> [abundances]
    group_labels_by_key = {}
    n_files_by_key = defaultdict(int)
    for group_key, group_labels, fpath in discover_equilibrium_files(input_dirs, verbose):
        if group_key not in group_labels_by_key:
            label_str = " ".join(f"{k}={group_labels[k]}" for k in GROUP_KEY_ORDER)
            print(f"Processing group: {label_str}")
        group_labels_by_key[group_key] = group_labels
        per_species = read_equilibrium_file(fpath, only_converged, verbose)
        for idx, values in per_species.items():
            groups[group_key][idx].extend(values)
        n_files_by_key[group_key] += 1
        if verbose:
            print(f"  pooled {fpath} ({n_files_by_key[group_key]} file(s) so far for this group)")
    return groups, group_labels_by_key, n_files_by_key


def build_output_name(labels):
    # "Extraction_.../Measure_..." is replaced by "AllExtractions_AllMeasures" since the data is pooled,
    # and "_PDF" is appended to distinguish this from a raw per-measure Equilibrium_Points file.
    return (
        f"Lotka-Volterra_epsilon_{labels['epsilon']}_{labels['ia_label']}_lambda_{labels['lambda']}"
        f"_h_{labels['h']}_tmax_{labels['tmax']}_deltatsave_{labels['deltatsave']}"
        f"_N_{labels['N']}_c_{labels['c']}_mu_{labels['mu']}_sigma_{labels['sigma']}_T_{labels['T']}"
        f"_PDF.txt"
    )


def write_group_pdf(out_path, labels, species_data, n_files, bin_edges, verbose=False):
    """bin_edges is the SAME shared grid used for every species (see build_bin_edges): this
    is what lets a later averaging step stack per-species histograms directly, with no
    interpolation. Samples falling outside [bin_edges[0], bin_edges[-1]] are silently dropped
    by np.histogram; n_samples still reports the species' true total pooled sample count (not
    just the in-range ones), so a frequency column (count / n_samples) built downstream
    correctly reflects any such missing mass instead of being invisibly rescaled to sum to 1."""
    dn = bin_edges[1] - bin_edges[0]
    with open(out_path, 'w') as fout:
        fout.write("# Empirical probability density of equilibrium abundances, pooled over all extractions and measures\n")
        fout.write("# " + " ".join(f"{key}={labels[key]}" for key in GROUP_KEY_ORDER) + "\n")
        fout.write(f"# n_files_pooled={n_files} nmin={bin_edges[0]:.17g} nmax={bin_edges[-1]:.17g} "
                    f"dn={dn:.17g} bins={len(bin_edges) - 1}\n")
        fout.write("# species_index\tbin_left\tbin_right\tbin_center\tdensity\tcount\tn_samples\n")
        for idx in sorted(species_data.keys()):
            values = np.asarray(species_data[idx], dtype=float)
            if values.size == 0:
                continue
            counts, edges = np.histogram(values, bins=bin_edges)
            n_out_of_range = values.size - int(counts.sum())
            if verbose and n_out_of_range > 0:
                print(f"WARNING: species {idx} in {out_path}: {n_out_of_range} of {values.size} "
                      f"samples fall outside [{bin_edges[0]}, {bin_edges[-1]}] and were dropped",
                      file=sys.stderr)
            density = counts / (values.size * dn)
            centers = 0.5 * (edges[:-1] + edges[1:])
            for count, left, right, center, dens in zip(counts, edges[:-1], edges[1:], centers, density):
                fout.write(f"{idx}\t{left:.17g}\t{right:.17g}\t{center:.17g}\t{dens:.17g}\t{int(count)}\t{values.size}\n")
    return


def build_bin_edges(nmin, nmax, dn):
    """Build the shared grid edges from nmin/nmax/dn, snapping the bin count to the nearest
    integer (nmax-nmin should be an integer multiple of dn, up to floating point rounding)."""
    n_bins = int(round((nmax - nmin) / dn))
    if n_bins < 1:
        print(f"ERROR: --nmax ({nmax}) must be greater than --nmin ({nmin}) by at least one --dn ({dn})",
              file=sys.stderr)
        sys.exit(1)
    actual_span = n_bins * dn
    requested_span = nmax - nmin
    if abs(actual_span - requested_span) > 1e-6 * dn:
        print(f"WARNING: (nmax - nmin) = {requested_span:.6g} is not an integer multiple of dn = {dn:.6g}; "
              f"using {n_bins} bins (span {actual_span:.6g}) instead", file=sys.stderr)
    return np.linspace(nmin, nmin + n_bins * dn, n_bins + 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate Equilibrium_Points files across extractions and measures, and build the "
                    "empirical probability density of the equilibrium abundance of each species, for every "
                    "distinct combination of the other run parameters."
    )
    parser.add_argument('--input-dirs', nargs='+', required=True,
                         help="One or more root directories to search (each expected to contain "
                              "epsilon_*/N_*_c_*/Equilibrium_Points/...). Pass shell globs directly, e.g. "
                              "--input-dirs results_1 results_2 genericname_* ; the shell expands them into "
                              "a list of paths before this script sees them.")
    parser.add_argument('--output-dir', required=True,
                         help="Directory where the aggregated PDF files are written (created if missing).")
    parser.add_argument('--nmin', type=float, required=True,
                         help="Lower edge of the abundance grid, shared by every species and every "
                              "parameter combination (this is what makes later cross-species/cross-file "
                              "averaging exact, with no interpolation needed).")
    parser.add_argument('--nmax', type=float, required=True,
                         help="Upper edge of the shared abundance grid.")
    parser.add_argument('--dn', type=float, required=True,
                         help="Bin width of the shared abundance grid. (nmax - nmin) should be an integer "
                              "multiple of dn (up to floating point rounding); otherwise the bin count is "
                              "rounded and a warning is printed.")
    parser.add_argument('--only-converged', action='store_true',
                         help="Only pool rows whose convergence flag is 'T' (species that reached equilibrium "
                              "in that measure). By default all rows are pooled regardless of convergence flag.")
    parser.add_argument('--verbose', action='store_true',
                         help="Print a warning for every skipped directory/file/line.")
    return parser.parse_args()


def main():
    sys.stdout.reconfigure(line_buffering=True)  # Ensure progress prints show up immediately even when redirected to a file/log
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_edges = build_bin_edges(args.nmin, args.nmax, args.dn)

    groups, group_labels_by_key, n_files_by_key = aggregate(args.input_dirs, args.only_converged, args.verbose)

    if not groups:
        print("No Equilibrium_Points files were found under the given input directories.", file=sys.stderr)
        return 1

    for group_key, species_data in groups.items():
        labels = group_labels_by_key[group_key]
        out_path = out_dir / build_output_name(labels)
        write_group_pdf(out_path, labels, species_data, n_files_by_key[group_key], bin_edges, args.verbose)
        n_species = len(species_data)
        n_samples = sum(len(v) for v in species_data.values())
        print(f"Wrote {out_path} ({n_species} species, {n_samples} pooled samples, "
              f"{n_files_by_key[group_key]} files pooled)")

    print(f"Done: {len(groups)} parameter combination(s) processed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
