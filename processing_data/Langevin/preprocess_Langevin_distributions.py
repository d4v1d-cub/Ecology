#!/usr/bin/env python
r"""
Fit truncated Gaussians to the modified abundance distributions \hat{P}(n) built from the
per-species empirical PDF files produced by aggregate_equilibrium_pdfs.py, and quantify how
far each \hat{P}(n) is from a truncated Gaussian. Mirrors preprocess_PBMF_distributions.py so
that the same downstream plotting logic can eventually be reused/adapted for both.

Input files are expected to follow the naming convention produced by aggregate_equilibrium_pdfs.py:

    Lotka-Volterra_epsilon_<eps>_<ia_label>_lambda_<lambda>_h_<h>_tmax_<tmax>_deltatsave_<deltatsave>
    _N_<N>_c_<c>_mu_<mu>_sigma_<sigma>_T_<T>_PDF.txt

and contain, after a few "#"-prefixed header lines, one row per (species_index, histogram bin):
    species_index, bin_left, bin_right, bin_center, density, count, n_samples

Unlike the PBMF solver output (one shared abundance grid n for every species), each species here
has its own histogram bins (same bin *count* across species within a file, but different bin
*edges*, since every species' range is set by its own observed abundances). P(n) is simply the
"density" column (already normalized: integral of density over that species' own bins is 1).

The empirical P(n) built this way is the full stationary distribution, which for this dynamics
includes a known power-law prefactor n^{beta*lambda - 1} (beta = 1/T, lambda = immigration rate /
extinction threshold, both parsed from the filename). That prefactor is divided out by hand to
obtain the modified distribution:

    \hat{P}(n) = P(n) / n^{beta*lambda - 1} / Z

(Z fixed by normalization; matching preprocess_PBMF_distributions.py, Z is computed internally by
fit_species() via trapezoidal integration, so the P(n)/n^{beta*lambda-1} passed to fit_species()
does not need to be pre-normalized.) \hat{P}(n) is only defined for n > 0 (the exponent is
generally non-integer); bins with bin_center <= 0 are dropped before this division.

For each species' \hat{P}_i(n) we fit A * TruncNorm(n; mu, sigma, a=0, b=inf), on that species'
own native grid, and report the same goodness-of-fit / distance-from-Gaussian diagnostics as
preprocess_PBMF_distributions.py. Two complementary measures of how close the modified
distribution is to a truncated Gaussian are reported in the summary: (a) fitting each species
individually and aggregating (mean, median, std) the per-species errors, and (b) computing the
species-averaged distribution first and fitting a single truncated Gaussian to that (columns
prefixed with "avgdist_"). Since species have different native grids, the species-averaged
distribution is built by linearly interpolating every species' P(n)/\hat{P}(n) onto one common
grid (spanning the union of all species' observed ranges, zero outside each species' own range)
before averaging pointwise. Summary output is one row per input file, written as a single
fixed-name CSV inside --output-dir.

For each input file, two companion files are also written (unless --skip-avg-dist-files is
given): a "<file>_avg_dist_fit.txt" file containing the common grid, the (species-)averaged
normalized P(n), the (species-)averaged Phat(n), and their pointwise median/std across species;
and a "<file>_top_nongaussian_species.txt" file containing each selected species' own native n
grid together with its P(n) and Phat(n), for the --top-nongaussian-n (default 3) individual
species with the largest per-species L1 fit error (i.e. the least Gaussian-looking single-species
distributions). Unlike preprocess_PBMF_distributions.py's version of this file (one shared n
column, since all species there share a grid), each species here gets its own n column, since
species generally do not share bin edges.
"""

import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import rel_entr
from scipy.stats import truncnorm

FILENAME_RE = re.compile(
    r"^Lotka-Volterra_epsilon_(?P<epsilon>[^_]+)_(?P<ia_label>.+)_lambda_(?P<lambda_>[^_]+)"
    r"_h_(?P<h>[^_]+)_tmax_(?P<tmax>[^_]+)_deltatsave_(?P<deltatsave>[^_]+)"
    r"_N_(?P<N>\d+)_c_(?P<c>[^_]+)_mu_(?P<mu>[^_]+)_sigma_(?P<sigma>[^_]+)_T_(?P<T>[^_]+)"
    r"_PDF\.txt$"
)

DEFAULT_PATTERN = "Lotka-Volterra_*_PDF.txt"

# Columns written by aggregate_equilibrium_pdfs.py's write_group_pdf()
PDF_COLUMNS = ["species_index", "bin_left", "bin_right", "bin_center", "density", "count", "n_samples"]


def _build_summary_csv_name():
    """Build the fixed summary CSV filename from the same skeleton as the input filenames,
    dropping every "<label>_<value>" parameter token (each one becomes a column in the CSV)
    while keeping the purely literal/structural parts of the name."""
    skeleton = FILENAME_RE.pattern.lstrip("^")
    skeleton = re.sub(r"\(\?P<\w+>[^)]*\)", "", skeleton)
    skeleton = skeleton.replace(r"\.txt$", "")
    skeleton = re.sub(r"_+", "_", skeleton).strip("_")
    return skeleton + ".csv"


SUMMARY_CSV_NAME = _build_summary_csv_name()


def parse_filename(path):
    m = FILENAME_RE.match(os.path.basename(path))
    if m is None:
        return None
    d = dict(m.groupdict())
    d["ia_label"] = d.pop("ia_label")
    d["lambda"] = d.pop("lambda_")
    for key in ("epsilon", "lambda", "h", "tmax", "deltatsave", "c", "mu", "sigma", "T"):
        d[key] = float(d[key])
    d["N"] = int(d["N"])
    return d


def load_species_pdfs(path):
    """Return dict: species_index -> DataFrame with columns bin_left/bin_right/bin_center/
    density/count/n_samples, sorted by bin_center, for every species found in the file."""
    df = pd.read_csv(path, sep="\t", comment="#", header=None, names=PDF_COLUMNS)
    species_data = {}
    for idx, sub in df.groupby("species_index"):
        species_data[int(idx)] = sub.sort_values("bin_center").reset_index(drop=True)
    return species_data


def _truncnorm_std_bounds(lower, upper, mu, sigma):
    a_std = (lower - mu) / sigma
    b_std = (upper - mu) / sigma
    return a_std, b_std


def fit_species(n, phat, lower=0.0, upper=np.inf):
    """Fit A * TruncNorm(n; mu, sigma, lower, upper) to phat(n) and compute diagnostics.

    Mirrors preprocess_PBMF_distributions.py's fit_species() (kept in sync manually rather
    than imported, since the two scripts live in separate, otherwise independent, folders).

    mu is constrained to mu >= lower: abundances (and their average) are non-negative, so a
    fitted mean below the physical truncation point is not a meaningful result -- without this
    bound a handful of poorly-conditioned species (e.g. flat/noisy phat) can pull curve_fit
    into a run-away negative-mu, huge-sigma degenerate solution.

    Returns a dict of results, or None if the fit could not be performed.
    """
    phat = np.clip(phat, 0.0, None)
    if not np.all(np.isfinite(phat)):
        return None

    Z = np.trapezoid(phat, n)
    if not np.isfinite(Z) or Z <= 0:
        return None

    p_emp = phat / Z
    mean0 = np.trapezoid(n * p_emp, n)
    var0 = np.trapezoid((n - mean0) ** 2 * p_emp, n)
    sigma0 = np.sqrt(var0) if var0 > 0 else 0.1
    A0 = Z

    def model(nn, A, mu, sigma):
        sigma = abs(sigma)
        a_std, b_std = _truncnorm_std_bounds(lower, upper, mu, sigma)
        return A * truncnorm.pdf(nn, a_std, b_std, loc=mu, scale=sigma)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                model, n, phat,
                p0=[A0, mean0, sigma0],
                bounds=([0.0, lower, 1e-8], [np.inf, np.inf, np.inf]),
                maxfev=5000,
            )
    except (RuntimeError, ValueError):
        return None

    A_fit, mu_fit, sigma_fit = popt
    sigma_fit = abs(sigma_fit)
    a_std, b_std = _truncnorm_std_bounds(lower, upper, mu_fit, sigma_fit)

    model_vals = A_fit * truncnorm.pdf(n, a_std, b_std, loc=mu_fit, scale=sigma_fit)

    denom = np.trapezoid(np.abs(phat), n)
    if denom <= 0:
        return None
    error_fit = np.trapezoid(np.abs(phat - model_vals), n) / denom

    p_fit_pdf = truncnorm.pdf(n, a_std, b_std, loc=mu_fit, scale=sigma_fit)

    mean_emp = mean0
    var_emp = var0
    if var_emp <= 0:
        return None
    skew_emp = np.trapezoid((n - mean_emp) ** 3 * p_emp, n) / var_emp ** 1.5
    kurt_emp = np.trapezoid((n - mean_emp) ** 4 * p_emp, n) / var_emp ** 2 - 3.0

    mean_t, var_t, skew_t, kurt_t = truncnorm.stats(
        a_std, b_std, loc=mu_fit, scale=sigma_fit, moments="mvsk"
    )

    error_skewness = skew_emp - float(skew_t)
    error_kurtosis = kurt_emp - float(kurt_t)

    m = 0.5 * (p_emp + p_fit_pdf)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.trapezoid(rel_entr(p_emp, m), n)
        kl_qm = np.trapezoid(rel_entr(p_fit_pdf, m), n)
    js_divergence = 0.5 * (kl_pm + kl_qm)

    cdf_emp = np.concatenate(([0.0], np.cumsum(
        0.5 * (p_emp[1:] + p_emp[:-1]) * np.diff(n)
    )))
    if cdf_emp[-1] > 0:
        cdf_emp = cdf_emp / cdf_emp[-1]
    cdf_fit = truncnorm.cdf(n, a_std, b_std, loc=mu_fit, scale=sigma_fit)
    ks_stat = np.max(np.abs(cdf_emp - cdf_fit))

    mask = phat > 0
    log_quadratic_r2 = np.nan
    if mask.sum() >= 4:
        logp = np.log(phat[mask])
        coeffs = np.polyfit(n[mask], logp, 2)
        pred = np.polyval(coeffs, n[mask])
        ss_res = np.sum((logp - pred) ** 2)
        ss_tot = np.sum((logp - logp.mean()) ** 2)
        if ss_tot > 0:
            log_quadratic_r2 = 1.0 - ss_res / ss_tot

    return {
        "mu_fit": mu_fit,
        "sigma_fit": sigma_fit,
        "amplitude_fit": A_fit,
        "mean_emp": mean_emp,
        "std_emp": np.sqrt(var_emp),
        "error_fit": error_fit,
        "error_skewness": error_skewness,
        "error_kurtosis": error_kurtosis,
        "js_divergence": js_divergence,
        "ks_stat": ks_stat,
        "log_quadratic_r2": log_quadratic_r2,
    }


METRIC_KEYS = [
    "mu_fit", "sigma_fit", "amplitude_fit", "mean_emp", "std_emp",
    "error_fit", "error_skewness", "error_kurtosis",
    "js_divergence", "ks_stat", "log_quadratic_r2",
]


def species_P_and_Phat(species_df, power):
    """Return (n, P, Phat) for one species, restricted to bins with bin_center > 0 (n^power
    is only defined there). P is the raw "density" column; Phat = P / n^power."""
    n = species_df["bin_center"].to_numpy(dtype=float)
    P = species_df["density"].to_numpy(dtype=float)
    mask = n > 0
    n, P = n[mask], P[mask]
    if n.size == 0:
        return n, P, P
    with np.errstate(divide="ignore", invalid="ignore"):
        Phat = P / n ** power
    finite = np.isfinite(Phat)
    return n[finite], P[finite], Phat[finite]


AVG_DIST_SUFFIX = "_avg_dist_fit.txt"


def interpolate_to_grid(n, values, grid):
    """Linearly interpolate one species' (n, values) onto grid; 0 outside the species'
    own observed [min(n), max(n)] range (we have no information there)."""
    if n.size == 0:
        return np.zeros_like(grid)
    return np.interp(grid, n, values, left=0.0, right=0.0)


def build_common_grid(per_species_np, n_grid_points):
    all_n = np.concatenate([n for n, _, _ in per_species_np.values() if n.size > 0])
    if all_n.size == 0:
        return None
    return np.linspace(all_n.min(), all_n.max(), n_grid_points)


def write_average_distribution_file(path, output_dir, grid, avgP, avgPhat,
                                     medianP, medianPhat, stdP, stdPhat):
    out_name = os.path.basename(path)[:-len(".txt")] + AVG_DIST_SUFFIX
    out_path = os.path.join(output_dir, out_name)

    header = "\t".join(
        ["n", "P_avg", "Phat_avg", "P_median", "Phat_median", "P_std", "Phat_std"]
    )

    data = np.column_stack([grid, avgP, avgPhat, medianP, medianPhat, stdP, stdPhat])
    np.savetxt(out_path, data,
               fmt=["%.6f", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e"],
               delimiter="\t", header=header, comments="#")
    return out_path


TOP_NONGAUSSIAN_SUFFIX = "_top_nongaussian_species.txt"


def write_top_nongaussian_file(path, output_dir, per_species_np, per_species_df, n_top=3):
    """Write, for the n_top species with the largest per-species L1 fit error (error_fit),
    that species' own native n grid together with its P(n) and Phat(n). One header comment
    line per selected species (rank, species number, and all its fit diagnostics) precedes
    the column-name header line. Unlike preprocess_PBMF_distributions.py's version (one
    shared n column), each species gets its own n column here, since species generally do
    not share bin edges."""
    out_name = os.path.basename(path)[:-len(".txt")] + TOP_NONGAUSSIAN_SUFFIX
    out_path = os.path.join(output_dir, out_name)

    ranked = per_species_df.sort_values("error_fit", ascending=False).head(n_top)

    header_lines = []
    columns = []
    data_cols = []
    max_len = 0
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        species_num = int(row["species_index"]) + 1
        metrics = " ".join(f"{key}={row[key]:.6e}" for key in METRIC_KEYS)
        header_lines.append(f"rank={rank} species={species_num} {metrics}")
        n_i, P_i, Phat_i = per_species_np[species_num - 1]
        columns += [f"n_species{species_num}", f"P_species{species_num}", f"Phat_species{species_num}"]
        data_cols += [n_i, P_i, Phat_i]
        max_len = max(max_len, n_i.size)

    # Species can have a different number of valid (bin_center > 0) bins; pad the shorter
    # columns with NaN so they can share one rectangular file.
    padded_cols = []
    for col in data_cols:
        if col.size < max_len:
            col = np.concatenate([col, np.full(max_len - col.size, np.nan)])
        padded_cols.append(col)

    header = "\n".join(header_lines) + "\n" + "\t".join(columns)
    data = np.column_stack(padded_cols)
    fmt = ["%.6e"] * len(padded_cols)
    np.savetxt(out_path, data, fmt=fmt, delimiter="\t", header=header, comments="#")
    return out_path


def process_file(path, upper_truncation="inf", avg_dist_dir=None, write_avg_dist_files=True,
                  top_nongaussian_n=3, grid_points=500):
    params = parse_filename(path)
    if params is None:
        print(f"  [skip] filename does not match expected pattern: {path}", file=sys.stderr)
        return None, None

    beta = 1.0 / params["T"] if params["T"] != 0 else None
    if beta is None:
        print(f"  [skip] T=0 in filename, beta=1/T undefined: {path}", file=sys.stderr)
        return None, None
    power = beta * params["lambda"] - 1.0

    species_data = load_species_pdfs(path)
    if not species_data:
        print(f"  [warn] no species found in {path}", file=sys.stderr)
        return None, None
    nspecies = len(species_data)

    per_species_np = {}  # species_index -> (n, P, Phat), bin_center > 0 only
    for idx, sdf in species_data.items():
        per_species_np[idx] = species_P_and_Phat(sdf, power)

    per_species_rows = []
    n_ok = 0
    for idx, (n, _P, Phat) in per_species_np.items():
        if n.size == 0:
            continue
        upper = np.inf if upper_truncation == "inf" else n[-1]
        res = fit_species(n, Phat, lower=0.0, upper=upper)
        if res is None:
            continue
        n_ok += 1
        row = dict(params)
        row["species_index"] = idx
        row.update(res)
        per_species_rows.append(row)

    if n_ok == 0:
        print(f"  [warn] no species could be fit in {path}", file=sys.stderr)
        return None, None

    per_species_df = pd.DataFrame(per_species_rows)

    summary_row = dict(params)
    summary_row["n_species_total"] = nspecies
    summary_row["n_species_fit_ok"] = n_ok
    for key in METRIC_KEYS:
        vals = per_species_df[key].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        summary_row[f"{key}_mean"] = np.mean(vals) if vals.size else np.nan
        summary_row[f"{key}_median"] = np.median(vals) if vals.size else np.nan
        summary_row[f"{key}_std"] = np.std(vals, ddof=1) if vals.size > 1 else np.nan

    grid = build_common_grid(per_species_np, grid_points)
    avg_fit = None
    if grid is not None:
        P_grid = np.stack([interpolate_to_grid(n, P, grid) for n, P, _ in per_species_np.values()])
        Phat_grid = np.stack([interpolate_to_grid(n, Phat, grid) for n, _, Phat in per_species_np.values()])

        avgP_raw = P_grid.mean(axis=0)
        Z_avgP = np.trapezoid(avgP_raw, grid)
        avgP = avgP_raw / Z_avgP if Z_avgP > 0 else avgP_raw
        avgPhat = Phat_grid.mean(axis=0)
        medianP = np.median(P_grid, axis=0)
        medianPhat = np.median(Phat_grid, axis=0)
        stdP = np.std(P_grid, axis=0, ddof=1)
        stdPhat = np.std(Phat_grid, axis=0, ddof=1)

        upper = np.inf if upper_truncation == "inf" else grid[-1]
        avg_fit = fit_species(grid, avgPhat, lower=0.0, upper=upper)
        if avg_fit is None:
            print(f"  [warn] truncated-Gaussian fit to the average distribution failed in {path}",
                  file=sys.stderr)
    else:
        print(f"  [warn] could not build a common grid (no species with bin_center > 0) in {path}",
              file=sys.stderr)

    for key in METRIC_KEYS:
        summary_row[f"avgdist_{key}"] = avg_fit[key] if avg_fit is not None else np.nan

    if write_avg_dist_files and grid is not None:
        write_average_distribution_file(
            path, avg_dist_dir, grid, avgP, avgPhat, medianP, medianPhat, stdP, stdPhat
        )
        write_top_nongaussian_file(
            path, avg_dist_dir, per_species_np, per_species_df, n_top=top_nongaussian_n
        )

    return summary_row, per_species_df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", help="Folder containing the aggregate_equilibrium_pdfs.py output files")
    parser.add_argument(
        "--output-dir", default=".",
        help=f"Directory to write the summary CSV to. Its filename is fixed: "
             f"{SUMMARY_CSV_NAME} (default: %(default)s)",
    )
    parser.add_argument(
        "--pattern", default=DEFAULT_PATTERN,
        help="Glob pattern (relative to input_dir) used to select files (default: %(default)s)",
    )
    parser.add_argument(
        "--upper-truncation", choices=["inf", "grid"], default="inf",
        help="Upper bound of the truncated Gaussian: physical infinity (default) or the "
             "last grid point of each file/species (grid).",
    )
    parser.add_argument(
        "--per-species-output", default=None,
        help="Optional path to also write a long-format CSV with one row per species per file.",
    )
    parser.add_argument(
        "--avg-dist-dir", default=None,
        help="Directory to write the per-file average-distribution .txt files to "
             "(default: same as --output-dir).",
    )
    parser.add_argument(
        "--skip-avg-dist-files", action="store_true",
        help="Do not write the per-file average-distribution and top-nongaussian-species "
             ".txt files. The summary CSV is produced as usual (including the avgdist_* "
             "columns).",
    )
    parser.add_argument(
        "--top-nongaussian-n", type=int, default=3,
        help="Number of individual species (largest per-species L1 fit error) to include "
             "in the per-file top-nongaussian-species .txt output (default: %(default)s).",
    )
    parser.add_argument(
        "--grid-points", type=int, default=500,
        help="Number of points in the common grid used to interpolate species before "
             "computing the species-averaged distribution (default: %(default)s).",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        print(f"No files matching '{args.pattern}' found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    avg_dist_dir = args.avg_dist_dir or args.output_dir
    if not args.skip_avg_dist_files:
        os.makedirs(avg_dist_dir, exist_ok=True)

    summary_rows = []
    per_species_frames = []
    for path in files:
        print(f"Processing {os.path.basename(path)} ...")
        summary_row, per_species_df = process_file(
            path, upper_truncation=args.upper_truncation, avg_dist_dir=avg_dist_dir,
            write_avg_dist_files=not args.skip_avg_dist_files,
            top_nongaussian_n=args.top_nongaussian_n, grid_points=args.grid_points,
        )
        if summary_row is None:
            continue
        summary_rows.append(summary_row)
        if args.per_species_output is not None:
            per_species_frames.append(per_species_df)

    if not summary_rows:
        print("No files could be processed successfully.", file=sys.stderr)
        sys.exit(1)

    summary_df = pd.DataFrame(summary_rows)
    lead_cols = ["T", "mu", "epsilon", "sigma", "N", "c"]
    other_cols = [c for c in summary_df.columns if c not in lead_cols]
    summary_df = summary_df[lead_cols + other_cols]
    output_path = os.path.join(args.output_dir, SUMMARY_CSV_NAME)
    summary_df.to_csv(output_path, sep="\t", index=False)
    print(f"Wrote summary for {len(summary_df)} file(s) to {output_path}")

    if args.per_species_output is not None:
        pd.concat(per_species_frames, ignore_index=True).to_csv(
            args.per_species_output, sep="\t", index=False
        )
        print(f"Wrote per-species results to {args.per_species_output}")


if __name__ == "__main__":
    main()
