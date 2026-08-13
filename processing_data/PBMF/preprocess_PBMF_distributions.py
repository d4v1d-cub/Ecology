#!/usr/bin/env python
r"""
Fit truncated Gaussians to the modified abundance distributions \hat{P}(n) produced by
the PBMF solver, and quantify how far each \hat{P}(n) is from a truncated Gaussian.

Input files are expected to follow the naming convention:

    PBMF_gr_in_RRG_eps_<eps>_mu_<mu>_sigma_<sigma>_N_<N>_c_<c>_sgraph_<sgraph>_LV_av0_<av0>
    _stdn0_<stdn0>_T_<T>_lda_<lda>_tol_<tol>_maxiter_<maxiter>_damping_<damping>_n1_<n1>
    _dn_<dn>_nmaxlimit_<nmaxlimit>_distributions_seedseq_<seedseq>_seedinit_<seedinit>.txt

and contain a header line followed by whitespace/tab separated columns:
    n, P_1(n), ..., P_N(n), Phat_1(n), ..., Phat_N(n)

For each species column Phat_i(n) we fit A * TruncNorm(n; mu, sigma, a=0, b=inf) and report
several goodness-of-fit / distance-from-Gaussian diagnostics. Two complementary measures of
how close the modified distribution is to a truncated Gaussian are reported in the summary:
(a) fitting each species individually and aggregating (mean, median, std) the per-species
errors, and (b) computing the species-averaged distribution first and fitting a single
truncated Gaussian to that (columns prefixed with "avgdist_"). Summary output is one row per
input file, written as a single fixed-name CSV inside --output-dir.

For each input file, a companion "<file>_avg_dist_fit.txt" file is also written (unless
--skip-avg-dist-files is given), containing the grid, the (species-)averaged normalized P(n),
the (species-)averaged Phat(n), and their pointwise median/std across species.
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
    r"eps_(?P<eps>[0-9.eE+-]+)_mu_(?P<mu>[0-9.eE+-]+)_sigma_(?P<sigma>[0-9.eE+-]+)"
    r"_N_(?P<N>\d+)_c_(?P<c>\d+)_sgraph_(?P<sgraph>\d+)"
    r"_LV_av0_(?P<av0>[0-9.eE+-]+)_stdn0_(?P<stdn0>[0-9.eE+-]+)"
    r"_T_(?P<T>[0-9.eE+-]+)_lda_(?P<lda>[0-9.eE+-]+)_tol_(?P<tol>[0-9.eE+-]+)"
    r"_maxiter_(?P<maxiter>\d+)_damping_(?P<damping>[0-9.eE+-]+)"
    r"_n1_(?P<n1>[0-9.eE+-]+)_dn_(?P<dn>[0-9.eE+-]+)_nmaxlimit_(?P<nmaxlimit>[0-9.eE+-]+)"
    r"_distributions_seedseq_(?P<seedseq>\d+)_seedinit_(?P<seedinit>\d+)\.txt$"
)

DEFAULT_PATTERN = "PBMF_gr_in_RRG_*_distributions_seedseq_*_seedinit_*.txt"


def _build_summary_csv_name():
    """Build the fixed summary CSV filename from the same skeleton as the input
    filenames, dropping every "<label>_<value>" parameter token (each one becomes a
    column in the CSV) while keeping the purely literal/structural parts of the name."""
    skeleton = "PBMF_gr_in_RRG_" + FILENAME_RE.pattern
    skeleton = re.sub(r"(\w+)_\(\?P<\1>[^)]*\)", "", skeleton)
    skeleton = skeleton.replace(r"\.txt$", "")
    skeleton = re.sub(r"_+", "_", skeleton).strip("_")
    return skeleton + ".csv"


SUMMARY_CSV_NAME = _build_summary_csv_name()


def parse_filename(path):
    m = FILENAME_RE.search(os.path.basename(path))
    if m is None:
        return None
    d = {k: float(v) for k, v in m.groupdict().items()}
    for int_key in ("N", "c", "sgraph", "maxiter", "seedseq", "seedinit"):
        d[int_key] = int(d[int_key])
    return d


def load_distributions(path):
    data = np.loadtxt(path, skiprows=1)
    n = data[:, 0]
    ncols = data.shape[1]
    nspecies = (ncols - 1) // 2
    P = data[:, 1:1 + nspecies]
    Phat = data[:, 1 + nspecies:1 + 2 * nspecies]
    return n, P, Phat, nspecies


def _truncnorm_std_bounds(lower, upper, mu, sigma):
    a_std = (lower - mu) / sigma
    b_std = (upper - mu) / sigma
    return a_std, b_std


def fit_species(n, phat, lower=0.0, upper=np.inf):
    """Fit A * TruncNorm(n; mu, sigma, lower, upper) to phat(n) and compute diagnostics.

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
                bounds=([0.0, -np.inf, 1e-8], [np.inf, np.inf, np.inf]),
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


AVG_DIST_SUFFIX = "_avg_dist_fit.txt"


def write_average_distribution_file(path, output_dir, n, avgP, avgPhat,
                                     medianP, medianPhat, stdP, stdPhat):
    out_name = os.path.basename(path)[:-len(".txt")] + AVG_DIST_SUFFIX
    out_path = os.path.join(output_dir, out_name)

    header = "\t".join(
        ["n", "P_avg", "Phat_avg", "P_median", "Phat_median", "P_std", "Phat_std"]
    )

    data = np.column_stack([n, avgP, avgPhat, medianP, medianPhat, stdP, stdPhat])
    np.savetxt(out_path, data,
               fmt=["%.6f", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e"],
               delimiter="\t", header=header, comments="#")
    return out_path


def process_file(path, upper_truncation="inf", avg_dist_dir=None, write_avg_dist_files=True):
    params = parse_filename(path)
    if params is None:
        print(f"  [skip] filename does not match expected pattern: {path}", file=sys.stderr)
        return None, None

    n, P, Phat, nspecies = load_distributions(path)
    upper = np.inf if upper_truncation == "inf" else n[-1]

    per_species_rows = []
    n_ok = 0
    for i in range(nspecies):
        res = fit_species(n, Phat[:, i], lower=0.0, upper=upper)
        if res is None:
            continue
        n_ok += 1
        row = dict(params)
        row["species_index"] = i
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

    avgP_raw = P.mean(axis=1)
    Z_avgP = np.trapezoid(avgP_raw, n)
    avgP = avgP_raw / Z_avgP if Z_avgP > 0 else avgP_raw
    avgPhat = Phat.mean(axis=1)
    medianP = np.median(P, axis=1)
    medianPhat = np.median(Phat, axis=1)
    stdP = np.std(P, axis=1, ddof=1)
    stdPhat = np.std(Phat, axis=1, ddof=1)

    avg_fit = fit_species(n, avgPhat, lower=0.0, upper=upper)
    if avg_fit is None:
        print(f"  [warn] truncated-Gaussian fit to the average distribution failed in {path}",
              file=sys.stderr)
    for key in METRIC_KEYS:
        summary_row[f"avgdist_{key}"] = avg_fit[key] if avg_fit is not None else np.nan

    if write_avg_dist_files:
        write_average_distribution_file(
            path, avg_dist_dir, n, avgP, avgPhat, medianP, medianPhat, stdP, stdPhat
        )

    return summary_row, per_species_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="Folder containing the PBMF distribution files")
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
             "last grid point of each file (grid).",
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
        help="Do not write the per-file average-distribution .txt files. The summary CSV "
             "is produced as usual (including the avgdist_* columns).",
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
    lead_cols = ["T", "mu", "eps", "sigma", "sgraph", "seedinit"]
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
