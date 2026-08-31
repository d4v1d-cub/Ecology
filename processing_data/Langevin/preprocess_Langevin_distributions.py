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

Every species in the file shares the SAME grid (bin_left/bin_right/bin_center are identical
across species -- aggregate_equilibrium_pdfs.py builds them from --nmin/--nmax/--dn), so no
interpolation is needed anywhere in this script: per-species arrays can be stacked directly.
P(n) is simply the "density" column (already normalized: integral of density over the grid is
1, modulo any probability mass that fell outside [nmin, nmax] and was dropped when the file was
built). count/n_samples are the raw per-bin sample count and that species' total pooled sample
count; frequency = count / n_samples is the fraction of that species' samples falling in the
bin (also always available, exact, and simpler to reason about than density since it does not
depend on the bin width).

The empirical P(n) built this way is the full stationary distribution, which for this dynamics
includes a known power-law prefactor n^{beta*lambda - 1} (beta = 1/T, lambda = immigration rate /
extinction threshold, both parsed from the filename). That prefactor is divided out by hand to
obtain the modified distribution:

    \hat{P}(n) = P(n) / n^{beta*lambda - 1} / Z

(Z fixed by normalization; matching preprocess_PBMF_distributions.py, Z is computed internally by
fit_species() via trapezoidal integration, so the P(n)/n^{beta*lambda-1} passed to fit_species()
does not need to be pre-normalized.) \hat{P}(n) is only defined for n > 0 (the exponent is
generally non-integer); grid points with n <= 0 are excluded before this division (dropped
entirely for the per-species fit, kept as NaN placeholders -- at the same grid index for every
species -- everywhere the modified distribution is tabulated/averaged, so the grid stays aligned).

For each species' \hat{P}_i(n) we fit A * TruncNorm(n; mu, sigma, a=0, b=inf) and report the same
goodness-of-fit / distance-from-Gaussian diagnostics as preprocess_PBMF_distributions.py. Two
complementary measures of how close the modified distribution is to a truncated Gaussian are
reported in the summary: (a) fitting each species individually and aggregating (mean, median,
std) the per-species errors, and (b) computing the species-averaged distribution first and
fitting a single truncated Gaussian to that (columns prefixed with "avgdist_"). Summary output is
one row per input file, written as a single fixed-name CSV inside --output-dir.

For each input file, an "<file>_avg_dist_fit.txt" companion file is also written (unless
--skip-avg-dist-files is given), with the shared grid and the mean/median/std across species of
every representation this script tracks: for P, count, frequency, and density; for \hat{P},
frequency and density (there is no natural "count" for \hat{P}, since it is a rescaled density,
not itself a sample count). All of them are written so that plotting can pick whichever is most
useful later.

If --pbmf-dir is given, an additional "<file>_PBMF_top_nongaussian_species.txt" companion file
is written: it pools every PBMF *_top_nongaussian_species.txt file (see
preprocess_PBMF_distributions.py) in that directory matching this file's (epsilon, sigma) --
across every sgraph (graph realization) and seedinit (initial condition) PBMF was run with -- to
find the set of species PBMF identified as top non-Gaussian in any of those runs, and writes
each such species' own count/frequency/density (from the Langevin data, on the shared grid) to
that file.
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
    density/count/n_samples, sorted by bin_center, for every species found in the file.
    Every species' bin_left/bin_right/bin_center are identical (the shared grid built by
    aggregate_equilibrium_pdfs.py), so the returned DataFrames can be stacked directly."""
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


def species_quantities(species_df, power):
    """Return (grid, count, freq, density, phat_density) for one species, all aligned to the
    file's full shared grid (same length/values for every species in the file):
      - count, freq (= count / n_samples), density: defined everywhere, straight from the file.
      - phat_density (= density / n^power): NaN at grid points with n <= 0 (the exponent is
        generally non-integer, so undefined there) -- NaN rather than dropped, so every
        species' arrays stay aligned to the same grid index for stacking/averaging.
    """
    grid = species_df["bin_center"].to_numpy(dtype=float)
    count = species_df["count"].to_numpy(dtype=float)
    n_samples = species_df["n_samples"].to_numpy(dtype=float)
    density = species_df["density"].to_numpy(dtype=float)
    freq = count / n_samples

    positive = grid > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        phat_density = np.where(positive, density / np.where(positive, grid, 1.0) ** power, np.nan)
    phat_density = np.where(np.isfinite(phat_density), phat_density, np.nan)
    return grid, count, freq, density, phat_density


AVG_DIST_SUFFIX = "_avg_dist_fit.txt"


def write_average_distribution_file(path, output_dir, grid,
                                     count_avg, count_median, count_std,
                                     freq_avg, freq_median, freq_std,
                                     density_avg, density_median, density_std,
                                     phat_freq_avg, phat_freq_median, phat_freq_std,
                                     phat_density_avg, phat_density_median, phat_density_std):
    out_name = os.path.basename(path)[:-len(".txt")] + AVG_DIST_SUFFIX
    out_path = os.path.join(output_dir, out_name)

    columns = [
        "n",
        "P_count_avg", "P_count_median", "P_count_std",
        "P_freq_avg", "P_freq_median", "P_freq_std",
        "P_density_avg", "P_density_median", "P_density_std",
        "Phat_freq_avg", "Phat_freq_median", "Phat_freq_std",
        "Phat_density_avg", "Phat_density_median", "Phat_density_std",
    ]
    data = np.column_stack([
        grid,
        count_avg, count_median, count_std,
        freq_avg, freq_median, freq_std,
        density_avg, density_median, density_std,
        phat_freq_avg, phat_freq_median, phat_freq_std,
        phat_density_avg, phat_density_median, phat_density_std,
    ])
    fmt = ["%.6f"] + ["%.6e"] * (len(columns) - 1)
    np.savetxt(out_path, data, fmt=fmt, delimiter="\t", header="\t".join(columns), comments="#")
    return out_path


PBMF_TOP_NONGAUSSIAN_SUFFIX = "_PBMF_top_nongaussian_species.txt"

# Matches preprocess_PBMF_distributions.py's FILENAME_RE, with its ".txt" ending replaced by
# "_top_nongaussian_species.txt" (the suffix that script appends to build that companion file's
# name from the same base filename).
PBMF_TOP_NONGAUSSIAN_RE = re.compile(
    r"^PBMF_gr_in_RRG_eps_(?P<eps>[0-9.eE+-]+)_mu_(?P<mu>[0-9.eE+-]+)_sigma_(?P<sigma>[0-9.eE+-]+)"
    r"_N_(?P<N>\d+)_c_(?P<c>\d+)_sgraph_(?P<sgraph>\d+)"
    r"_LV_av0_(?P<av0>[0-9.eE+-]+)_stdn0_(?P<stdn0>[0-9.eE+-]+)"
    r"_T_(?P<T>[0-9.eE+-]+)_lda_(?P<lda>[0-9.eE+-]+)_tol_(?P<tol>[0-9.eE+-]+)"
    r"_maxiter_(?P<maxiter>\d+)_damping_(?P<damping>[0-9.eE+-]+)"
    r"_n1_(?P<n1>[0-9.eE+-]+)_dn_(?P<dn>[0-9.eE+-]+)_nmaxlimit_(?P<nmaxlimit>[0-9.eE+-]+)"
    r"_distributions_seedseq_(?P<seedseq>\d+)_seedinit_(?P<seedinit>\d+)"
    r"_top_nongaussian_species\.txt$"
)

PBMF_TOP_NONGAUSSIAN_SPECIES_LINE_RE = re.compile(r"species=(\d+)")


def find_pbmf_top_nongaussian_files(pbmf_dir, eps, sigma, tol=1e-6):
    """Find every PBMF *_top_nongaussian_species.txt file in pbmf_dir matching (eps, sigma),
    pooling over every sgraph (graph realization) and seedinit (initial condition) PBMF was
    run with."""
    pattern = os.path.join(pbmf_dir, "PBMF_gr_in_RRG_*_top_nongaussian_species.txt")
    matches = []
    for path in sorted(glob.glob(pattern)):
        m = PBMF_TOP_NONGAUSSIAN_RE.match(os.path.basename(path))
        if m is None:
            continue
        if np.isclose(float(m["eps"]), eps, atol=tol) and np.isclose(float(m["sigma"]), sigma, atol=tol):
            matches.append(path)
    return matches


def extract_pbmf_top_nongaussian_species(paths):
    """Return the sorted set of (0-indexed) species that appear as a "top non-Gaussian"
    species in the header of any of the given PBMF *_top_nongaussian_species.txt files.

    preprocess_PBMF_distributions.py writes one "#rank=<r> species=<s> ..." header comment
    line per selected species, with s = species_index + 1 (1-indexed) -- subtracted back to
    0-indexed here to match this script's own species_index convention.
    """
    species = set()
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                m = PBMF_TOP_NONGAUSSIAN_SPECIES_LINE_RE.search(line)
                if m:
                    species.add(int(m.group(1)) - 1)
    return sorted(species)


def write_pbmf_top_nongaussian_file(path, output_dir, species_indices, grid,
                                     count_by_species, freq_by_species, density_by_species,
                                     pbmf_paths):
    """Write, for each PBMF-identified top-non-Gaussian species, that species' own
    count/frequency/density on the shared grid (all species in a Langevin PDF file already
    share one grid, so no interpolation is needed, unlike PBMF's own per-species outputs)."""
    out_name = os.path.basename(path)[:-len(".txt")] + PBMF_TOP_NONGAUSSIAN_SUFFIX
    out_path = os.path.join(output_dir, out_name)

    header_lines = [
        "PBMF-selected species (0-indexed): " + ", ".join(str(idx) for idx in species_indices),
        f"pooled from {len(pbmf_paths)} PBMF file(s): "
        + ", ".join(os.path.basename(p) for p in pbmf_paths),
    ]

    columns = ["n"]
    data_cols = [grid]
    for idx in species_indices:
        columns += [f"count_species{idx}", f"freq_species{idx}", f"density_species{idx}"]
        data_cols += [count_by_species[idx], freq_by_species[idx], density_by_species[idx]]

    header = "\n".join(header_lines) + "\n" + "\t".join(columns)
    data = np.column_stack(data_cols)
    fmt = ["%.6f"] + ["%.6e"] * (len(data_cols) - 1)
    np.savetxt(out_path, data, fmt=fmt, delimiter="\t", header=header, comments="#")
    return out_path


def process_file(path, upper_truncation="inf", avg_dist_dir=None, write_avg_dist_files=True,
                  pbmf_dir=None, pbmf_tol=1e-6):
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

    grid = None
    count_by_species, freq_by_species, density_by_species, phat_density_by_species = {}, {}, {}, {}
    for idx, sdf in species_data.items():
        g, count, freq, density, phat_density = species_quantities(sdf, power)
        if grid is None:
            grid = g
        count_by_species[idx] = count
        freq_by_species[idx] = freq
        density_by_species[idx] = density
        phat_density_by_species[idx] = phat_density

    per_species_rows = []
    n_ok = 0
    for idx in species_data:
        phat_density = phat_density_by_species[idx]
        valid = np.isfinite(phat_density)
        n_fit = grid[valid]
        phat_fit = phat_density[valid]
        if n_fit.size == 0:
            continue
        upper = np.inf if upper_truncation == "inf" else n_fit[-1]
        res = fit_species(n_fit, phat_fit, lower=0.0, upper=upper)
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

    # Every species shares the same grid, so these stack directly -- no interpolation needed.
    count_mat = np.stack([count_by_species[idx] for idx in sorted(count_by_species)])
    freq_mat = np.stack([freq_by_species[idx] for idx in sorted(freq_by_species)])
    density_mat = np.stack([density_by_species[idx] for idx in sorted(density_by_species)])
    phat_density_mat = np.stack([phat_density_by_species[idx] for idx in sorted(phat_density_by_species)])

    count_avg, count_median, count_std = count_mat.mean(0), np.median(count_mat, 0), count_mat.std(0, ddof=1)
    freq_avg, freq_median, freq_std = freq_mat.mean(0), np.median(freq_mat, 0), freq_mat.std(0, ddof=1)
    density_avg, density_median, density_std = density_mat.mean(0), np.median(density_mat, 0), density_mat.std(0, ddof=1)

    # phat_density_mat has the same NaNs (n <= 0) at the same grid index for every species, so
    # nanmean/nanmedian/nanstd reduce to plain mean/median/std over the valid sub-range and
    # reproduce those same NaNs in the output, aligned with the P columns above.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice at n <= 0 grid points
        phat_density_avg = np.nanmean(phat_density_mat, axis=0)
        phat_density_median = np.nanmedian(phat_density_mat, axis=0)
        phat_density_std = np.nanstd(phat_density_mat, axis=0, ddof=1)
    dn = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
    phat_freq_avg = phat_density_avg * dn
    phat_freq_median = phat_density_median * dn
    phat_freq_std = phat_density_std * dn

    avg_valid = np.isfinite(phat_density_avg)
    upper = np.inf if upper_truncation == "inf" else grid[avg_valid][-1]
    avg_fit = fit_species(grid[avg_valid], phat_density_avg[avg_valid], lower=0.0, upper=upper)
    if avg_fit is None:
        print(f"  [warn] truncated-Gaussian fit to the average distribution failed in {path}",
              file=sys.stderr)
    for key in METRIC_KEYS:
        summary_row[f"avgdist_{key}"] = avg_fit[key] if avg_fit is not None else np.nan

    if write_avg_dist_files:
        write_average_distribution_file(
            path, avg_dist_dir, grid,
            count_avg, count_median, count_std,
            freq_avg, freq_median, freq_std,
            density_avg, density_median, density_std,
            phat_freq_avg, phat_freq_median, phat_freq_std,
            phat_density_avg, phat_density_median, phat_density_std,
        )

        if pbmf_dir is not None:
            pbmf_paths = find_pbmf_top_nongaussian_files(pbmf_dir, params["epsilon"], params["sigma"], tol=pbmf_tol)
            if not pbmf_paths:
                print(f"  [warn] no PBMF top-nongaussian files found for epsilon={params['epsilon']}, "
                      f"sigma={params['sigma']} in {pbmf_dir}", file=sys.stderr)
            else:
                pbmf_species = extract_pbmf_top_nongaussian_species(pbmf_paths)
                valid_species = [idx for idx in pbmf_species if idx in species_data]
                missing = sorted(set(pbmf_species) - set(valid_species))
                if missing:
                    print(f"  [warn] PBMF top-nongaussian species {missing} not present in {path}",
                          file=sys.stderr)
                if valid_species:
                    write_pbmf_top_nongaussian_file(
                        path, avg_dist_dir, valid_species, grid,
                        count_by_species, freq_by_species, density_by_species, pbmf_paths,
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
             "last (n > 0) grid point (grid).",
    )
    parser.add_argument(
        "--per-species-output", default=None,
        help="Optional path to also write a long-format CSV with one row per species per file.",
    )
    parser.add_argument(
        "--avg-dist-dir", default=None,
        help="Directory to write the per-file average-distribution (and, if --pbmf-dir is "
             "given, PBMF-top-nongaussian) .txt files to (default: same as --output-dir).",
    )
    parser.add_argument(
        "--skip-avg-dist-files", action="store_true",
        help="Do not write the per-file average-distribution or PBMF-top-nongaussian .txt "
             "files. The summary CSV is produced as usual (including the avgdist_* columns).",
    )
    parser.add_argument(
        "--pbmf-dir", default=None,
        help="Directory containing PBMF's *_top_nongaussian_species.txt files (see "
             "preprocess_PBMF_distributions.py). When given, for each input file this pools "
             "every PBMF file matching that file's (epsilon, sigma) -- across every sgraph "
             "(graph realization) and seedinit (initial condition) -- to find the species PBMF "
             "identified as top non-Gaussian in any of those runs, and writes a "
             "'<file>_PBMF_top_nongaussian_species.txt' companion file with each such species' "
             "own count/frequency/density (from the Langevin data). Omit to skip this entirely.",
    )
    parser.add_argument(
        "--pbmf-tol", type=float, default=1e-6,
        help="Absolute tolerance used to match (epsilon, sigma) between the input file and "
             "PBMF files when --pbmf-dir is given (default: %(default)s).",
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
            pbmf_dir=args.pbmf_dir, pbmf_tol=args.pbmf_tol,
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
