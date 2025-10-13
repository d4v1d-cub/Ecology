__author__ = 'david'

import numpy as np
import os
import fnmatch


def is_number(s):
    try:
        float(s)
        if (s == 'nan') or (s == '-nan') or (s == 'inf') or (s == '-inf'):
            return False
        return True
    except ValueError:
        return False


def filter_files(path, eps, lda, tol_fixed_point, N, c, T):
    files_mu_sigma = []

    # Define the pattern for matching filenames
    pattern = f'Lotka-Volterra_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_mu_*_sigma_*_T_{T}_Phase_Diagram*.txt'

    # Iterate through the files in the specified directory
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            # Extract the unique strings from the filename
            parts = filename.split('_')
            files_mu_sigma.append((filename, float(parts[14]), float(parts[16])))
    sorted_data = sorted(files_mu_sigma, key=lambda x: (x[1], x[2]))  # Sort by mu value
    return sorted_data


def read_carefully(str_val, thr):
    if is_number(str_val):
        num = float(str_val)
        if num > thr:
            return thr
        else:
            return float(str_val)
    else:
        return thr


def summary_statistics(path, filename):
    fin = open(f'{path}/{filename}', 'r')
    all_lines = fin.readlines()
    fin.close()
    if len(all_lines) > 0:
        av_time = 0.0
        samples_div = 0
        samples_multiple_eq = 0
        nsamples = 0
        for line in all_lines:
            if not line.startswith('#'):
                line_split = line.split()
                av_time += float(line_split[3])
                if line_split[8] == "1":
                    samples_div += 1
                if line_split[10] == "1" or line_split[8] == "1":
                    samples_multiple_eq += 1
                nsamples += 1
        av_time /= nsamples
        return av_time, samples_div, samples_multiple_eq, nsamples, True
    else:
        print(f"No data found in file {filename}. Returning zeros.")
        return 0.0, 0, 0, 0, False


def get_all_vals(path, eps, lda, tol_fixed_point, N, c, T):
    # Find all files that match the pattern
    sorted_data = filter_files(path, eps, lda, tol_fixed_point, N, c, T)
    vals_list = []
    for filename, mu, sigma in sorted_data:
        av_time, samples_div, samples_multiple_eq, nsamples, found = summary_statistics(path, filename)
        if found:
            vals_list.append((mu, sigma, av_time, samples_div, samples_multiple_eq, nsamples))
        print(f'Processed N={N}   mu={mu}   sigma={sigma}')
    return vals_list




def print_summary(path_in, path_out, eps, lda, tol_fixed_point, N, c, T, ndigits):
    fout = open(f'{path_out}/Lotka-Volterra_summary_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'w')
    fout.write("#mu sigma av_time samples_div samples_mult_eq nsamples\n")
    vals_list = get_all_vals(path_in, eps, lda, tol_fixed_point, N, c, T)
    for vals in vals_list:
        mu, sigma, av_time, samples_div, samples_multiple_eq, nsamples = vals
        fout.write(f"{mu:.{ndigits}f} {sigma:.{ndigits}f} {av_time:.6f} {samples_div} {samples_multiple_eq} {nsamples}\n")
    fout.close()


def main():
    eps = "0.0"
    lda = "1e-06"
    N_list = ["128", "256", "512", "1024", "2048", "4096"]
    c = "3.00"
    T = "0.0"
    tol_fixed_point = "1e-08"

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/AllData/T0/"
    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"

    ndigits = 3

    for N in N_list:
        print_summary(path_in, path_out, eps, lda, tol_fixed_point, N, c, T, ndigits)
    return 0


if __name__ == '__main__':
    main()
