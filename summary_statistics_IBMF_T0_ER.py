__author__ = 'david'

import numpy as np
import os
import fnmatch



def filter_files(path, eps, avn0, tol, max_iter):
    files_mu_sigma = []

    # Define the pattern for matching filenames
    pattern = f'IBMF_T0_ER_PD_Lotka_Volterra_final_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_mu_*_sigma_*_N_*_c_*.txt'

    # Iterate through the files in the specified directory
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            parts = filename.split('_')
            files_mu_sigma.append((filename, int(parts[20]), float(parts[22][:-4]), float(parts[16]), float(parts[18])))
    sorted_pairs = sorted(files_mu_sigma, key=lambda x: (x[1], x[2], x[3], x[4]))  # Sort by mu value
    return sorted_pairs


def summary_statistics(path, filename):
    fin = open(f'{path}/{filename}', 'r')
    fin.readline()  # Skip header line
    all_lines = fin.readlines()
    fin.close()
    if len(all_lines) > 0:
        av_time = 0.0
        av_num_div = 0.0
        samples_div_m = 0
        av_m = 0.0
        av_m_sqr = 0.0
        for line in all_lines:
            line_split = line.split()
            av_time += int(line_split[0])
            av_num_div += int(line_split[4])
            if line_split[1] != "1":
                samples_div_m += 1
            av_m += float(line_split[2])
            av_m_sqr += float(line_split[2]) * float(line_split[2])
        av_time /= len(all_lines)
        av_num_div /= len(all_lines)
        av_m /= len(all_lines)
        av_m_sqr /= len(all_lines)
        std_av_m = np.sqrt(av_m_sqr - av_m * av_m)
        return av_time, av_num_div, samples_div_m, len(all_lines), av_m, std_av_m, True
    else:
        print(f"No data found in file {filename}. Returning zeros.")
        return 0.0, 0.0, 0, 0, 0.0, 0.0, False


def get_all_vals(path, eps, avn0, tol, max_iter):
    # Find all files that match the pattern
    sorted_data = filter_files(path, eps, avn0, tol, max_iter)
    vals_list = []
    for filename, N, c, mu, sigma in sorted_data:
        av_time, av_num_div, samples_div_m, nsamples, av_m, std_av_m, found = summary_statistics(path, filename)
        if found:
            vals_list.append((N, c, mu, sigma, av_time, av_num_div, samples_div_m, nsamples, av_m, std_av_m))
        print(f'Processed N={N}  c={c}  mu={mu}   sigma={sigma}')
    return vals_list




def print_summary(path_in, path_out, eps, avn0, tol, max_iter, ndigits):
    fout = open(f'{path_out}/IBMF_T0_directed_ER_PD_Lotka_Volterra_summary_av0_{avn0}_tol_{tol}_maxiter_{max_iter}.txt', 'w')
    fout.write("# N c mu sigma av_time av_div samples_div prob_div av_m std_m\n")
    vals_list = get_all_vals(path_in, eps, avn0, tol, max_iter)
    for vals in vals_list:
        N, c, mu, sigma, av_time, av_num_div, samples_div_m, nsamples, av_m, std_av_m = vals
        fout.write(f'{N} {c} {mu:.{ndigits}f} {sigma:.{ndigits}f} {av_time:.{6}f} {av_num_div:.{6}f} {samples_div_m} {samples_div_m / nsamples:.{6}f} {av_m:.{6}f} {std_av_m:.{6}f}\n')
    fout.close()


def main():
    eps = "0.000"
    avn0 = "0.08"
    tol = "1e-4"
    max_iter = "10000"

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/AllData/directed_ER/"
    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/"

    ndigits = 3

    print_summary(path_in, path_out, eps, avn0, tol, max_iter, ndigits)
    return 0


if __name__ == '__main__':
    main()
