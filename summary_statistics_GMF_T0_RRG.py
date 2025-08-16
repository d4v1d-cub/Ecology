__author__ = 'david'

import numpy as np
import os
import fnmatch



def filter_files(path, eps, N, c, avn0, tol, max_iter):
    files_mu_sigma = []

    # Define the pattern for matching filenames
    pattern = f'GMF_T0_RRG_PD_Lotka_Volterra_final_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_mu_*_sigma_*_N_{N}_c_{c}.txt'

    # Iterate through the files in the specified directory
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            parts = filename.split('_')
            files_mu_sigma.append((filename, float(parts[16]), float(parts[18])))
    sorted_pairs = sorted(files_mu_sigma, key=lambda x: (x[1], x[2]))  # Sort by mu value
    return sorted_pairs


def summary_statistics(path, filename):
    fin = open(f'{path}/{filename}', 'r')
    all_lines = fin.readlines()
    fin.close()
    if len(all_lines) > 0:
        av_time = 0.0
        av_num_div_m = 0.0
        av_num_div_chi = 0.0
        samples_div_m = 0
        samples_div_chi = 0
        av_m = 0.0
        av_m_sqr = 0.0
        av_chi = 0.0
        av_chi_sqr = 0.0
        for line in all_lines:
            line_split = line.split()
            av_time += int(line_split[0])
            n_div_m = int(line_split[10])
            n_neg_chi = int(line_split[11])
            n_div_chi = int(line_split[12])
            av_num_div_m += n_div_m
            av_num_div_chi += max(n_neg_chi, n_div_chi)
            if n_neg_chi > 0 or n_div_chi > 0:
                samples_div_chi += 1
            if n_div_m > 0:
                samples_div_m += 1
            av_m += float(line_split[6])
            av_m_sqr += float(line_split[6]) * float(line_split[6])
            av_chi += float(line_split[8])
            av_chi_sqr += float(line_split[8]) * float(line_split[8])
        av_time /= len(all_lines)
        av_num_div_m /= len(all_lines)
        av_num_div_chi /= len(all_lines)
        av_m /= len(all_lines)
        av_m_sqr /= len(all_lines)
        av_chi /= len(all_lines)
        av_chi_sqr /= len(all_lines)
        std_av_m = np.sqrt(av_m_sqr - av_m * av_m)
        std_av_chi = np.sqrt(av_chi_sqr - av_chi * av_chi)
        return av_time, av_num_div_m, av_num_div_chi, samples_div_m, samples_div_chi, av_m, std_av_m, av_chi, std_av_chi, True
    else:
        print(f"No data found in file {filename}. Returning zeros.")
        return 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, False


def get_all_vals(path, eps, N, c, avn0, tol, max_iter):
    # Find all files that match the pattern
    sorted_data = filter_files(path, eps, N, c, avn0, tol, max_iter)
    vals_list = []
    for filename, mu, sigma in sorted_data:
        av_time, av_num_div_m, av_num_div_chi, samples_div_m, samples_div_chi, av_m, std_av_m, av_chi, std_av_chi, found = summary_statistics(path, filename)
        if found:
            vals_list.append((mu, sigma, av_time, av_num_div_m, av_num_div_chi, samples_div_m, samples_div_chi, av_m, std_av_m, av_chi, std_av_chi))
        print(f'Processed  mu={mu}   sigma={sigma}')
    return vals_list




def print_summary(path_in, path_out, eps, N, c, avn0, tol, max_iter, ndigits):
    fout = open(f'{path_out}/GMF_T0_RRG_PD_Lotka_Volterra_summary_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}.txt', 'w')
    fout.write("# mu sigma av_time av_div_m av_div_chi samples_div_m samples_div_chi av_m std_m av_chi std_chi\n")
    vals_list = get_all_vals(path_in, eps, N, c, avn0, tol, max_iter)
    for vals in vals_list:
        fout.write(f"{vals[0]:.{ndigits}f} {vals[1]:.{ndigits}f} {vals[2]:.6f} {vals[3]:.6f} {vals[4]:.6f} {vals[5]} {vals[6]} {vals[7]:.6f} {vals[8]:.6f} {vals[9]:.6f} {vals[10]:.6f}\n")
    fout.close()


def main():
    eps = "0.000"
    avn0 = "0.08"
    tol = "1e-4"
    max_iter = "10000"
    N = "1024"
    c = "3"

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/AllData/PhaseDiagram/T0/"
    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/"

    ndigits = 3

    print_summary(path_in, path_out, eps, N, c, avn0, tol, max_iter, ndigits)
    return 0


if __name__ == '__main__':
    main()
