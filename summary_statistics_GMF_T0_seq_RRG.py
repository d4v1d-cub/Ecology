__author__ = 'david'

import numpy as np
import os
import fnmatch



def filter_files(path, eps, N, c, avn0, tol, max_iter, damping, nseq):
    files_mu_sigma = []

    # Define the pattern for matching filenames
    pattern = f'GMF_T0_seq_RRG_PD_Lotka_Volterra_final_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_mu_*_sigma_*_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt'

    # Iterate through the files in the specified directory
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            parts = filename.split('_')
            files_mu_sigma.append((filename, float(parts[17]), float(parts[19])))
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
        samples_multiple_eq = 0
        samples_with_deaths = 0
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
            if line_split[15] != "1":
                samples_multiple_eq += 1
            if int(line_split[13]) > 0:
                samples_with_deaths += 1 
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
        std_av_m = np.sqrt(abs(av_m_sqr - av_m * av_m))
        std_av_chi = np.sqrt(abs(av_chi_sqr - av_chi * av_chi))
        return av_time, av_num_div_m, av_num_div_chi, samples_div_m, samples_div_chi, samples_multiple_eq, samples_with_deaths, av_m, std_av_m, av_chi, std_av_chi, len(all_lines), True
    else:
        print(f"No data found in file {filename}. Returning zeros.")
        return 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, False


def get_all_vals(path, eps, N, c, avn0, tol, max_iter, damping, nseq):
    # Find all files that match the pattern
    sorted_data = filter_files(path, eps, N, c, avn0, tol, max_iter, damping, nseq)
    vals_list = []
    for filename, mu, sigma in sorted_data:
        av_time, av_num_div_m, av_num_div_chi, samples_div_m, samples_div_chi, samples_multiple_eq, samples_with_deaths, av_m, std_av_m, av_chi, std_av_chi, nsamples, found = summary_statistics(path, filename)
        if found:
            vals_list.append((mu, sigma, av_time, av_num_div_m, av_num_div_chi, samples_div_m, samples_div_chi, samples_multiple_eq, samples_with_deaths, av_m, std_av_m, av_chi, std_av_chi, nsamples))
        print(f'Processed  mu={mu}   sigma={sigma}')
    return vals_list




def print_summary(path_in, path_out, eps, N, c, avn0, tol, max_iter, ndigits, damping, nseq):
    fout = open(f'{path_out}/GMF_T0_seq_RRG_PD_Lotka_Volterra_summary_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt', 'w')
    fout.write("# mu sigma av_time av_div_m av_div_chi samples_div_m samples_div_chi samples_multiple_eq samples_with_deaths av_m std_m av_chi std_chi nsamples\n")
    vals_list = get_all_vals(path_in, eps, N, c, avn0, tol, max_iter, damping, nseq)
    for vals in vals_list:
        mu, sigma, av_time, av_num_div_m, av_num_div_chi, samples_div_m, samples_div_chi, samples_multiple_eq, samples_with_deaths, av_m, std_av_m, av_chi, std_av_chi, nsamples = vals
        fout.write(f"{mu:.{ndigits}f} {sigma:.{ndigits}f} {av_time:.6f} {av_num_div_m:.6f} {av_num_div_chi:.6f} {samples_div_m} {samples_div_chi} {samples_multiple_eq} {samples_with_deaths} {av_m:.6f} {std_av_m:.6f} {av_chi:.6f} {std_av_chi:.6f} {nsamples}\n")
    fout.close()


def main():
    eps = "0.000"
    avn0 = "0.08"
    tol = "1e-6"
    max_iter = "10000"
    N = "1024"
    c = "3"
    damping = "0.2"
    nseq = "10"

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/AllData/PhaseDiagram/T0/"
    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/"

    ndigits = 3

    print_summary(path_in, path_out, eps, N, c, avn0, tol, max_iter, ndigits, damping, nseq)
    return 0


if __name__ == '__main__':
    main()
