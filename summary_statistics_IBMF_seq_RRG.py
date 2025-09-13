__author__ = 'david'

import numpy as np
import os
import fnmatch



def filter_files(path, lda, eps, sigma, N, c, avn0, tol, max_iter, damping, nseq):
    files_T_mu = []

    # Define the pattern for matching filenames
    pattern = f'IBMF_seq_RRG_PD_Lotka_Volterra_final_av0_{avn0}_T_*_lambda_{lda}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_mu_*_sigma_{sigma}_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt'

    # Iterate through the files in the specified directory
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            parts = filename.split('_')
            files_T_mu.append((filename, float(parts[10]), float(parts[20])))
    sorted_pairs = sorted(files_T_mu, key=lambda x: (x[1], x[2]))  # Sort by mu value
    return sorted_pairs


def summary_statistics(path, filename):
    fin = open(f'{path}/{filename}', 'r')
    all_lines = fin.readlines()
    fin.close()
    if len(all_lines) > 0:
        av_time = 0.0
        av_num_div = 0.0
        samples_with_deaths = 0
        samples_div_m = 0
        samples_multiple_eq = 0
        av_m = 0.0
        av_m_sqr = 0.0
        for line in all_lines:
            line_split = line.split()
            av_time += int(line_split[0])
            av_num_div += int(line_split[4])
            if line_split[1] != "1":
                samples_div_m += 1
            if line_split[7] != "1":
                samples_multiple_eq += 1
            if int(line_split[5]) > 0:
                samples_with_deaths += 1
            av_m += float(line_split[2])
            av_m_sqr += float(line_split[2]) * float(line_split[2])
        av_time /= len(all_lines)
        av_num_div /= len(all_lines)
        av_m /= len(all_lines)
        av_m_sqr /= len(all_lines)
        std_av_m = np.sqrt(abs(av_m_sqr - av_m * av_m))
        return av_time, av_num_div, samples_div_m, samples_multiple_eq, samples_with_deaths, av_m, std_av_m, len(all_lines), True
    else:
        print(f"No data found in file {filename}. Returning zeros.")
        return 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0, False


def get_all_vals(path, lda, eps, sigma, N, c, avn0, tol, max_iter, damping, nseq):
    # Find all files that match the pattern
    sorted_data = filter_files(path, lda, eps, sigma, N, c, avn0, tol, max_iter, damping, nseq)
    vals_list = []
    for filename, T, mu in sorted_data:
        av_time, av_num_div, samples_div_m, samples_multiple_eq, samples_with_deaths, av_m, std_av_m, nsamples, found = summary_statistics(path, filename)
        if found:
            vals_list.append((T, mu, av_time, av_num_div, samples_div_m, samples_multiple_eq, samples_with_deaths, av_m, std_av_m, nsamples))
        print(f'Processed  T={T}   mu={mu}')
    return vals_list




def print_summary(path_in, path_out, lda, eps, sigma, N, c, avn0, tol, max_iter, ndigits, damping, nseq):
    fout = open(f'{path_out}/IBMF_seq_RRG_PD_Lotka_Volterra_summary_av0_{avn0}_lambda_{lda}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_sigma_{sigma}_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt', 'w')
    fout.write("# T mu av_time av_div samples_div samples_multiple_eq samples_with_deaths av_m std_m nsamples\n")
    vals_list = get_all_vals(path_in, lda, eps, sigma, N, c, avn0, tol, max_iter, damping, nseq)
    for vals in vals_list:
        T, mu, av_time, av_num_div, samples_div_m, samples_multiple_eq, samples_with_deaths, av_m, std_av_m, nsamples = vals
        fout.write(f"{T:.{ndigits}f} {mu:.{ndigits}f} {av_time:.6f} {av_num_div:.6f} {samples_div_m} {samples_multiple_eq} {samples_with_deaths} {av_m:.6f} {std_av_m:.6f} {nsamples}\n")
    fout.close()


def main():
    eps = "0.000"
    sigma = "0.000"
    avn0 = "0.08"
    lda = "1e-6"
    tol = "1e-6"
    max_iter = "10000"
    N = "2048"
    c = "3"
    damping = "1.0"
    nseq = "10"

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/AllData/PhaseDiagram/sigma0/"
    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/"

    ndigits = 3

    print_summary(path_in, path_out, lda, eps, sigma, N, c, avn0, tol, max_iter, ndigits, damping, nseq)
    return 0


if __name__ == '__main__':
    main()
