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


def filter_files(path, eps, lda, h, N, c, sigma):
    files_T_mu = []

    # Define the pattern for matching filenames
    pattern = f'Lotka-Volterra_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_h_{h}_N_{N}_c_{c}_mu_*_sigma_{sigma}_T_*_Equilibrium_Points.txt'

    # Iterate through the files in the specified directory
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            # Extract the unique strings from the filename
            parts = filename.split('_')
            files_T_mu.append((filename, float(parts[18]), float(parts[14])))
    sorted_data = sorted(files_T_mu, key=lambda x: (x[1], x[2]))  # Sort by mu value
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


def summary_statistics(path, filename, thr=10):
    fin = open(f'{path}/{filename}', 'r')
    fin.readline()  # Skip header line
    all_lines = fin.readlines()
    fin.close()
    if len(all_lines) > 0:
        av_time = 0.0
        av_div = 0.0
        samples_div = 0
        samples_deaths = 0
        av_ni = 0.0
        std_av_ni = 0.0
        av_std_ni = 0.0
        std_av_std_ni = 0.0
        for line in all_lines:
            line_split = line.split()
            av_time += float(line_split[3])
            ni_div = int(line_split[4])
            av_div += ni_div
            if ni_div > 0:
                samples_div += 1
            if float(line_split[5]) > 0:
                samples_deaths += 1
            av_ni += read_carefully(line_split[6], thr)
            std_av_ni += read_carefully(line_split[7], thr)
            av_std_ni += read_carefully(line_split[8], thr)
            std_av_std_ni += read_carefully(line_split[9], thr)
        av_time /= len(all_lines)
        av_div /= len(all_lines)
        av_ni /= len(all_lines)
        std_av_ni /= len(all_lines)
        av_std_ni /= len(all_lines)
        std_av_std_ni /= len(all_lines)
        return av_time, av_div, samples_div, samples_deaths, av_ni, std_av_ni, av_std_ni, std_av_std_ni, len(all_lines), True
    else:
        print(f"No data found in file {filename}. Returning zeros.")
        return 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, False


def get_all_vals(path, eps, lda, h, N, c, sigma):
    # Find all files that match the pattern
    sorted_data = filter_files(path, eps, lda, h, N, c, sigma)
    vals_list = []
    for filename, T, mu in sorted_data:
        av_time, av_div, samples_div, samples_deaths, av_ni, std_av_ni, av_std_ni, std_av_std_ni, nsamples, found = summary_statistics(path, filename)
        if found:
            vals_list.append((T, mu, av_time, av_div, samples_div, samples_deaths, av_ni, std_av_ni, av_std_ni, std_av_std_ni, nsamples))
        print(f'Processed T={T}   mu={mu}')
    return vals_list




def print_summary(path_in, path_out, eps, lda, h, N, c, sigma, ndigits):
    fout = open(f'{path_out}/Lotka-Volterra_summary_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_h_{h}_N_{N}_c_{c}_sigma_{sigma}.txt', 'w')
    fout.write("#mu sigma av_time av_div samples_div av_ni std_av_ni av_std_ni std_av_std_ni nsamples\n")
    vals_list = get_all_vals(path_in, eps, lda, h, N, c, sigma)
    for vals in vals_list:
        T, mu, av_time, av_div, samples_div, samples_deaths, av_ni, std_av_ni, av_std_ni, std_av_std_ni, nsamples = vals
        fout.write(f"{T:.{ndigits}f} {mu:.{ndigits}f} {av_time:.6f} {av_div:.6f} {samples_div} {samples_deaths} {av_ni:.6f} {std_av_ni:.6f} {av_std_ni:.6f} {std_av_std_ni:.6f} {nsamples}\n")
    fout.close()


def main():
    eps = "1.0"
    lda = "1e-06"
    N = "256"
    c = "3.00"
    h = "0.001"
    sigma = "0.0"

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/AllData/sigma0/"
    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"

    ndigits = 3


    print_summary(path_in, path_out, eps, lda, h, N, c, sigma, ndigits)
    return 0


if __name__ == '__main__':
    main()
