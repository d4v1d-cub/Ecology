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


def filter_files(path, eps, lda, h, N, c, T):
    files_mu_sigma = []

    # Define the pattern for matching filenames
    pattern = f'Lotka-Volterra_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_h_{h}_N_{N}_c_{c}_mu_*_sigma_*_T_{T}_Equilibrium_Points.txt'

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


def summary_statistics(path, filename, thr=10):
    fin = open(f'{path}/{filename}', 'r')
    fin.readline()  # Skip header line
    all_lines = fin.readlines()
    fin.close()
    if len(all_lines) > 0:
        av_time = 0.0
        av_div = 0.0
        samples_div = 0
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
            av_ni += read_carefully(line_split[5], thr)
            std_av_ni += read_carefully(line_split[6], thr)
            av_std_ni += read_carefully(line_split[7], thr)
            std_av_std_ni += read_carefully(line_split[8], thr)
        av_time /= len(all_lines)
        av_div /= len(all_lines)
        av_ni /= len(all_lines)
        std_av_ni /= len(all_lines)
        av_std_ni /= len(all_lines)
        std_av_std_ni /= len(all_lines)
        return av_time, av_div, samples_div, av_ni, std_av_ni, av_std_ni, std_av_std_ni, True
    else:
        print(f"No data found in file {filename}. Returning zeros.")
        return 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, False


def get_all_vals(path, eps, lda, h, N, c, T):
    # Find all files that match the pattern
    sorted_data = filter_files(path, eps, lda, h, N, c, T)
    vals_list = []
    for filename, mu, sigma in sorted_data:
        av_time, av_div, samples_div, av_ni, std_av_ni, av_std_ni, std_av_std_ni, found = summary_statistics(path, filename)
        if found:
            vals_list.append((mu, sigma, av_time, av_div, samples_div, av_ni, std_av_ni, av_std_ni, std_av_std_ni))
    return vals_list




def print_summary(path_in, path_out, eps, lda, h, N, c, T, ndigits):
    fout = open(f'{path_out}/Lotka-Volterra_summary_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_h_{h}_N_{N}_c_{c}_T_{T}.txt', 'w')
    fout.write("#mu sigma av_time av_div samples_div av_ni std_av_ni av_std_ni std_av_std_ni\n")
    vals_list = get_all_vals(path_in, eps, lda, h, N, c, T)
    for vals in vals_list:
        mu, sigma, av_time, av_div, samples_div, av_ni, std_av_ni, av_std_ni, std_av_std_ni = vals
        fout.write(f"{mu:.{ndigits}f} {sigma:.{ndigits}f} {av_time:.6f} {av_div:.6f} {samples_div} {av_ni:.6f} {std_av_ni:.6f} {av_std_ni:.6f} {std_av_std_ni:.6f}\n")
    fout.close()


def main():
    eps = "0.000"
    lda = "1e-06"
    N = "256"
    c = "3.00"
    h = "0.001"
    T = "0.000"

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/AllData/T0/"
    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"

    ndigits = 3


    print_summary(path_in, path_out, eps, lda, h, N, c, T, ndigits)
    return 0


if __name__ == '__main__':
    main()
