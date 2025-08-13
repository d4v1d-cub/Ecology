__author__ = 'david'

import numpy as np
import os
import fnmatch



def filter_files(path, eps, lda, h, N, c, sigma, T):
    # Create a list to store the filenames
    filenames = []
    # Create a set to store unique strings for the '**' placeholders
    mu_vals = []

    # Define the pattern for matching filenames
    pattern = f'Lotka-Volterra_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_h_{h}_N_{N}_c_{c}_mu_*_sigma_{sigma}_T_{T}_Equilibrium_Points.txt'

    # Iterate through the files in the specified directory
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            filenames.append(filename)
            # Extract the unique strings from the filename
            parts = filename.split('_')
            mu_vals.append(float(parts[14]))
    sorted_pairs = sorted(zip(filenames, mu_vals), key=lambda x: x[1])  # Sort by mu value
    return sorted_pairs


def summary_statistics(path, filename):
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
            av_ni += float(line_split[5])
            std_av_ni += float(line_split[6])
            av_std_ni += float(line_split[7])
            std_av_std_ni += float(line_split[8])
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


def find_transition(path, eps, lda, h, N, c, sigma, T, ndigits_T):
    # Find all files that match the pattern
    format_str = "{0:." + str(ndigits_T) + "f}"
    str_T = format_str.format(T)
    sorted_pairs = filter_files(path, eps, lda, h, N, c, sigma, str_T)
    vals_list = []
    for filename, mu in sorted_pairs:
        av_time, av_div, samples_div, av_ni, std_av_ni, av_std_ni, std_av_std_ni, found = summary_statistics(path, filename)
        if found:
            vals_list.append((T, mu, av_time, av_div, samples_div, av_ni, std_av_ni, av_std_ni, std_av_std_ni))
    return vals_list




def print_summary(path_in, path_out, temp_list, eps, lda, h, N, c, sigma, ndigits):
    fout = open(f'{path_out}/Lotka-Volterra_summary_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_h_{h}_N_{N}_c_{c}_sigma_{sigma}.txt', 'w')
    fout.write("# T mu av_time av_div samples_div av_ni std_av_ni av_std_ni std_av_std_ni\n")
    for T in temp_list:
        vals_list = find_transition(path_in, eps, lda, h, N, c, sigma, T, ndigits)
        for vals in vals_list:
            fout.write(f"{vals[0]:.{ndigits}f} {vals[1]:.{ndigits}f} {vals[2]:.6f} {vals[3]:.6f} {vals[4]} {vals[5]:.6f} {vals[6]:.6f} {vals[7]:.6f} {vals[8]:.6f}\n")
    fout.close()


def main():
    eps = "1.0"
    lda = "1e-06"
    N = "256"
    c = "3.00"
    h = "0.001"
    sigma = "0"

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/AllData/sigma0/"
    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"

    ndigits = 3

    T0 = 0.001
    T0_langevin = 0.002
    dT_langevin = 0.002
    Tf_langevin = 0.050

    temp_list = np.arange(T0_langevin, Tf_langevin + dT_langevin / 2, dT_langevin)
    temp_list = np.insert(temp_list, 0, T0)

    print_summary(path_in, path_out, temp_list, eps, lda, h, N, c, sigma, ndigits)
    return 0


if __name__ == '__main__':
    main()
