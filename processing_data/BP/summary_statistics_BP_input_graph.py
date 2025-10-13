__author__ = 'david'

import numpy as np
import os
import fnmatch



def filter_files(path, N, damping):
    files_list = []

    # Define the pattern for matching filenames
    pattern = f'BP_print_distr_from_imput_RRG_Lotka_Volterra_final_T_*_mu_*_N_{N}_damping_{damping}.txt'

    # Iterate through the files in the specified directory
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            parts = filename.split('_')
            files_list.append((filename, float(parts[10]), float(parts[12])))
    sorted_pairs = sorted(files_list, key=lambda x: (x[1], x[2]))  # Sort by mu value
    return sorted_pairs


def read_file(path, filename):
    fin = open(f'{path}/{filename}', 'r')
    line = fin.readline()
    fin.close()
    line_split = line.split()
    niter = int(line_split[1])
    conv = int(line_split[2])
    av_mess = float(line_split[3])
    var_mess = float(line_split[4])
    true_av = float(line_split[5])
    true_var = float(line_split[6])
    gauss_av = float(line_split[7])
    gauss_var = float(line_split[8])
    gauss_skewness = float(line_split[9])
    
    return niter, conv, av_mess, var_mess, true_av, true_var, gauss_av, gauss_var, gauss_skewness


def gather(path, N, damping):
    # Find all files that match the pattern
    sorted_data = filter_files(path, N, damping)
    vals_list = []
    for filename, T, mu in sorted_data:
        niter, conv, av_mess, var_mess, true_av, true_var, gauss_av, gauss_var, gauss_skewness = read_file(path, filename)
        if niter < 10001:
            vals_list.append((T, mu, niter, conv, av_mess, var_mess, true_av, true_var, gauss_av, gauss_var, gauss_skewness))
            print(f'Processed  N={N}    T={T}    mu={mu}')
    return vals_list




def print_summary(path_in, path_out, N, damping, ndigits):
    fout = open(f'{path_out}/BP_from_input_RRG_PD_Lotka_Volterra_summary_N_{N}_damping_{damping}.txt', 'w')
    fout.write("# T mu conv av_mess var_mess true_av true_var gauss_av gauss_var gauss_skewness\n")
    vals_list = gather(path_in, N, damping)
    for vals in vals_list:
        T, mu, niter, conv, av_mess, var_mess, true_av, true_var, gauss_av, gauss_var, gauss_skewness = vals
        fout.write(f"{int(T * 10 ** ndigits)} {int(mu * 10 ** ndigits)} {niter} {conv} {av_mess:.6f} {var_mess:.6f} {true_av:.6f} {true_var:.6f} {gauss_av:.6f} {gauss_var:.6f} {gauss_skewness:.6f}\n")
    fout.close()


def main():
    N_list = ["1024"]
    damping = "1.0"

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/BP/Results/AllData/from_input/"
    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/BP/Results/"

    ndigits = 3

    for N in N_list:
        print_summary(path_in, path_out, N, damping, ndigits)
    return 0


if __name__ == '__main__':
    main()
