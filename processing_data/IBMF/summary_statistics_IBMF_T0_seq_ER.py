__author__ = 'david'

import numpy as np
import os
import fnmatch



def filter_files(path, T, lda, eps, avn0, dn, ninitconds, tol, max_iter, damping, nseq):
    files_mu_sigma = []

    # Define the pattern for matching filenames
    pattern = f'IBMF_seq_ER_T_{T}_lambda_{lda}_PD_Lotka_Volterra_final_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_mu_*_sigma_*_N_*_c_*_damping_{damping}_nseq_{nseq}.txt'

    # Iterate through the files in the specified directory
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            parts = filename.split('_')
            files_mu_sigma.append((filename, int(parts[28]), float(parts[30]), float(parts[24]), float(parts[26])))
    sorted_pairs = sorted(files_mu_sigma, key=lambda x: (x[1], x[2], x[3], x[4]))  # Sort by mu value
    return sorted_pairs


def summary_statistics(path, filename):
    fin = open(f'{path}/{filename}', 'r')
    all_lines = fin.readlines()
    fin.close()
    if len(all_lines) > 0:
        av_time_conv = 0.0
        av_time_conv_sqr = 0.0
        av_num_div = 0.0
        samples_div_m = 0
        av_m = 0.0
        av_m_sqr = 0.0
        runtime_conv = 0.0
        runtime_conv_sqr = 0.0
        for line in all_lines:
            line_split = line.split()
            if len(line_split) < 11:  # Skip lines that do not have enough data
                continue
            av_num_div += int(line_split[4])
            if line_split[1] != "1":
                samples_div_m += 1
            else:
                av_time_conv += int(line_split[0])
                av_time_conv_sqr += int(line_split[0]) * int(line_split[0])
                runtime_conv += float(line_split[10])
                runtime_conv_sqr += float(line_split[10]) * float(line_split[10])
            av_m += float(line_split[2])
            av_m_sqr += float(line_split[2]) * float(line_split[2])
        av_num_div /= len(all_lines)
        av_m /= len(all_lines)
        av_m_sqr /= len(all_lines)
        std_av_m = np.sqrt(abs(av_m_sqr - av_m * av_m))
        if (len(all_lines) - samples_div_m) > 0:
            av_time_conv /= (len(all_lines) - samples_div_m)
            av_time_conv_sqr /= (len(all_lines) - samples_div_m)
            std_time_conv = np.sqrt(abs(av_time_conv_sqr - av_time_conv * av_time_conv))
            runtime_conv /= (len(all_lines) - samples_div_m)
            runtime_conv_sqr /= (len(all_lines) - samples_div_m)
            std_runtime_conv = np.sqrt(abs(runtime_conv_sqr - runtime_conv * runtime_conv))
        else:
            av_time_conv = "nodata"
            std_time_conv = "nodata"
            runtime_conv = "nodata"
            std_runtime_conv = "nodata"
        return av_time_conv, std_time_conv, av_num_div, samples_div_m, av_m, std_av_m, \
               runtime_conv, std_runtime_conv, len(all_lines), True
    else:
        print(f"No data found in file {filename}. Returning zeros.")
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0, False


def get_all_vals(path, T, lda, eps, avn0, dn, ninitconds, tol, max_iter, damping, nseq):
    # Find all files that match the pattern
    sorted_data = filter_files(path, T, lda, eps, avn0, dn, ninitconds, tol, max_iter, damping, nseq)
    vals_list = []
    for filename, N, c, mu, sigma in sorted_data:
        av_time_conv, std_time_conv, av_num_div, samples_div_m, av_m, std_av_m, runtime_conv, std_runtime_conv, nsamples, found = summary_statistics(path, filename)
        if found:
            vals_list.append((N, c, mu, sigma, av_time_conv, std_time_conv, av_num_div, samples_div_m, av_m, std_av_m, runtime_conv, std_runtime_conv, nsamples))
        print(f'Processed N={N}  c={c}  mu={mu}   sigma={sigma}')
    return vals_list




def print_summary(path_in, path_out, T, lda, eps, avn0, dn, ninitconds, tol, max_iter, damping, nseq, ndigits):
    fout = open(f'{path_out}/IBMF_seq_directed_ER_summary_T_{T}_lambda_{lda}_PD_Lotka_Volterra_final_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_damping_{damping}_nseq_{nseq}.txt', 'w')
    fout.write("# N  c  mu  sigma  av_time  av_div  samples_div  prob_div  error_prob  av_m  error_av_m  runtime(s)  std_runtime(s) nsamples\n")
    vals_list = get_all_vals(path_in, T, lda, eps, avn0, dn, ninitconds, tol, max_iter, damping, nseq)
    for vals in vals_list:
        N, c, mu, sigma, av_time_conv, std_time_conv, av_num_div, samples_div_m, av_m, std_av_m, runtime_conv, std_runtime_conv, nsamples = vals
        prob = samples_div_m / nsamples if nsamples > 0 else 0.0
        error = np.sqrt(prob * (1 - prob) / nsamples) if nsamples > 0 else 0.0
        if runtime_conv != "nodata":
            av_time_conv = f'{av_time_conv:.{6}f}'
            std_time_conv = f'{std_time_conv:.{6}f}'
            runtime_conv = f'{runtime_conv:.{6}f}'
            std_runtime_conv = f'{std_runtime_conv:.{6}f}'
        fout.write(f'{N} {int(c * 10 ** ndigits)} {int(mu * 10 ** ndigits)} {int(sigma * 10 ** ndigits)} {av_time_conv} {std_time_conv} {av_num_div:.{6}f} {samples_div_m} {samples_div_m / nsamples:.{6}f} {error:.{6}f} {av_m:.{6}f} {std_av_m:.{6}f} {runtime_conv} {std_runtime_conv} {nsamples}\n')
    fout.close()


def main():
    T = "0.000"
    lda = "0.000"
    eps = "0.000"
    avn0 = "0.08"
    dn = "0"
    ninitconds = "1"
    tol = "1e-6"
    max_iter = "10000"
    damping = "0.2"
    nseq = "1"


    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/AllData/directed_ER/"
    # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/"
    path_in = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Results/IBMF/AllData/directed_ER/"
    path_out = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Results/IBMF/"

    ndigits = 3

    print_summary(path_in, path_out, T, lda, eps, avn0, dn, ninitconds, tol, max_iter, damping, nseq, ndigits)
    return 0


if __name__ == '__main__':
    main()
