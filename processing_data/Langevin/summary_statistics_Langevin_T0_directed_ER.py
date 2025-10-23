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


def filter_files(path, eps, lda, tol_fixed_point, T):
    files_mu_sigma = []

    # Define the pattern for matching filenames
    pattern = f'Lotka-Volterra_epsilon_{eps}_Directed_Trivial_lambda_{lda}_tol_{tol_fixed_point}_N_*_c_*_mu_*_sigma_*_T_{T}_Phase_Diagram*.txt'

    # Iterate through the files in the specified directory
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            # Extract the unique strings from the filename
            parts = filename.split('_')
            files_mu_sigma.append((filename, int(parts[10]), float(parts[12]), float(parts[14]), float(parts[16])))
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
        av_time_conv = 0.0
        av_time_conv_sqr = 0.0
        samples_div = 0
        samples_multiple_eq = 0
        runtime_conv = 0.0
        runtime_conv_sqr = 0.0
        nsamples = 0
        for line in all_lines:
            if not line.startswith('#'):
                line_split = line.split()
                if line_split[5] == "0":
                    samples_div += 1
                else:
                    av_time_conv += float(line_split[3])
                    av_time_conv_sqr += float(line_split[3]) * float(line_split[3])
                    runtime_conv += float(line_split[11]) / 1000
                    runtime_conv_sqr += (float(line_split[11]) / 1000) * (float(line_split[11]) / 1000)
                if line_split[9] == "1" or line_split[7] == "1":
                    samples_multiple_eq += 1
                nsamples += 1
        if (nsamples - samples_div) > 0:
            av_time_conv /= (nsamples - samples_div)
            av_time_conv_sqr /= (nsamples - samples_div)
            std_time_conv = np.sqrt(abs(av_time_conv_sqr - av_time_conv * av_time_conv))
            runtime_conv /= (nsamples - samples_div)
            runtime_conv_sqr /= (nsamples - samples_div)
            std_runtime_conv = np.sqrt(abs(runtime_conv_sqr - runtime_conv * runtime_conv))
        else:
            av_time_conv = "nodata"
            std_time_conv = "nodata"
            runtime_conv = "nodata"
            std_runtime_conv = "nodata"
        return av_time_conv, std_time_conv, samples_div, samples_multiple_eq, runtime_conv, std_runtime_conv, nsamples, True
    else:
        print(f"No data found in file {filename}. Returning zeros.")
        return 0.0, 0.0, 0, 0, 0.0, 0.0, 0, False


def get_all_vals(path, eps, lda, tol_fixed_point, T):
    # Find all files that match the pattern
    sorted_data = filter_files(path, eps, lda, tol_fixed_point, T)
    vals_list = []
    for filename, N, c, mu, sigma in sorted_data:
        av_time_conv, std_time_conv, samples_div, samples_multiple_eq, runtime_conv, std_runtime_conv, nsamples, found = summary_statistics(path, filename)
        if found:
            vals_list.append((N, c, mu, sigma, av_time_conv, std_time_conv, samples_div, samples_multiple_eq, runtime_conv, std_runtime_conv, nsamples))
        print(f'Processed N={N}  c={c}   mu={mu}   sigma={sigma}')
    return vals_list




def print_summary(path_in, path_out, eps, lda, tol_fixed_point, T, ndigits):
    fout = open(f'{path_out}/Lotka-Volterra_summary_epsilon_{eps}_Directed_Trivial_lambda_{lda}_tol_{tol_fixed_point}_T_{T}.txt', 'w')
    fout.write("#N  c  mu  sigma  av_time_conv  std_time_conv  samples_div  samples_mult_eq  prob_div  error_prob  runtime(s)  std_runtime(s)  nsamples\n")
    vals_list = get_all_vals(path_in, eps, lda, tol_fixed_point, T)
    for vals in vals_list:
        N, c, mu, sigma, av_time_conv, std_time_conv, samples_div, samples_multiple_eq, runtime_conv, std_runtime_conv, nsamples = vals
        prob = samples_div / nsamples if nsamples > 0 else 0.0
        error = np.sqrt(prob * (1 - prob) / nsamples) if nsamples > 0 else 0.0
        if runtime_conv != "nodata":
            av_time_conv = f'{av_time_conv:.{6}f}'
            std_time_conv = f'{std_time_conv:.{6}f}'
            runtime_conv = f'{runtime_conv:.{6}f}'
            std_runtime_conv = f'{std_runtime_conv:.{6}f}'
        fout.write(f"{N} {int(c * 10 ** ndigits)} {int(mu * 10 ** ndigits)} {int(sigma * 10 ** ndigits)} {av_time_conv} {std_time_conv} {samples_div} {samples_multiple_eq} {prob:.6f} {error:.6f} {runtime_conv} {std_runtime_conv} {nsamples}\n")
    fout.close()


def main():
    eps = "0.000"
    lda = "1e-06"
    T = "0.000"
    tol_fixed_point = "1e-08"

    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/AllData/directed_ER/"
    # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"
    path_in = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Langevin/Results/AllData/directed_ER/"
    path_out = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Langevin/Results/"

    ndigits = 3

    print_summary(path_in, path_out, eps, lda, tol_fixed_point, T, ndigits)

    return 0


if __name__ == '__main__':
    main()
