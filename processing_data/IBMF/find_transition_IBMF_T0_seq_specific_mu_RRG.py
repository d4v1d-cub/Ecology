__author__ = 'david'

import os
import fnmatch
import numpy as np
from scipy.optimize import curve_fit


def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


def fit_sigmoid(sigma_data, fraction_data, error_in_fraction, maxfev=10000):
    try:
        popt, pcov = curve_fit(sigmoid, sigma_data, fraction_data, sigma=error_in_fraction, 
                               maxfev=maxfev, bounds=([0, 0], [np.inf, np.inf]))
        error_parameters = np.sqrt(np.diag(pcov))
        return popt, error_parameters
    except RuntimeError:
        print("Error: Sigmoid fit did not converge.")
        return None



def find_transition_fit(lines, position):
    sigma_data = {}
    fraction_data = {}
    error_in_fraction_list = {}
    for line in lines:
        line_split = line.split()
        par_key = float(line_split[0])
        par_trans = float(line_split[1])

        num = int(line_split[position])
        nsamples = int(line_split[-1])
        fraction = num / nsamples
        error_in_fraction = np.sqrt(fraction * (1 - fraction) / nsamples)
        if par_key not in sigma_data:
            sigma_data[par_key] = []
            fraction_data[par_key] = []
            error_in_fraction_list[par_key] = []
        if error_in_fraction > 0:
            sigma_data[par_key].append(par_trans)
            fraction_data[par_key].append(fraction)
            error_in_fraction_list[par_key].append(error_in_fraction)
    transitions = {}
    for par_key in sigma_data:
        popt, error_parameters = fit_sigmoid(sigma_data[par_key], fraction_data[par_key], error_in_fraction_list[par_key])
        _, b = popt
        _ , error_b = error_parameters
        if popt is not None and b > 0 and error_b < np.inf:
            transitions[par_key] = (b - error_b, b + error_b)
        else:
            print(f"Could not fit sigmoid for par_key {par_key}.")
    return transitions




def find_transition(lines, position):
    index = 0
    transitions = {}
    transition_found = {}
    while index < len(lines) - 1:
        line_split = lines[index].split()
        par_key = float(line_split[0])
        par_trans = float(line_split[1])

        num = int(line_split[position])
        nsamples = int(line_split[-1])
        
        line_split_below = lines[index + 1].split()
        key_below = float(line_split_below[0])
        if key_below == par_key:
            if not par_key in transition_found:
                nsamples_below = int(line_split_below[-1])
                num_below = int(line_split_below[position])
                if num_below >= nsamples_below / 2 and num < nsamples / 2:
                    transition_found[par_key] = True
                    par_trans_below = float(line_split_below[1])
                    transitions[par_key] = (par_trans, par_trans_below)
        index += 1
    return transitions


def find_identify_transition(lines, position_1, position_2):
    index = 0
    transitions = {}
    transition_found = {}
    trans_type = {}
    while index < len(lines) - 1:
        line_split = lines[index].split()
        par_key = float(line_split[0])
        par_trans = float(line_split[1])

        num_1 = int(line_split[position_1])
        nsamples = int(line_split[-1])

        
        line_split_below = lines[index + 1].split()
        key_below = float(line_split_below[0])
        if key_below == par_key:
            if not par_key in transition_found:
                nsamples_below = int(line_split_below[-1])
                num_1_below = int(line_split_below[position_1])
                num_2_below = int(line_split_below[position_2])
                if num_1_below >= nsamples_below / 2 and num_1 < nsamples / 2:
                    transition_found[par_key] = True
                    par_trans_below = float(line_split_below[1])
                    transitions[par_key] = (par_trans, par_trans_below)
                    if num_1_below - num_2_below > num_2_below:
                        trans_type[par_key] = 1
                    else:
                        trans_type[par_key] = 2
        index += 1
    return transitions, trans_type


def filter_files(path, pattern, pos_par):
    files_list = []

    
    # Iterate through the files in the specified directory
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            parts = filename.split('_')
            files_list.append((filename, float(parts[pos_par])))
    sorted_pairs = sorted(files_list, key=lambda x: x[1])  # Sort by mu value
    return sorted_pairs


def read_file_specific_mu(path, filename, mu, pos_mu):
    fin = open(f'{path}/{filename}', 'r')
    all_lines = fin.readlines()
    fin.close()
    lines_mu = []
    for line in all_lines:
        line_split = line.split()
        try:
            mu_value = line_split[pos_mu]
            if mu_value == mu:
                lines_mu.append(line)
        except (IndexError, ValueError):
            print(f"Skipping line due to error: {line.strip()}")
    return lines_mu


def find_all_trans(path, T, lda, eps, mu, c, avn0, dn, ninitconds, tol, max_iter, damping, nseq, ndigits, 
                   pos_mu=0, pos_N=24):
    fout_div = open(f'{path}/IBMF_seq_RRG_T_{T}_lambda_{lda}_PD_Lotka_Volterra_transitions_div_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_mu_{mu}_c_{c}_damping_{damping}_nseq_{nseq}.txt', 'w')
    fout_mult = open(f'{path}/IBMF_seq_RRG_T_{T}_lambda_{lda}_PD_Lotka_Volterra_transitions_mult_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_mu_{mu}_c_{c}_damping_{damping}_nseq_{nseq}.txt', 'w')
    
    pattern = f'IBMF_seq_RRG_T_{T}_lambda_{lda}_PD_Lotka_Volterra_summary_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_*_c_{c}_damping_{damping}_nseq_{nseq}.txt'
    files_sorted = filter_files(path, pattern, pos_N)

    for filename, N in files_sorted:
        lines_mu = read_file_specific_mu(path, filename, mu, pos_mu)
        transitions_m = find_transition(lines_mu, 4)
        transitions_multiple_eq, trans_type = find_identify_transition(lines_mu, 5, 4)
        for mu_key in transitions_m:
           sigma_m, sigma_below_m = transitions_m[mu_key]
           fout_div.write(f"{int(N)}\t{mu_key:.{ndigits}f}\t{sigma_m:.{ndigits}f}\t{sigma_below_m:.{ndigits}f}\n") 
        for mu_key in transitions_multiple_eq:
            sigma_multiple_eq, sigma_below_multiple_eq = transitions_multiple_eq[mu_key]
            kind = trans_type[mu_key]
            fout_mult.write(f"{int(N)}\t{mu_key:.{ndigits}f}\t{sigma_multiple_eq:.{ndigits}f}\t{sigma_below_multiple_eq:.{ndigits}f}\t{kind}\n")
    
    fout_div.close()
    fout_mult.close()


def find_all_trans_fit(path, T, lda, eps, mu, c, avn0, dn, ninitconds, tol, max_iter, damping, nseq, ndigits, 
                   pos_mu=0, pos_N=24, ndigits_sigma=6):
    fout_div = open(f'{path}/IBMF_seq_RRG_T_{T}_lambda_{lda}_PD_Lotka_Volterra_transitions_div_fit_sigmoid_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_mu_{mu}_c_{c}_damping_{damping}_nseq_{nseq}.txt', 'w')
    fout_mult = open(f'{path}/IBMF_seq_RRG_T_{T}_lambda_{lda}_PD_Lotka_Volterra_transitions_mult_fit_sigmoid_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_mu_{mu}_c_{c}_damping_{damping}_nseq_{nseq}.txt', 'w')
    
    pattern = f'IBMF_seq_RRG_T_{T}_lambda_{lda}_PD_Lotka_Volterra_summary_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_*_c_{c}_damping_{damping}_nseq_{nseq}.txt'
    files_sorted = filter_files(path, pattern, pos_N)

    for filename, N in files_sorted:
        lines_mu = read_file_specific_mu(path, filename, mu, pos_mu)
        transitions_m = find_transition_fit(lines_mu, 4)
        transitions_multiple_eq = find_transition_fit(lines_mu, 5)
        for mu_key in transitions_m:
           sigma_m, sigma_below_m = transitions_m[mu_key]
           fout_div.write(f"{int(N)}\t{mu_key:.{ndigits}f}\t{sigma_m:.{ndigits_sigma}f}\t{sigma_below_m:.{ndigits_sigma}f}\n") 
        for mu_key in transitions_multiple_eq:
            sigma_multiple_eq, sigma_below_multiple_eq = transitions_multiple_eq[mu_key]
            fout_mult.write(f"{int(N)}\t{mu_key:.{ndigits}f}\t{sigma_multiple_eq:.{ndigits_sigma}f}\t{sigma_below_multiple_eq:.{ndigits_sigma}f}\n")
    
    fout_div.close()
    fout_mult.close()


def main():
    T = "0.000"
    lda = "0.000"
    eps = "0.000"
    avn0 = "0.5"
    dn = "0.5"
    ninitconds = "10"
    tol = "1e-6"
    max_iter = "10000"
    c = "3"
    damping = "0.2"
    nseq = "1"

    path = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Results/IBMF/"
    # path = "/mnt/d/Research/Ecology/Results/IBMF/"


    ndigits = 3
    mu = "0.270"

    # find_all_trans(path, T, lda, eps, mu, c, avn0, dn, ninitconds, tol, max_iter, damping, nseq, ndigits)
    find_all_trans_fit(path, T, lda, eps, mu, c, avn0, dn, ninitconds, tol, max_iter, damping, nseq, ndigits)

    return 0


if __name__ == '__main__':
    main()
