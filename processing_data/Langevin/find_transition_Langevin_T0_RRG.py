__author__ = 'david'

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
        if len(sigma_data[par_key]) > 0:
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
    while index < len(lines) - 1:
        line_split = lines[index].split()
        par_key = float(line_split[0])
        par_trans = float(line_split[1])

        num = int(line_split[position])
        nsamples = int(line_split[-1])

        
        if not par_key in transitions:
            line_split_below = lines[index + 1].split()
            key_below = float(line_split_below[0])
            if key_below == par_key:
                num_samples_below = int(line_split_below[-1])
                num_below = int(line_split_below[position])
                if num_below >= num_samples_below / 2 and num < nsamples / 2:
                    par_trans_below = float(line_split_below[1])
                    transitions[par_key] = (par_trans, par_trans_below)
        index += 1
    return transitions


def find_identify_transition(lines, position_1, position_2):
    index = 0
    transitions = {}
    trans_type = {}
    while index < len(lines) - 1:
        line_split = lines[index].split()
        par_key = float(line_split[0])
        par_trans = float(line_split[1])

        num_1 = int(line_split[position_1])
        nsamples = int(line_split[-1])

        
        if not par_key in transitions: 
            line_split_below = lines[index + 1].split()
            key_below = float(line_split_below[0])
            if key_below == par_key:
                num_samples_below = int(line_split_below[-1])
                num_1_below = int(line_split_below[position_1])
                num_2_below = int(line_split_below[position_2])
                if num_1_below >= num_samples_below / 2 and num_1 < nsamples / 2:
                    par_trans_below = float(line_split_below[1])
                    transitions[par_key] = (par_trans, par_trans_below)
                    if num_1_below - num_2_below > num_2_below:
                        trans_type[par_key] = 1
                    else:
                        trans_type[par_key] = 2
        index += 1
    return transitions, trans_type


def find_all_trans(path, eps, lda, tol_fixed_point, N, c, T, ndigits):
    fout_div = open(f'{path}/Lotka-Volterra_transition_div_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'w')
    fout_mult = open(f'{path}/Lotka-Volterra_transition_mult_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'w')
    fout_deaths = open(f'{path}/Lotka-Volterra_transition_deaths_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'w')
   
    fin = open(f'{path}/Lotka-Volterra_summary_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'r')
    fin.readline()  # Skip header line

    lines = fin.readlines()

    fin.close()

    transitions_div = find_transition(lines, 3)
    transitions_multiple_eq, trans_type = find_identify_transition(lines, 4, 3)
    transitions_deaths = find_transition(lines, 5)
    
    
    for mu in transitions_div:
        sigma_div, sigma_below_div = transitions_div[mu]
        fout_div.write(f"{mu:.{ndigits}f}\t{sigma_div:.{ndigits}f}\t{sigma_below_div:.{ndigits}f}\n")
     
    fout_div.close()


    for mu in transitions_multiple_eq:
        sigma, sigma_below = transitions_multiple_eq[mu]
        kind = trans_type[mu]
        fout_mult.write(f"{mu:.{ndigits}f}\t{sigma:.{ndigits}f}\t{sigma_below:.{ndigits}f}\t{kind}\n")
     
    fout_mult.close()


    for mu in transitions_deaths:
        sigma_deaths, sigma_below_deaths = transitions_deaths[mu]
        fout_deaths.write(f"{mu:.{ndigits}f}\t{sigma_deaths:.{ndigits}f}\t{sigma_below_deaths:.{ndigits}f}\n")
    
    fout_deaths.close()


def find_all_trans_fit(path, eps, lda, tol_fixed_point, N, c, T, ndigits, ndigits_sigma=6):
    fout_div = open(f'{path}/Lotka-Volterra_transition_div_fit_sigmoid_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'w')
    fout_mult = open(f'{path}/Lotka-Volterra_transition_mult_fit_sigmoid_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'w')
    fout_deaths = open(f'{path}/Lotka-Volterra_transition_deaths_fit_sigmoid_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'w')
   
    fin = open(f'{path}/Lotka-Volterra_summary_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'r')
    fin.readline()  # Skip header line

    lines = fin.readlines()

    fin.close()

    transitions_div = find_transition_fit(lines, 3)
    transitions_multiple_eq = find_transition_fit(lines, 4)
    transitions_deaths = find_transition_fit(lines, 5)
    
    
    for mu in transitions_div:
        sigma_div, sigma_below_div = transitions_div[mu]
        fout_div.write(f"{mu:.{ndigits}f}\t{sigma_div:.{ndigits_sigma}f}\t{sigma_below_div:.{ndigits_sigma}f}\n")
     
    fout_div.close()


    for mu in transitions_multiple_eq:
        sigma, sigma_below = transitions_multiple_eq[mu]
        fout_mult.write(f"{mu:.{ndigits}f}\t{sigma:.{ndigits_sigma}f}\t{sigma_below:.{ndigits_sigma}f}\n")
     
    fout_mult.close()


    for mu in transitions_deaths:
        sigma_deaths, sigma_below_deaths = transitions_deaths[mu]
        fout_deaths.write(f"{mu:.{ndigits}f}\t{sigma_deaths:.{ndigits_sigma}f}\t{sigma_below_deaths:.{ndigits_sigma}f}\n")
    
    fout_deaths.close()


def main():
    eps = "0.0"
    lda = "1e-06"
    N_list = ["128", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768"]
    # N_list = ["8192", "16384", "32768"]
    c = "3.00"
    T = "0.0"
    tol_fixed_point = "1e-08"

    # path = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"
    # path = "/mnt/d/Research/Ecology/Langevin/Results"
    path = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Langevin/Results/"

    ndigits = 3

    for N in N_list:
        # find_all_trans(path, eps, lda, tol_fixed_point, N, c, T, ndigits)
        find_all_trans_fit(path, eps, lda, tol_fixed_point, N, c, T, ndigits)

    return 0


if __name__ == '__main__':
    main()
