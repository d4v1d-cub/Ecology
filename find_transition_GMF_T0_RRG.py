__author__ = 'david'

import numpy as np




def find_all_trans(path, eps, N, c, avn0, tol, max_iter, ndigits, nsamples):
    fout = open(f'{path}/GMF_T0_RRG_PD_Lotka_Volterra_transitions_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}.txt', 'w')
    
    fin = open(f'{path}/GMF_T0_RRG_PD_Lotka_Volterra_summary_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}.txt', 'r')
    fin.readline()  # Skip header line

    lines = fin.readlines()

    index = 0
    transitions_m = {}
    transitions_chi = {}
    transition_found_m = {}
    transition_found_chi = {}
    while index < len(lines) - 1:
        line_split = lines[index].split()
        mu = float(line_split[0])
        sigma = float(line_split[1])

        num_div_m = int(line_split[5])
        num_div_chi = int(line_split[6])

        line_split_below = lines[index + 1].split()
        mu_below = float(line_split_below[0])
        if mu_below == mu:
            if not mu in transition_found_m:
                num_div_below_m = int(line_split_below[5])
                if num_div_below_m >= nsamples / 2 and num_div_m < nsamples / 2:
                    transition_found_m[mu] = True
                    sigma_below = float(line_split_below[1])
                    transitions_m[mu] = (sigma, sigma_below)
                    
            if not mu in transition_found_chi:
                num_div_below_chi = int(line_split_below[6])
                if num_div_below_chi >= nsamples / 2 and num_div_chi < nsamples / 2:
                    transition_found_chi[mu] = True
                    sigma_below = float(line_split_below[1])
                    transitions_chi[mu] = (sigma, sigma_below)
        index += 1
    
    for mu in transitions_m:
        sigma_m, sigma_below_m = transitions_m[mu]
        sigma_chi, sigma_below_chi = transitions_chi[mu]
        fout.write(f"{mu:.{ndigits}f}\t{sigma_m:.{ndigits}f}\t{sigma_below_m:.{ndigits}f}\t{sigma_chi:.{ndigits}f}\t{sigma_below_chi:.{ndigits}f}\n")
    
    fin.close()
    fout.close()



def main():
    eps = "0.000"
    avn0 = "0.08"
    tol = "1e-4"
    max_iter = "10000"
    N = "1024"
    c = "3"

    path = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/"

    ndigits = 3
    nsamples = 10000

    find_all_trans(path, eps, N, c, avn0, tol, max_iter, ndigits, nsamples)
    find_all_trans(path, eps, N, c, avn0, tol, max_iter, ndigits, nsamples)

    return 0


if __name__ == '__main__':
    main()
