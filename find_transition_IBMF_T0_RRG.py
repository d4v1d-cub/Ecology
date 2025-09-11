__author__ = 'david'

import numpy as np




def find_all_trans(path, eps, N, c, avn0, tol, max_iter, ndigits):
    fout = open(f'{path}/IBMF_T0_RRG_PD_Lotka_Volterra_transitions_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}.txt', 'w')
    
    fin = open(f'{path}/IBMF_T0_RRG_PD_Lotka_Volterra_summary_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}.txt', 'r')
    fin.readline()  # Skip header line

    lines = fin.readlines()

    index = 0
    transition_found = {}
    while index < len(lines) - 1:
        line_split = lines[index].split()
        mu = float(line_split[0])
        sigma = float(line_split[1])

        num_div = int(line_split[4])
        nsamples = int(line_split[-1])

        line_split_below = lines[index + 1].split()
        mu_below = float(line_split_below[0])
        if mu_below == mu and not mu in transition_found:
            num_samples_below = int(line_split_below[-1])
            num_div_below = int(line_split_below[4])
            if num_div_below >= num_samples_below / 2 and num_div < nsamples / 2:
                transition_found[mu] = True
                sigma_below = float(line_split_below[1])
                fout.write(f"{mu:.{ndigits}f}\t{sigma:.{ndigits}f}\t{sigma_below:.{ndigits}f}\n")
        index += 1
    
    fin.close()
    fout.close()



def main():
    eps = "0.000"
    avn0 = "0.08"
    tol = "1e-4"
    max_iter = "10000"
    N = "1024"
    c = "3"

    path = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/"

    ndigits = 3

    find_all_trans(path, eps, N, c, avn0, tol, max_iter, ndigits)

    return 0


if __name__ == '__main__':
    main()
