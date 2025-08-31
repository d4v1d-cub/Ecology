__author__ = 'david'

import numpy as np



def read_transitions(filein, pos1, pos2):
    fin = open(filein, 'r')
    all_lines = fin.readlines()
    fin.close()
    sigma_transitions = {}
    for line in all_lines:
        line_split = line.split()
        mu = float(line_split[0])
        sigma_min = float(line_split[pos1])
        sigma_max = float(line_split[pos2])
        sigma_transitions[mu] = (sigma_min, sigma_max)
    return sigma_transitions



def print_params(mu_vals, path_in, filein, path_out, fileout, sigma_margin, dsigma, pos1, pos2):
    sigma_transitions = read_transitions(f'{path_in}/{filein}', pos1, pos2)
    counter = 0
    with open(f'{path_out}/{fileout}', 'w') as fo:
        mu_theo = sigma_transitions.keys()
        for mu in mu_vals:
            mu = round(mu, 3)
            if mu in mu_theo:
                sigma_min, sigma_max = sigma_transitions[mu]
                sigma_av = (sigma_min + sigma_max) / 2
                sigma_left = max(0, sigma_av - sigma_margin)
                sigma_right = sigma_av + sigma_margin
                sigma_list = np.arange(sigma_left, sigma_right + dsigma / 2, dsigma)
                for sigma in sigma_list:
                    fo.write(f"{mu:.3f} {sigma:.3f}\n")
                    counter += 1
            else:
                sigma_left = 0
                sigma_right = 0.1
                sigma_list = np.arange(sigma_left, sigma_right + dsigma / 2, dsigma)
                for sigma in sigma_list:
                    fo.write(f"{mu:.3f} {sigma:.3f}\n")
                    counter += 1
    print(f"Parameters saved to {path_out}/{fileout}")
    print(f"Total parameters: {counter}")


def main():
    
    # EPSILON = "0.0" (ASYMMETRIC)  params: (mu, sigma)

    eps = "0.000"
    avn0 = "0.08"
    tol = "1e-6"
    max_iter = "10000"
    N = "1024"
    c = "3"
    damping = "1.0"
    nseq = "10"

    dsigma = 0.004
    sigma_margin = 0.08


    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/"
    filein = f'GMF_T0_seq_RRG_PD_Lotka_Volterra_transitions_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt'

    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    fileout = "params_Langevin_T0_phase_diagram_eps0_1.txt"
    pos1 = 5
    pos2 = 6

    mu0 = 0.0
    muf = 0.354
    dmu = 0.009
    mu_vals = np.arange(mu0, muf + dmu / 2, dmu)
    print(f"Number of mu values: {len(mu_vals)}")

    print_params(mu_vals, path_in, filein, path_out, fileout, sigma_margin, dsigma, pos1, pos2)


    # EPSILON = "1.0" (SYMMETRIC)  params: (T, mu)

    # dmu = 0.004
    # mu_margin = 0.05


    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/"
    # filein = "GMF_seq_RRG_PD_Lotka_Volterra_transitions_av0_0.08_lambda_1e-6_tol_1e-6_maxiter_10000_eps_1.000_sigma_0.000_N_1024_c_3_damping_1.0_nseq_10.txt"

    # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    # fileout = "params_Langevin_phase_diagram_sigma0_1.txt"
    # pos1 = 5
    # pos2 = 6

    # T0 = 0.001
    # Tf = 0.054
    # dT = 0.002
    # T_vals = np.arange(T0, Tf + dT / 2, dT)
    # print(f"Number of T values: {len(T_vals)}")

    # print_params(T_vals, path_in, filein, path_out, fileout, mu_margin, dmu, pos1, pos2)

    return 0


if __name__ == '__main__':
    main()
