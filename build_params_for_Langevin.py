__author__ = 'david'

import numpy as np



def read_transitions(filein, pos_par_fixed, pos_par_trans_1, pos_par_trans_2, every=1):
    fin = open(filein, 'r')
    all_lines = fin.readlines()
    fin.close()
    transitions = {}
    for i in range(0, len(all_lines), every):
        line = all_lines[i]
        if line.startswith('#'):
            continue
        line_split = line.split()
        par_fixed = float(line_split[pos_par_fixed])
        par_trans_min = float(line_split[pos_par_trans_1])
        par_trans_max = float(line_split[pos_par_trans_2])
        transitions[par_fixed] = (par_trans_min, par_trans_max)
    return transitions



def print_params(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                 pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
                 every=1, add_to_the_end=0, dpar_fixed=0, shift_below_end=0, shift_above_end=0):
    transitions = read_transitions(f'{path_in}/{filein}', pos_par_fixed, 
                                         pos_par_trans_1, pos_par_trans_2, every)
    counter = 0
    with open(f'{path_out}/{fileout}', 'w') as fo:
        for par_fixed in transitions:
            trans_min, trans_max = transitions[par_fixed]
            min_val = max(0, trans_min - shift_below)
            max_val = trans_max + shift_above
            par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
            for par in par_list:
                fo.write(f"{par_fixed:.3f} {par:.3f}\n")
                counter += 1
        par_fixed = max(transitions.keys())
        trans_min, trans_max = transitions[par_fixed]
        min_val = max(0, trans_min - shift_below_end)
        max_val = trans_max + shift_above_end
        par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
        for i in range(add_to_the_end):
            par_fixed += dpar_fixed
            for par in par_list:
                fo.write(f"{par_fixed:.3f} {par:.3f}\n")
                counter += 1
    print(f"Parameters saved to {path_out}/{fileout}")
    print(f"Total parameters: {counter}")


def print_params_ref_max(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                         pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
                         every=1, precision=0):
    transitions = read_transitions(f'{path_in}/{filein}', pos_par_fixed, 
                                         pos_par_trans_1, pos_par_trans_2, every)
    counter = 0
    with open(f'{path_out}/{fileout}', 'w') as fo:
        for par_fixed in transitions:
            trans_min, trans_max = transitions[par_fixed]
            if trans_max - trans_min > precision * 1.01:
                min_val = max(0, trans_max - shift_below)
                max_val = trans_max + shift_above
                par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
                for par in par_list:
                    fo.write(f"{par_fixed:.3f} {par:.3f}\n")
                    counter += 1
    print(f"Parameters saved to {path_out}/{fileout}")
    print(f"Total parameters: {counter}")


def main():
    
    # EPSILON = "0.0" (ASYMMETRIC)  params: (mu, sigma)

    dpar_trans = 0.004
    shift_below = 0.008
    shift_above = 0.008


    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"
    filein = f'Lotka-Volterra_transition_epsilon_0.0_Partially_AsymGauss_lambda_1e-06_tol_1e-08_N_256_c_3.00_T_0.0.txt'

    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    fileout = "params_Langevin_T0_phase_diagram_eps0_1.txt"
    pos_par_fixed = 0
    pos_par_trans_1 = 3
    pos_par_trans_2 = 4
    every = 1
    add_to_the_end = 0
    dpar_fixed = 0
    shift_below_end = 0
    shift_above_end = 0

    print_params(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                 pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, every, 
                 add_to_the_end, dpar_fixed, shift_below_end, shift_above_end)

    # print_params_ref_max(path_in, filein, path_out, fileout, dsigma, pos_par_fixed, 
    #                      pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
    #                      every, dsigma)


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
