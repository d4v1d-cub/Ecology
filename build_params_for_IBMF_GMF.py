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
                 eps, seed_block, nsampl_each, every=1, add_to_the_end=0, dpar_fixed=0, 
                 shift_below_end=0, shift_above_end=0):
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
                fo.write(f"{eps} {par_fixed:.3f} {par:.3f} {seed_block} {nsampl_each}\n")
                counter += 1
        par_fixed = max(transitions.keys())
        trans_min, trans_max = transitions[par_fixed]
        min_val = max(0, trans_min - shift_below_end)
        max_val = trans_max + shift_above_end
        par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
        for i in range(add_to_the_end):
            par_fixed += dpar_fixed
            for par in par_list:
                fo.write(f"{eps} {par_fixed:.3f} {par:.3f} {seed_block} {nsampl_each}\n")
                counter += 1
    print(f"Parameters saved to {path_out}/{fileout}")
    print(f"Total parameters: {counter}")


def print_params_ref_max(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                         pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
                         eps, seed_block, nsampl_each, every=1, precision=0):
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
                    fo.write(f"{eps} {par_fixed:.3f} {par:.3f} {seed_block} {nsampl_each}\n")
                    counter += 1
    print(f"Parameters saved to {path_out}/{fileout}")
    print(f"Total parameters: {counter}")


def main():

    shift_below = -0.004
    shift_above = -0.004

    seed_block = "1"
    nsampl_each = "1000"

    # EPSILON = "0.0" (ASYMMETRIC)  params: (mu, sigma)

    # IBMF
    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/"
    filein = f'IBMF_T0_seq_RRG_PD_Lotka_Volterra_transitions_av0_0.08_tol_1e-6_maxiter_10000_eps_0.000_N_1024_c_3_damping_0.2_nseq_10.txt'

    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/IBMF"
    fileout = "params_IBMF_T0_seq_phase_diagram_eps0_1.txt"
    pos_par_fixed = 0
    pos_par_trans_1 = 3
    pos_par_trans_2 = 4
    eps = "0.000"
    every = 1
    add_to_the_end = 0
    dpar_fixed = 0
    shift_below_end = 0
    shift_above_end = 0
    dpar_trans = 0.004
    

    # GMF
    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/"
    # filein = f'GMF_T0_seq_RRG_PD_Lotka_Volterra_transitions_av0_0.08_tol_1e-6_maxiter_10000_eps_0.000_N_1024_c_3_damping_1.0_nseq_10.txt'

    # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/GMF"
    # fileout = "params_GMF_T0_seq_phase_diagram_eps0_1.txt"
    # pos_par_fixed = 0
    # pos_par_trans_1 = 5
    # pos_par_trans_2 = 6
    # eps = "0.000"
    # every = 1
    # add_to_the_end = 0
    # dpar_fixed = 0
    # shift_below_end = 0
    # shift_above_end = 0
    # dpar_trans = 0.02

    
    

    # EPSILON = "1.0" (SYMMETRIC) params: (T, mu)

    # IBMF
    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/"
    # filein = 'IBMF_seq_RRG_PD_Lotka_Volterra_transitions_av0_0.08_lambda_1e-6_tol_1e-6_maxiter_10000_eps_1.000_sigma_0.000_N_1024_c_3_damping_1.0_nseq_10.txt'
    # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/IBMF"
    # fileout = "params_IBMF_seq_phase_diagram_sigma0_1.txt"
    # pos_par_fixed = 0
    # pos_par_trans_1 = 3
    # pos_par_trans_2 = 4
    # eps = "1.000"
    # every = 1
    # add_to_the_end = 0
    # dpar_fixed = 0
    # shift_below_end = 0
    # shift_above_end = 0
    # dpar_trans = 0.002

    # GMF
    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/"
    # filein = 'GMF_seq_RRG_PD_Lotka_Volterra_transitions_av0_0.08_lambda_1e-6_tol_1e-6_maxiter_10000_eps_1.000_sigma_0.000_N_1024_c_3_damping_1.0_nseq_10.txt'
    # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/GMF"
    # fileout = "params_GMF_seq_phase_diagram_sigma0_1.txt"
    # pos_par_fixed = 0
    # pos_par_trans_1 = 5
    # pos_par_trans_2 = 6
    # eps = "1.000"
    # every = 1
    # add_to_the_end = 0
    # dpar_fixed = 0
    # shift_below_end = 0
    # shift_above_end = 0
    # dpar_trans = 0.02
    

    print_params(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, pos_par_trans_1, 
                 pos_par_trans_2, shift_below, shift_above, eps, seed_block, nsampl_each, 
                 every, add_to_the_end, dpar_fixed, shift_below_end, shift_above_end)
    

    # print_params_ref_max(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, pos_par_trans_1, 
    #                      pos_par_trans_2, shift_below, shift_above, eps, seed_block, nsampl_each, 
    #                      every, dpar_trans)
    

    return 0


if __name__ == '__main__':
    main()
