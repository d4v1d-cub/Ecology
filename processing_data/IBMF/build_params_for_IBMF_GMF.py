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
        if par_trans_min != par_trans_max:
            transitions[par_fixed] = (par_trans_min, par_trans_max)
    return transitions
        

        
def print_params(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                 pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
                 par_fixed_list, eps, seed_block, nsampl_each):
    transitions = read_transitions(f'{path_in}/{filein}', pos_par_fixed, 
                                         pos_par_trans_1, pos_par_trans_2)
    counter = 0
    par_in_trans = transitions.keys()
    with open(f'{path_out}/{fileout}', 'w') as fo:
        for par_fixed in par_fixed_list:
            if par_fixed in par_in_trans:
                trans_min, trans_max = transitions[par_fixed]
                min_val = max(0, trans_min - shift_below)
                max_val = trans_max + shift_above
                par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
                for par in par_list:
                    fo.write(f"{eps} {par_fixed:.3f} {par:.3f} {seed_block} {nsampl_each}\n")
                    counter += 1
            else:
                closest_above = max(par_in_trans)
                closest_below = min(par_in_trans)
                for par in par_in_trans:
                    if par < par_fixed and par > closest_below:
                        closest_below = par
                    if par > par_fixed and par < closest_above:
                        closest_above = par
                trans_above_min, trans_above_max = transitions[closest_above]
                trans_below_min, trans_below_max = transitions[closest_below]
                min_val = min(trans_above_min, trans_below_min)
                max_val = max(trans_above_max, trans_below_max)
                par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
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

    dpar_trans = 0.004
    shift_below = -0.012
    shift_above = 0.036
    ndigits = 3

    eps = "0.000"
    seed_block = "1"
    nsampl_each = "10000"

    # EPSILON = "0.0" (ASYMMETRIC)  params: (mu, sigma)

    # IBMF
    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/"
    path_in = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Results/IBMF"
    N_list = [128, 256, 512, 1024]
    dpar_fixed = 0.003
    par_fixed_start = 0.000
    par_fixed_end = 0.354
    par_fixed_list = np.arange(par_fixed_start, par_fixed_end + dpar_fixed / 2, dpar_fixed)
    for i in range(len(par_fixed_list)):
        par_fixed_list[i] = round(par_fixed_list[i], ndigits)

    for N in N_list:
        filein = f'IBMF_T0_seq_RRG_PD_Lotka_Volterra_transitions_mult_av0_0.08_tol_1e-6_maxiter_10000_eps_0.000_N_{N}_c_3_damping_1.0_nseq_10.txt'

        path_out = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Scripts/Dresden/IBMF"
        fileout = f'params_IBMF_T0_seq_dif_init_conds_eps0_N_{N}.txt'
        pos_par_fixed = 0
        pos_par_trans_1 = 1
        pos_par_trans_2 = 2

        print_params(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                    pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
                    par_fixed_list, eps, seed_block, nsampl_each)
        
    
    # EPSILON = "1.0" (SYMMETRIC)  params: (mu, sigma)
    
    # epsilon = "1.000"
    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/"
    # dpar_fixed = 0.001
    # par_fixed_start = 0.001
    # par_fixed_end = 0.055
    # par_fixed_list = np.arange(par_fixed_start, par_fixed_end + dpar_fixed / 2, dpar_fixed)
    # for i in range(len(par_fixed_list)):
    #     par_fixed_list[i] = round(par_fixed_list[i], ndigits)

    # filein = f'IBMF_seq_RRG_PD_Lotka_Volterra_transitions_av0_0.08_lambda_1e-6_tol_1e-6_maxiter_10000_eps_1.000_sigma_0.000_N_1024_c_3_damping_1.0_nseq_10.txt'

    # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/IBMF"
    # fileout = f'params_IBMF_seq_phase_diagram_sigma0_1.txt'
    # pos_par_fixed = 0
    # pos_par_trans_1 = 3
    # pos_par_trans_2 = 4

    # print_params(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
    #              pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
    #              par_fixed_list, eps, seed_block, nsampl_each)
    

    return 0


if __name__ == '__main__':
    main()
