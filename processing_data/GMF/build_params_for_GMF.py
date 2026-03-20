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
                 par_fixed_list, eps, seed_block, nsampl_each, N, T, lda, already_printed=None):
    transitions = read_transitions(f'{path_in}/{filein}', pos_par_fixed, 
                                         pos_par_trans_1, pos_par_trans_2)
    counter = 0
    par_in_trans = transitions.keys()
    if already_printed is None:
        already_printed = []
    with open(f'{path_out}/{fileout}', 'a') as fo:
        for par_fixed in par_fixed_list:
            if par_fixed in par_in_trans:
                trans_min, trans_max = transitions[par_fixed]
                min_val = max(0, trans_min - shift_below)
                max_val = trans_max + shift_above
                par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
                for par in par_list:
                    if (par_fixed, par) not in already_printed:
                        fo.write(f"{eps} {T} {lda} {par_fixed:.3f} {par:.3f} {seed_block} {nsampl_each} {N}\n")
                        already_printed.append((par_fixed, par))
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
                    if (par_fixed, par) not in already_printed:
                        fo.write(f"{eps} {T} {lda} {par_fixed:.3f} {par:.3f} {seed_block} {nsampl_each} {N}\n")
                        already_printed.append((par_fixed, par))
                        counter += 1
        
    print(f"Parameters saved to {path_out}/{fileout}")
    print(f"Total parameters: {counter}")
    return already_printed, counter




def main():
    
    # EPSILON = "0.000" (ASYMMETRIC)  params: (mu, sigma)

    dpar_trans = 0.020
    shift_below = 0.020
    shift_above = 0.020
    ndigits = 3

    eps = "0.000"
    seed_block = "1"
    nsampl_each = "10000"
    T = "0.000"
    lda = "0.000"

    # IBMF
    path_in = "/mnt/d/Research/Ecology/Langevin/Results"
    N_list = [1024, 4096]
    dpar_fixed = 0.03
    par_fixed_start = 0.00
    par_fixed_end = 0.36
    par_fixed_list = np.arange(par_fixed_start, par_fixed_end + dpar_fixed / 2, dpar_fixed)
    for i in range(len(par_fixed_list)):
        par_fixed_list[i] = round(par_fixed_list[i], ndigits)

    for N in N_list:
        filein_mult = f'Lotka-Volterra_transition_mult_epsilon_0.0_Partially_AsymGauss_lambda_1e-06_tol_1e-08_N_{N}_c_3.00_T_0.0.txt'
        filein_div = f'Lotka-Volterra_transition_div_epsilon_0.0_Partially_AsymGauss_lambda_1e-06_tol_1e-08_N_{N}_c_3.00_T_0.0.txt'

        path_out = "/mnt/d/Research/Ecology/Scripts/idra/GMF"
        fileout = f'params_GMF_T0_seq_dif_init_conds_eps0_N_{N}.txt'
        pos_par_fixed = 0
        pos_par_trans_1 = 1
        pos_par_trans_2 = 2

        already_printed, counter1 = print_params(path_in, filein_mult, path_out, fileout, dpar_trans, pos_par_fixed, 
                                                 pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
                                                 par_fixed_list, eps, seed_block, nsampl_each, N, T, lda)
        _, counter2 = print_params(path_in, filein_div, path_out, fileout, dpar_trans, pos_par_fixed, 
                                  pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
                                  par_fixed_list, eps, seed_block, nsampl_each, N, T, lda, already_printed)
        print(f"Total unique parameters for N={N}: {counter1 + counter2}")

    return 0


if __name__ == '__main__':
    main()
