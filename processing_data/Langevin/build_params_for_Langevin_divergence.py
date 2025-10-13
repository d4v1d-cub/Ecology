__author__ = 'david'

import numpy as np



def read_transitions(filein, pos_par_fixed, pos_par_trans_1, pos_par_trans_2, every=1):
    fin = open(filein, 'r')
    all_lines = fin.readlines()
    fin.close()
    transitions_x = []
    transitions_y = []
    transitions = {}
    for i in range(0, len(all_lines)):
        line = all_lines[i]
        if line.startswith('#'):
            continue
        line_split = line.split()
        par_fixed = float(line_split[pos_par_fixed])
        par_trans_min = float(line_split[pos_par_trans_1])
        par_trans_max = float(line_split[pos_par_trans_2])
        if not(par_trans_min == 0  and par_trans_max == 0):
            transitions_x.append(par_fixed)
            transitions_y.append((par_trans_min + par_trans_max) / 2)
            transitions[par_fixed] = (par_trans_min, par_trans_max)
    return transitions_x, transitions_y, transitions



def print_params(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                 pos_par_trans_1, pos_par_trans_2, shift_below_fit, shift_above_fit, 
                 par_fixed_list, shift_below_trans, shift_above_trans):
    transitions_x, transitions_y, transitions = read_transitions(f'{path_in}/{filein}', pos_par_fixed, 
                                                                 pos_par_trans_1, pos_par_trans_2)
    m, b = np.polyfit(transitions_x, transitions_y, 1)
    print(m, b)
    counter = 0
    with open(f'{path_out}/{fileout}', 'w') as fo:
        for par_fixed in par_fixed_list:
            if par_fixed not in transitions_x:
                trans = m * par_fixed + b
                min_val = max(0, trans - shift_below_fit)
                max_val = trans + shift_above_fit
                par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
                for par in par_list:
                    fo.write(f"{par_fixed:.3f} {par:.3f}\n")
                    counter += 1
            else:
                trans_min, trans_max = transitions[par_fixed]
                min_val = max(0, trans_min - shift_below_trans)
                max_val = trans_max + shift_above_trans
                par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
                for par in par_list:
                    fo.write(f"{par_fixed:.3f} {par:.3f}\n")
                    counter += 1

    print(f"Parameters saved to {path_out}/{fileout}")
    print(f"Total parameters: {counter}")


def main():
    
    # EPSILON = "0.0" (ASYMMETRIC)  params: (mu, sigma)

    # dpar_trans = 0.004
    # shift_below_fit = 0.016
    # shift_above_fit = 0.016
    # shift_below_trans = -dpar_trans
    # shift_above_trans = dpar_trans
    # ndigits = 3

    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"
    # N_list = [128, 256, 512, 1024, 2048, 4096]
    # dpar_fixed = 0.003
    # par_fixed_start = 0.000
    # par_fixed_end = 0.351
    # par_fixed_list = np.arange(par_fixed_start, par_fixed_end + dpar_fixed / 2, dpar_fixed)
    # for i in range(len(par_fixed_list)):
    #     par_fixed_list[i] = round(par_fixed_list[i], ndigits)

    # for N in N_list:
    #     filein = f'Lotka-Volterra_transition_div_epsilon_0.0_Partially_AsymGauss_lambda_1e-06_tol_1e-08_N_{N}_c_3.00_T_0.0.txt'

    #     path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    #     fileout = f'params_Langevin_T0_phase_diagram_eps0_N_{N}.txt'
    #     pos_par_fixed = 0
    #     pos_par_trans_1 = 1
    #     pos_par_trans_2 = 2

    #     print_params(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
    #                 pos_par_trans_1, pos_par_trans_2, shift_below_fit, shift_above_fit, 
    #                 par_fixed_list, shift_below_trans, shift_above_trans)
        

    dpar_trans = 0.01
    shift_below_fit = 0.2
    shift_above_fit = 0.2
    shift_below_trans = -dpar_trans
    shift_above_trans = dpar_trans
    ndigits = 3

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"
    N_list = [128, 256, 512, 1024, 2048, 4096]
    dpar_fixed = 0.05
    par_fixed_start = -0.8
    par_fixed_end = -0.05
    par_fixed_list = np.arange(par_fixed_start, par_fixed_end + dpar_fixed / 2, dpar_fixed)
    for i in range(len(par_fixed_list)):
        par_fixed_list[i] = round(par_fixed_list[i], ndigits)

    for N in N_list:
        filein = f'Lotka-Volterra_transition_div_epsilon_0.0_Partially_AsymGauss_lambda_1e-06_tol_1e-08_N_{N}_c_3.00_T_0.0.txt'

        path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
        fileout = f'params_Langevin_T0_phase_diagram_eps0_N_{N}_negative.txt'
        pos_par_fixed = 0
        pos_par_trans_1 = 1
        pos_par_trans_2 = 2

        print_params(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                    pos_par_trans_1, pos_par_trans_2, shift_below_fit, shift_above_fit, 
                    par_fixed_list, shift_below_trans, shift_above_trans)

    
    return 0


if __name__ == '__main__':
    main()
