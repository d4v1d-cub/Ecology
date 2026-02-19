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
                 par_fixed_list, already_printed=None):
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
                        fo.write(f"{par_fixed:.3f} {par:.3f}\n")
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
                min_val = max(0, min_val - shift_below)
                max_val = max(trans_above_max, trans_below_max)
                max_val = max_val + shift_above
                par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
                for par in par_list:
                    if (par_fixed, par) not in already_printed:
                        fo.write(f"{par_fixed:.3f} {par:.3f}\n")
                        already_printed.append((par_fixed, par))
                        counter += 1
        
    print(f"Parameters saved to {path_out}/{fileout}")
    print(f"Total parameters: {counter}")
    return already_printed, counter


def print_params_search_below_line(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                                   pos_par_trans_1, pos_par_trans_2, 
                                   par_fixed_list, already_printed=None):
    transitions = read_transitions(f'{path_in}/{filein}', pos_par_fixed, 
                                         pos_par_trans_1, pos_par_trans_2)
    counter = 0
    par_in_trans = transitions.keys()
    if already_printed is None:
        already_printed = []
    with open(f'{path_out}/{fileout}', 'a') as fo:
        for par_fixed in par_fixed_list:
            if par_fixed in par_in_trans:
                _, trans_max = transitions[par_fixed]
                min_val = 0
                max_val = trans_max
                par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
                for par in par_list:
                    if (par_fixed, par) not in already_printed:
                        fo.write(f"{par_fixed:.3f} {par:.3f}\n")
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
                _, trans_above_max = transitions[closest_above]
                _, trans_below_max = transitions[closest_below]
                min_val = 0
                max_val = max(trans_above_max, trans_below_max)
                par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
                for par in par_list:
                    if (par_fixed, par) not in already_printed:
                        fo.write(f"{par_fixed:.3f} {par:.3f}\n")
                        already_printed.append((par_fixed, par))
                        counter += 1
        
    print(f"Parameters saved to {path_out}/{fileout}")
    print(f"Total parameters: {counter}")
    return already_printed, counter


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
    shift_above = 0.024
    ndigits = 3

    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"
    # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    # path_in = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Langevin/Results"
    # path_out = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    path_in = "/mnt/d/Research/Ecology/Langevin/Results/"
    # path_out = "/mnt/d/Research/Ecology/Scripts/Dresden/Langevin"
    path_out = "/mnt/d/Research/Ecology/Scripts/Lecce/Langevin"
    N_list = [128, 256, 512, 1024, 2048, 4096]
    dpar_fixed = 0.03
    par_fixed_start = 0.360
    par_fixed_end = 0.360
    par_fixed_list = np.arange(par_fixed_start, par_fixed_end + dpar_fixed / 2, dpar_fixed)
    for i in range(len(par_fixed_list)):
        par_fixed_list[i] = round(par_fixed_list[i], ndigits)

    for N in N_list:
        filein = f'Lotka-Volterra_transition_div_epsilon_0.0_Partially_AsymGauss_lambda_1e-06_tol_1e-08_N_{N}_c_3.00_T_0.0.txt'

        fileout = f'params_Langevin_T0_phase_diagram_eps0_N_{N}.txt'
        pos_par_fixed = 0
        pos_par_trans_1 = 1
        pos_par_trans_2 = 2

        print_params(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                    pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
                    par_fixed_list)

    dpar_trans = 0.004
    shift_below = 0.000
    shift_above = 0.002
    ndigits = 3

    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"
    # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    # path_in = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Langevin/Results"
    # path_out = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    path_in = "/mnt/d/Research/Ecology/Langevin/Results/"
    # path_out = "/mnt/d/Research/Ecology/Scripts/Dresden/Langevin"
    path_out = "/mnt/d/Research/Ecology/Scripts/Lecce/Langevin"
    N_list = [8192]
    dpar_fixed = 0.03
    par_fixed_start = -0.330
    par_fixed_end = 0.360
    par_fixed_list = np.arange(par_fixed_start, par_fixed_end + dpar_fixed / 2, dpar_fixed)
    for i in range(len(par_fixed_list)):
        par_fixed_list[i] = round(par_fixed_list[i], ndigits)

    for N in N_list:
        filein = f'Lotka-Volterra_transition_div_epsilon_0.0_Partially_AsymGauss_lambda_1e-06_tol_1e-08_N_{N}_c_3.00_T_0.0.txt'

        fileout = f'params_Langevin_T0_phase_diagram_eps0_N_{N}.txt'
        pos_par_fixed = 0
        pos_par_trans_1 = 1
        pos_par_trans_2 = 2

        print_params(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                    pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
                    par_fixed_list)
        
    
    dpar_trans = 0.004
    shift_below = 0.000
    shift_above = 0.002
    ndigits = 3

    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"
    # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    # path_in = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Langevin/Results"
    # path_out = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    path_in = "/mnt/d/Research/Ecology/Langevin/Results/"
    # path_out = "/mnt/d/Research/Ecology/Scripts/Dresden/Langevin"
    path_out = "/mnt/d/Research/Ecology/Scripts/Lecce/Langevin"
    N_list = [16384]
    dpar_fixed = 0.03
    par_fixed_start = -0.330
    par_fixed_end = 0.360
    par_fixed_list = np.arange(par_fixed_start, par_fixed_end + dpar_fixed / 2, dpar_fixed)
    for i in range(len(par_fixed_list)):
        par_fixed_list[i] = round(par_fixed_list[i], ndigits)

    for N in N_list:
        filein = f'Lotka-Volterra_transition_div_epsilon_0.0_Partially_AsymGauss_lambda_1e-06_tol_1e-08_N_{N}_c_3.00_T_0.0.txt'

        fileout = f'params_Langevin_T0_phase_diagram_eps0_N_{N}.txt'
        pos_par_fixed = 0
        pos_par_trans_1 = 1
        pos_par_trans_2 = 2

        print_params(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                    pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
                    par_fixed_list)
        

    dpar_trans = 0.004
    shift_below = 0.000
    shift_above = 0.002
    ndigits = 3

    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"
    # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    # path_in = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Langevin/Results"
    # path_out = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    path_in = "/mnt/d/Research/Ecology/Langevin/Results/"
    # path_out = "/mnt/d/Research/Ecology/Scripts/Dresden/Langevin"
    path_out = "/mnt/d/Research/Ecology/Scripts/Lecce/Langevin"
    # N_list = [128, 256, 512, 1024, 2048, 4096]
    N_list = [32768]
    dpar_fixed = 0.03
    par_fixed_start = -0.330
    par_fixed_end = 0.360
    par_fixed_list = np.arange(par_fixed_start, par_fixed_end + dpar_fixed / 2, dpar_fixed)
    for i in range(len(par_fixed_list)):
        par_fixed_list[i] = round(par_fixed_list[i], ndigits)

    for N in N_list:
        filein = f'Lotka-Volterra_transition_div_epsilon_0.0_Partially_AsymGauss_lambda_1e-06_tol_1e-08_N_{N}_c_3.00_T_0.0.txt'

        fileout = f'params_Langevin_T0_phase_diagram_eps0_N_{N}.txt'
        pos_par_fixed = 0
        pos_par_trans_1 = 1
        pos_par_trans_2 = 2

        print_params(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                    pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
                    par_fixed_list)
        

    # EPSILON = "0.0" (ASYMMETRIC)  params: (mu, sigma)  BIGGER N

    # dpar_trans = 0.01
    # ndigits = 3

    # # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"
    # # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    # path_in = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Langevin/Results"
    # path_out = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Scripts/Dresden/Langevin"
    # N_list = [8192, 16384, 32768]
    # dpar_fixed = 0.03
    # par_fixed_start = -0.330
    # par_fixed_end = 0.360
    # par_fixed_list = np.arange(par_fixed_start, par_fixed_end + dpar_fixed / 2, dpar_fixed)
    # for i in range(len(par_fixed_list)):
    #     par_fixed_list[i] = round(par_fixed_list[i], ndigits)

    # filein = f'Lotka-Volterra_transition_div_epsilon_0.0_Partially_AsymGauss_lambda_1e-06_tol_1e-08_N_4096_c_3.00_T_0.0.txt'
    
    # for N in N_list:
    #     fileout = f'params_Langevin_T0_phase_diagram_eps0_N_{N}.txt'
    #     pos_par_fixed = 0
    #     pos_par_trans_1 = 1
    #     pos_par_trans_2 = 2

    #     print_params_search_below_line(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
    #                 pos_par_trans_1, pos_par_trans_2, par_fixed_list)
        

    # EPSILON = "0.5" (PARTIALLY ASYMMETRIC)  params: (mu, sigma)

    # dpar_trans = 0.004
    # shift_below = 0.004
    # shift_above = 0.004
    # ndigits = 3

    # path_in = "/mnt/d/Research/Ecology/Langevin/Results/"
    # N_list = [1024, 4096]
    # dpar_fixed = 0.012
    # par_fixed_start = -0.333
    # par_fixed_end = 0.351
    # par_fixed_list = np.arange(par_fixed_start, par_fixed_end + dpar_fixed / 2, dpar_fixed)
    # for i in range(len(par_fixed_list)):
    #     par_fixed_list[i] = round(par_fixed_list[i], ndigits)

    
    # for N in N_list:
    #     filein_mult = f'Lotka-Volterra_transition_mult_epsilon_0.5_Partially_AsymGauss_lambda_1e-06_tol_1e-08_N_{N}_c_3.00_T_0.0.txt'
    #     filein_div = f'Lotka-Volterra_transition_div_epsilon_0.5_Partially_AsymGauss_lambda_1e-06_tol_1e-08_N_{N}_c_3.00_T_0.0.txt'


    #     path_out = "/mnt/d/Research/Ecology/Scripts/Dresden/Langevin"
    #     fileout = f'params_Langevin_T0_phase_diagram_eps05_N_{N}.txt'
    #     pos_par_fixed = 0
    #     pos_par_trans_1 = 1
    #     pos_par_trans_2 = 2

    #     already_printed, counter1 = print_params(path_in, filein_mult, path_out, fileout, dpar_trans, pos_par_fixed, 
    #                                             pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
    #                                             par_fixed_list)
    #     _, counter2 = print_params(path_in, filein_div, path_out, fileout, dpar_trans, pos_par_fixed, 
    #                               pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
    #                               par_fixed_list, already_printed)
    #     print(f"Total unique parameters for N={N}: {counter1 + counter2}")


    return 0


if __name__ == '__main__':
    main()
