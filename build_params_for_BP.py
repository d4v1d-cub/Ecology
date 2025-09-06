__author__ = 'david'

import numpy as np



def read_transitions(filein, pos_par_fixed, pos_par_trans_1, pos_par_trans_2):
    fin = open(filein, 'r')
    all_lines = fin.readlines()
    fin.close()
    transitions = {}
    for i in range(0, len(all_lines)):
        line = all_lines[i]
        if line.startswith('#'):
            continue
        line_split = line.split()
        par_fixed = float(line_split[pos_par_fixed])
        par_trans_min = float(line_split[pos_par_trans_1])
        par_trans_max = float(line_split[pos_par_trans_2])
        transitions[par_fixed] = (par_trans_min, par_trans_max)
    return transitions



def print_params(par_fixed_vals, path_in, filein, path_out, fileout, dpar_trans, 
                 seed_block, nsamples_each, pos_par_fixed, pos_par_trans_1, 
                 pos_par_trans_2, shift_below, shift_above):
    transitions = read_transitions(f'{path_in}/{filein}', pos_par_fixed, 
                                         pos_par_trans_1, pos_par_trans_2)
    counter = 0

    par_in_trans = transitions.keys()
    
    with open(f'{path_out}/{fileout}', 'w') as fo:
        for par_fixed in par_fixed_vals:
            if par_fixed in par_in_trans:
                trans_min, trans_max = transitions[par_fixed]
                min_val = max(0, trans_min - shift_below)
                max_val = trans_max + shift_above
                
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
                min_val = (trans_above_min + trans_below_min) / 2
                max_val = (trans_above_max + trans_below_max) / 2
            par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
            for par in par_list:
                fo.write(f"{par_fixed:.3f} {par:.3f} {seed_block} {nsamples_each}\n")
                counter += 1

    print(f"Parameters saved to {path_out}/{fileout}")
    print(f"Total parameters: {counter}")



def main():
    
    dpar_trans = 0.002
    shift_below = 0.008
    shift_above = 0.008

    seed_block = "1"
    nsamples_each = "1000"

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/BP/Results/"
    filein = f'BP_transition_sigma0_sequential_changing_order.txt'

    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/BP"
    fileout = "params_BP_seq_phase_diagram_sigma0_1.txt"
    pos_par_fixed = 0
    pos_par_trans_1 = 1
    pos_par_trans_2 = 1

    Tmin = 0.001
    Tmax = 0.033
    dT = 0.001

    par_fixed_vals = np.arange(Tmin, Tmax + dT / 2, dT)

    print_params(par_fixed_vals, path_in, filein, path_out, fileout, dpar_trans,
                 seed_block, nsamples_each, pos_par_fixed, 
                 pos_par_trans_1, pos_par_trans_2, shift_below, shift_above)


    return 0


if __name__ == '__main__':
    main()
