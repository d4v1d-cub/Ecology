__author__ = 'david'

import numpy as np


def read_transitions_largeN(filein, pos_par_fixed, pos_par_trans_1, pos_par_trans_2, pos_N, every=1):
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
        N = int(line_split[pos_N])
        if par_trans_min != par_trans_max:
            transitions[(par_fixed, N)] = (par_trans_min, par_trans_max)
    return transitions

        
def print_params_largeN(path_in, filein, path_out, fileout, dpar_trans, pos_par_fixed, 
                        pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
                        par_fixed_list, eps, seed0, nsampl_each, nsampl_total, pos_N, already_printed=None):
    transitions = read_transitions_largeN(f'{path_in}/{filein}', pos_par_fixed, 
                                          pos_par_trans_1, pos_par_trans_2, pos_N)
    counter = 0
    keys = transitions.keys()
    if already_printed is None:
        already_printed = []
    with open(f'{path_out}/{fileout}', 'w') as fo:
        for par_trans, N in keys:
            if par_trans in par_fixed_list:
                trans_min, trans_max = transitions[(par_trans, N)]
                min_val = max(0, trans_min - shift_below)
                max_val = trans_max + shift_above
                par_list = np.arange(min_val, max_val + dpar_trans / 2, dpar_trans)
                for par in par_list:
                    if (par_trans, par, N) not in already_printed:
                        for seed_block in range(seed0, nsampl_total, nsampl_each[N]):
                            fo.write(f"{eps} {par_trans:.3f} {par:.3f} {seed_block} {nsampl_each[N]} {N}\n")
                            counter += 1
                        already_printed.append((par_trans, par, N))
    print(f"Parameters saved to {path_out}/{fileout}")
    print(f"Total parameters: {counter}")
    return already_printed, counter


def main():
    
    # EPSILON = "0.0" (ASYMMETRIC)  mu=0.000  large N  params: (mu, sigma)

    dpar_trans = 0.004
    shift_below = 0.000
    shift_above = -0.002
    ndigits = 3

    eps = "0.000"
    mu = "0.000"
    seed0 = 1
    nsampl_each = {}
    for N in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]:
        nsampl_each[N] = 10000
    nsampl_each[1048576] = 1000
    nsampl_each[2097152] = 500
    nsampl_each[4194304] = 200
    nsampl_each[8388608] = 200

    nsampl_total = 10000

    # IBMF
    path_in = "/mnt/d/Research/Ecology/Results/IBMF/"
    par_fixed_list = [round(float(mu), ndigits)]

    filein_mult = f'IBMF_seq_RRG_T_0.000_lambda_0.000_PD_Lotka_Volterra_transitions_mult_av0_0.5_dn_0.5_ninitconds_10_tol_1e-6_maxiter_10000_eps_{eps}_mu_{mu}_c_3_damping_0.2_nseq_1.txt'
    path_out = "/mnt/d/Research/Ecology/Scripts/Lecce/IBMF"
    fileout = f'params_IBMF_T0_seq_largeN_eps0_2.txt'
    pos_par_fixed = 1
    pos_par_trans_1 = 2
    pos_par_trans_2 = 3
    pos_N = 0

    print_params_largeN(path_in, filein_mult, path_out, fileout, dpar_trans, pos_par_fixed, 
                        pos_par_trans_1, pos_par_trans_2, shift_below, shift_above, 
                        par_fixed_list, eps, seed0, nsampl_each, nsampl_total, pos_N)
    return 0


if __name__ == '__main__':
    main()
