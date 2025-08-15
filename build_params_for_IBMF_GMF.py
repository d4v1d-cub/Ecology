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
        

        
def print_params(path_in, filein, path_out, fileout, dsigma, pos1, pos2, 
                 eps, seed_block, nsampl_each):
    sigma_transitions = read_transitions(f'{path_in}/{filein}', pos1, pos2)
    counter = 0
    with open(f'{path_out}/{fileout}', 'w') as fo:
        for mu in sigma_transitions:
            sigma_min, sigma_max = sigma_transitions[mu]
            sigma_list = np.arange(sigma_min, sigma_max + dsigma / 2, dsigma)
            for sigma in sigma_list[1:-1]:
                fo.write(f"{eps} {mu:.3f} {sigma:.3f} {seed_block} {nsampl_each}\n")
                counter += 1
    print(f"Parameters saved to {path_out}/{fileout}")
    print(f"Total parameters: {counter}")


def main():
    eps = "0.000"
    avn0 = "0.08"
    tol = "1e-4"
    max_iter = "10000"
    N = "1024"
    c = "3"

    dsigma = 0.004

    seed_block = "1"
    nsampl_each = "10000"

    # path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/"
    # filein = f'GMF_T0_RRG_PD_Lotka_Volterra_transitions_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}.txt'

    # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Lecce"
    # fileout = "params_GMF_T0_phase_diagram_eps0_1.txt"
    # pos1 = 3
    # pos2 = 4

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/"
    filein = f'IBMF_T0_RRG_PD_Lotka_Volterra_transitions_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}.txt'

    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Lecce"
    fileout = "params_IBMF_T0_phase_diagram_eps0_1.txt"
    pos1 = 1
    pos2 = 2
    

    print_params(path_in, filein, path_out, fileout, dsigma, pos1, pos2, 
                 eps, seed_block, nsampl_each)

    return 0


if __name__ == '__main__':
    main()
