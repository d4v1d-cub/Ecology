__author__ = 'david'

import numpy as np
from scipy.special import lambertw


def prob(n0, c):
    lambw = lambertw(c).real  # Get the real part of the Lambert W function
    sum = 0.5 * np.log((1 + lambw) / (1 - lambw))
    for i in range(n0 // 2):
        sum -= np.power(lambw, 2 * i + 1) / (2 * i + 1)
    return 1 - np.exp(-sum)

def prob_IBMF(n0, c):
    lambw = lambertw(c).real  # Get the real part of the Lambert W function
    sum = -np.log(1 - lambw)
    for i in range(1, n0):
        sum -= np.power(lambw, i) / i
    return 1 - np.exp(-sum)


def print_pairs(c_vals, mu, fileout):
    n0 = 3
    mu_c = 1.0 / np.cos(np.pi / n0)
    while mu < mu_c:
        n0 += 2
        mu_c = 1.0 / np.cos(np.pi / n0)
    print(f"n0 = {n0}, mu_c = {mu_c}")
    
    fout = open(fileout, 'w')
    fout.write("# c p(fluc)\n")
    for c in c_vals:
        p = prob(n0, c)
        fout.write(f'{c}\t{p}\n')
        print(f'{c}\t{p}')


def print_pairs_IBMF(n0, c_vals, fileout):
    fout = open(fileout, 'w')
    fout.write("# c p(fluc)\n")
    for c in c_vals:
        p = prob_IBMF(n0, c)
        fout.write(f'{c}\t{p}\n')
        print(f'{c}\t{p}')



def main():
    cmin = 0.0001
    cmax = np.e
    dc = 0.0001
    c_vals = np.arange(cmin, cmax, dc)

    path = '/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/Comparison'

    mu = 1.05
    fileout = f'{path}/p_fluc_directed_ER_mu_{mu}.txt'
    print_pairs(c_vals, mu, fileout)

    n0 = 4
    fileout = f'{path}/p_fluc_IBMF_directed_ER_n0_{n0}.txt'
    print_pairs_IBMF(n0, c_vals, fileout)

    return 0


if __name__ == '__main__':
    main()
