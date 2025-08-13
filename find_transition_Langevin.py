__author__ = 'david'

import numpy as np




def find_all_trans(path, eps, lda, h, N, c, sigma, ndigits):
    fout = open(f'{path}/Lotka-Volterra_transitions_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_h_{h}_N_{N}_c_{c}_sigma_{sigma}.txt', 'w')
    
    fin = open(f'{path}/Lotka-Volterra_summary_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_h_{h}_N_{N}_c_{c}_sigma_{sigma}.txt', 'r')
    fin.readline()  # Skip header line

    lines = fin.readlines()

    index = 0
    min_mu = {}
    max_mu = {}
    transition_found = {}
    while index < len(lines) - 1:
        line_split = lines[index].split()
        T = float(line_split[0])
        mu = float(line_split[1])
        if T not in min_mu:
            min_mu[T] = mu
            max_mu[T] = mu
        else:
            if mu < min_mu[T]:
                min_mu[T] = mu
            if mu > max_mu[T]:
                max_mu[T] = mu

        num_div = int(line_split[4])

        line_split_below = lines[index + 1].split()
        T_below = float(line_split_below[0])
        if T_below == T and not T in transition_found:
            num_div_below = int(line_split_below[4])
            if num_div_below >= 50 and num_div < 50:
                transition_found[T] = True
                mu_below = float(line_split_below[1])
                fout.write(f"{T:.{ndigits}f}\t{mu:.{ndigits}f}\t{mu_below:.{ndigits}f}\n")
        index += 1
    
    fin.close()
    fout.close()
    return min_mu, max_mu


def print_min_max_mu(min_mu, max_mu, path, eps, lda, h, N, c, sigma):
    fout = open(f'{path}/Lotka-Volterra_min_max_mu_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_h_{h}_N_{N}_c_{c}_sigma_{sigma}.txt', 'w')
    fout.write("# T min_mu max_mu\n")
    for T in sorted(min_mu.keys()):
        fout.write(f"{T:.3f}\t{min_mu[T]:.3f}\t{max_mu[T]:.3f}\n")
    fout.close()


def main():
    eps = "1.0"
    lda = "1e-06"
    N = "256"
    c = "3.00"
    h = "0.001"
    sigma = "0"

    path = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"

    ndigits = 3

    min_mu, max_mu = find_all_trans(path, eps, lda, h, N, c, sigma, ndigits)
    print_min_max_mu(min_mu, max_mu, path, eps, lda, h, N, c, sigma)

    return 0


if __name__ == '__main__':
    main()
