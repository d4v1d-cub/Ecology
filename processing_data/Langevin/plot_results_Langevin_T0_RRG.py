__author__ = 'david'

import numpy as np
from matplotlib import pyplot as plt



def get_pairs(path, eps, lda, h, N, c, T, pos1, pos2):
    fin = open(f'{path}/Lotka-Volterra_summary_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_h_{h}_N_{N}_c_{c}_T_{T}.txt', 'r')
    fin.readline()  # Skip header line
    lines = fin.readlines()
    fin.close()

    pairs_mu = {}
    for line in lines:
        line_split = line.split()
        mu = float(line_split[0])
        if mu not in pairs_mu:
            pairs_mu[mu] = []
        pairs_mu[mu].append((float(line_split[pos1]), float(line_split[pos2])))
    return pairs_mu
    


def plot_pairs(pairs_mu, path, eps, lda, h, N, c, T, pos1_label, pos2_label, pos1_str, pos2_str):
    plt.figure(figsize=(8, 6))
    for mu, pairs in pairs_mu.items():
        x_vals, y_vals = zip(*pairs)
        plt.scatter(x_vals, y_vals, label=f"mu={mu:.3f}", alpha=0.6)
        plt.xlabel(pos1_label)
        plt.ylabel(pos2_label)
        plt.yscale('log')
        plt.savefig(f'{path}/Lotka-Volterra_pairs_{pos1_str}_{pos2_str}_mu_{mu:.3f}_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_h_{h}_N_{N}_c_{c}_T_{T}.png')
        plt.clf()
    


def main():
    eps = "0.000"
    lda = "1e-06"
    N = "256"
    c = "3.00"
    h = "0.001"
    T = "0.000"

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"
    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/Plots/"

    pos1 = 1
    pos2 = 6

    pos1_label = r'$\sigma$'
    pos2_label = r'$std(\langle n_i \rangle)$'
    pos1_str = "sigma"
    pos2_str = "std_av_ni"

    pairs_mu = get_pairs(path_in, eps, lda, h, N, c, T, pos1, pos2)
    plot_pairs(pairs_mu, path_out, eps, lda, h, N, c, T, pos1_label, pos2_label, pos1_str, pos2_str)

    return 0


if __name__ == '__main__':
    main()
