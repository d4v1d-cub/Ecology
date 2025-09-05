__author__ = 'david'


def find_transition(lines, position):
    index = 0
    transitions = {}
    transition_found = {}
    while index < len(lines) - 1:
        line_split = lines[index].split()
        par_key = float(line_split[0])
        par_trans = float(line_split[1])

        num = int(line_split[position])
        nsamples = int(line_split[-1])

        if num >= nsamples / 2:
            if not par_key in transition_found:
                transitions[par_key] = (0, par_trans)
                transition_found[par_key] = True
        
        line_split_below = lines[index + 1].split()
        key_below = float(line_split_below[0])
        if key_below == par_key:
            if not par_key in transition_found:
                num_below = int(line_split_below[position])
                if num_below >= nsamples / 2 and num < nsamples / 2:
                    transition_found[par_key] = True
                    par_trans_below = float(line_split_below[1])
                    transitions[par_key] = (par_trans, par_trans_below)
        index += 1
    return transitions



def find_all_trans(path, eps, lda, tol_fixed_point, N, c, T, ndigits):
    fout = open(f'{path}/Lotka-Volterra_transition_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'w')
    
    fin = open(f'{path}/Lotka-Volterra_summary_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'r')
    fin.readline()  # Skip header line

    lines = fin.readlines()

    transitions_div = find_transition(lines, 3)
    transitions_multiple_eq = find_transition(lines, 4)
    
    
    keys_list = [transitions_div.keys(), transitions_multiple_eq.keys()]
    mu_list = sorted(set().union(*keys_list))
    for mu in mu_list:
        if mu not in transitions_div:
            transitions_div[mu] = (0, 0)
        if mu not in transitions_multiple_eq:
            transitions_multiple_eq[mu] = (0, 0)
        sigma_div, sigma_below_div = transitions_div[mu]
        sigma_multiple_eq, sigma_below_multiple_eq = transitions_multiple_eq[mu]
        fout.write(f"{mu:.{ndigits}f}\t{sigma_div:.{ndigits}f}\t{sigma_below_div:.{ndigits}f}\t{sigma_multiple_eq:.{ndigits}f}\t{sigma_below_multiple_eq:.{ndigits}f}\n")
    
    fin.close()
    fout.close()



def main():
    eps = "0.0"
    lda = "1e-06"
    N = "256"
    c = "3.00"
    T = "0.0"
    tol_fixed_point = "1e-08"

    path = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"

    ndigits = 3

    find_all_trans(path, eps, lda, tol_fixed_point, N, c, T, ndigits)

    return 0


if __name__ == '__main__':
    main()
