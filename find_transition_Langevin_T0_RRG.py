__author__ = 'david'


def find_transition(lines, position):
    index = 0
    transitions = {}
    while index < len(lines) - 1:
        line_split = lines[index].split()
        par_key = float(line_split[0])
        par_trans = float(line_split[1])

        num = int(line_split[position])
        nsamples = int(line_split[-1])

        
        if not par_key in transitions:
            line_split_below = lines[index + 1].split()
            key_below = float(line_split_below[0])
            if key_below == par_key:
                num_samples_below = int(line_split_below[-1])
                num_below = int(line_split_below[position])
                if num_below >= num_samples_below / 2 and num < nsamples / 2:
                    par_trans_below = float(line_split_below[1])
                    transitions[par_key] = (par_trans, par_trans_below)
        index += 1
    return transitions


def find_identify_transition(lines, position_1, position_2):
    index = 0
    transitions = {}
    trans_type = {}
    while index < len(lines) - 1:
        line_split = lines[index].split()
        par_key = float(line_split[0])
        par_trans = float(line_split[1])

        num_1 = int(line_split[position_1])
        nsamples = int(line_split[-1])

        
        if not par_key in transitions: 
            line_split_below = lines[index + 1].split()
            key_below = float(line_split_below[0])
            if key_below == par_key:
                num_samples_below = int(line_split_below[-1])
                num_1_below = int(line_split_below[position_1])
                num_2_below = int(line_split_below[position_2])
                if num_1_below >= num_samples_below / 2 and num_1 < nsamples / 2:
                    par_trans_below = float(line_split_below[1])
                    transitions[par_key] = (par_trans, par_trans_below)
                    if num_1_below - num_2_below > num_2_below:
                        trans_type[par_key] = 1
                    else:
                        trans_type[par_key] = 2
        index += 1
    return transitions, trans_type


def find_all_trans(path, eps, lda, tol_fixed_point, N, c, T, ndigits):
    fout_div = open(f'{path}/Lotka-Volterra_transition_div_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'w')
    fout_mult = open(f'{path}/Lotka-Volterra_transition_mult_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'w')
    
    fin = open(f'{path}/Lotka-Volterra_summary_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_tol_{tol_fixed_point}_N_{N}_c_{c}_T_{T}.txt', 'r')
    fin.readline()  # Skip header line

    lines = fin.readlines()

    fin.close()

    transitions_div = find_transition(lines, 3)
    transitions_multiple_eq, trans_type = find_identify_transition(lines, 4, 3)
    
    
    for mu in transitions_div:
        sigma_div, sigma_below_div = transitions_div[mu]
        fout_div.write(f"{mu:.{ndigits}f}\t{sigma_div:.{ndigits}f}\t{sigma_below_div:.{ndigits}f}\n")
     
    fout_div.close()


    for mu in transitions_multiple_eq:
        sigma, sigma_below = transitions_multiple_eq[mu]
        kind = trans_type[mu]
        fout_mult.write(f"{mu:.{ndigits}f}\t{sigma:.{ndigits}f}\t{sigma_below:.{ndigits}f}\t{kind}\n")
     
    fout_mult.close()



def main():
    eps = "0.0"
    lda = "1e-06"
    N_list = ["128", "256", "512", "1024", "2048", "4096"]
    c = "3.00"
    T = "0.0"
    tol_fixed_point = "1e-08"

    path = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"

    ndigits = 3

    for N in N_list:
        find_all_trans(path, eps, lda, tol_fixed_point, N, c, T, ndigits)

    return 0


if __name__ == '__main__':
    main()
