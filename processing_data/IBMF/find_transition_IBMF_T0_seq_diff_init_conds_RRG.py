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
        
        line_split_below = lines[index + 1].split()
        key_below = float(line_split_below[0])
        if key_below == par_key:
            if not par_key in transition_found:
                nsamples_below = int(line_split_below[-1])
                num_below = int(line_split_below[position])
                if num_below >= nsamples_below / 2 and num < nsamples / 2:
                    transition_found[par_key] = True
                    par_trans_below = float(line_split_below[1])
                    transitions[par_key] = (par_trans, par_trans_below)
        index += 1
    return transitions


def find_identify_transition(lines, position_1, position_2):
    index = 0
    transitions = {}
    transition_found = {}
    trans_type = {}
    while index < len(lines) - 1:
        line_split = lines[index].split()
        par_key = float(line_split[0])
        par_trans = float(line_split[1])

        num_1 = int(line_split[position_1])
        nsamples = int(line_split[-1])

        
        line_split_below = lines[index + 1].split()
        key_below = float(line_split_below[0])
        if key_below == par_key:
            if not par_key in transition_found:
                nsamples_below = int(line_split_below[-1])
                num_1_below = int(line_split_below[position_1])
                num_2_below = int(line_split_below[position_2])
                if num_1_below >= nsamples_below / 2 and num_1 < nsamples / 2:
                    transition_found[par_key] = True
                    par_trans_below = float(line_split_below[1])
                    transitions[par_key] = (par_trans, par_trans_below)
                    if num_1_below - num_2_below > num_2_below:
                        trans_type[par_key] = 1
                    else:
                        trans_type[par_key] = 2
        index += 1
    return transitions, trans_type


def find_all_trans(path, T, lda, eps, N, c, avn0, dn, ninitconds, tol, max_iter, damping, nseq, ndigits):
    fout_div = open(f'{path}/IBMF_seq_RRG_T_{T}_lambda_{lda}_PD_Lotka_Volterra_transitions_div_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt', 'w')
    fout_mult = open(f'{path}/IBMF_seq_RRG_T_{T}_lambda_{lda}_PD_Lotka_Volterra_transitions_mult_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt', 'w')
    
    fin = open(f'{path}/IBMF_seq_RRG_T_{T}_lambda_{lda}_PD_Lotka_Volterra_summary_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt', 'r')
    fin.readline()  # Skip header line

    lines = fin.readlines()
    fin.close()

    transitions_m = find_transition(lines, 4)
    transitions_multiple_eq, trans_type = find_identify_transition(lines, 5, 4)
    
    for mu in transitions_m:
        sigma_m, sigma_below_m = transitions_m[mu]
        fout_div.write(f"{mu:.{ndigits}f}\t{sigma_m:.{ndigits}f}\t{sigma_below_m:.{ndigits}f}\n")
     
    fout_div.close()
    
    for mu in transitions_multiple_eq:
        sigma_multiple_eq, sigma_below_multiple_eq = transitions_multiple_eq[mu]
        kind = trans_type[mu]
        fout_mult.write(f"{mu:.{ndigits}f}\t{sigma_multiple_eq:.{ndigits}f}\t{sigma_below_multiple_eq:.{ndigits}f}\t{kind}\n")
    
    fout_mult.close()


def main():
    T = "0.000"
    lda = "0.000"
    eps = "0.000"
    avn0 = "0.5"
    dn = "0.5"
    ninitconds = "10"
    tol = "1e-6"
    max_iter = "10000"
    N_list = ["128", "256", "512", "1024", "2048", "4096"]
    c = "3"
    damping = "1.0"
    nseq = "1"

    path = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/"
    # path = "/media/david/Seagate Expansion Drive/Salva/Salva_Data_Investigacion/Grupo_de_investigacion/Ecology/Results/IBMF/"

    ndigits = 3

    for N in N_list:
        find_all_trans(path, T, lda, eps, N, c, avn0, dn, ninitconds, tol, max_iter, damping, nseq, ndigits)

    return 0


if __name__ == '__main__':
    main()
