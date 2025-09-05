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



def find_all_trans(path, eps, N, c, avn0, tol, max_iter, ndigits, damping, nseq):
    fout = open(f'{path}/IBMF_T0_seq_RRG_PD_Lotka_Volterra_transitions_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt', 'w')
    
    fin = open(f'{path}/IBMF_T0_seq_RRG_PD_Lotka_Volterra_summary_av0_{avn0}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt', 'r')
    fin.readline()  # Skip header line

    lines = fin.readlines()

    transitions_m = find_transition(lines, 4)
    transitions_multiple_eq = find_transition(lines, 5)
    transitions_deaths = find_transition(lines, 6)
    
    
    keys_list = [transitions_m.keys(), transitions_multiple_eq.keys(), transitions_deaths.keys()]
    mu_list = sorted(set().union(*keys_list))
    for mu in mu_list:
        if mu not in transitions_m:
            transitions_m[mu] = (0, 0)
        if mu not in transitions_multiple_eq:
            transitions_multiple_eq[mu] = (0, 0)
        if mu not in transitions_deaths:
            transitions_deaths[mu] = (0, 0)
        sigma_m, sigma_below_m = transitions_m[mu]
        sigma_multiple_eq, sigma_below_multiple_eq = transitions_multiple_eq[mu]
        sigma_deaths, sigma_below_deaths = transitions_deaths[mu]
        fout.write(f"{mu:.{ndigits}f}\t{sigma_m:.{ndigits}f}\t{sigma_below_m:.{ndigits}f}\t{sigma_multiple_eq:.{ndigits}f}\t{sigma_below_multiple_eq:.{ndigits}f}\t{sigma_deaths:.{ndigits}f}\t{sigma_below_deaths:.{ndigits}f}\n")
    
    fin.close()
    fout.close()



def main():
    eps = "0.000"
    avn0 = "0.08"
    tol = "1e-6"
    max_iter = "10000"
    N = "1024"
    c = "3"
    damping = "0.2"
    nseq = "10"

    path = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF/"

    ndigits = 3

    find_all_trans(path, eps, N, c, avn0, tol, max_iter, ndigits, damping, nseq)

    return 0


if __name__ == '__main__':
    main()
