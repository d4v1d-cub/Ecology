__author__ = 'david'


def find_transition(lines, nsamples, position):
    index = 0
    transitions = {}
    transition_found = {}
    while index < len(lines) - 1:
        line_split = lines[index].split()
        par_key = float(line_split[0])
        par_trans = float(line_split[1])

        num = int(line_split[position])

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



def find_all_trans(path, lda, eps, sigma, N, c, avn0, tol, max_iter, ndigits, nsamples, damping, nseq):
    fout = open(f'{path}/GMF_seq_RRG_PD_Lotka_Volterra_transitions_av0_{avn0}_lambda_{lda}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_sigma_{sigma}_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt', 'w')
    
    fin = open(f'{path}/GMF_seq_RRG_PD_Lotka_Volterra_summary_av0_{avn0}_lambda_{lda}_tol_{tol}_maxiter_{max_iter}_eps_{eps}_sigma_{sigma}_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt', 'r')
    fin.readline()  # Skip header line

    lines = fin.readlines()

    transitions_m = find_transition(lines, nsamples, 5)
    transitions_chi = find_transition(lines, nsamples, 6)
    transitions_multiple_eq = find_transition(lines, nsamples, 7)
    transitions_deaths = find_transition(lines, nsamples, 8)
    
    keys_list = [transitions_m.keys(), transitions_chi.keys(), transitions_multiple_eq.keys(), transitions_deaths.keys()]
    T_list = sorted(set().union(*keys_list))
    for T in T_list:
        if T not in transitions_m:
            transitions_m[T] = (0, 0)
        if T not in transitions_chi:
            transitions_chi[T] = (0, 0)
        if T not in transitions_multiple_eq:
            transitions_multiple_eq[T] = (0, 0)
        if T not in transitions_deaths:
            transitions_deaths[T] = (0, 0)
        mu_m, mu_below_m = transitions_m[T]
        mu_chi, mu_below_chi = transitions_chi[T]
        mu_multiple_eq, mu_below_multiple_eq = transitions_multiple_eq[T]
        mu_deaths, mu_below_deaths = transitions_deaths[T]
        fout.write(f"{T:.{ndigits}f}\t{mu_m:.{ndigits}f}\t{mu_below_m:.{ndigits}f}\t{mu_chi:.{ndigits}f}\t{mu_below_chi:.{ndigits}f}\t{mu_multiple_eq:.{ndigits}f}\t{mu_below_multiple_eq:.{ndigits}f}\t{mu_deaths:.{ndigits}f}\t{mu_below_deaths:.{ndigits}f}\n")
    
    fin.close()
    fout.close()



def main():
    eps = "1.000"
    sigma = "0.000"
    avn0 = "0.08"
    lda = "1e-6"
    tol = "1e-6"
    max_iter = "10000"
    N = "1024"
    c = "3"
    damping = "1.0"
    nseq = "10"

    path = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/"

    ndigits = 3
    nsamples = 1000

    find_all_trans(path, lda, eps, sigma, N, c, avn0, tol, max_iter, ndigits, nsamples, damping, nseq)

    return 0


if __name__ == '__main__':
    main()
