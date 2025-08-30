__author__ = 'david'

import numpy as np


def is_number(s):
    try:
        float(s)
        if (s == 'nan') or (s == 'inf') or (s == '-inf'):
            return False
        return True
    except ValueError:
        return False


def get_counts(filein):
    count_single = 0
    count_multiple = 0
    count_total = 0
    try:
        fin = open(filein, 'r')
        while True:
            j = fin.readline()
            if not j:
                break
            elif j[0] != '#':
                line = j.split()
                if len(line) > 3:
                    if is_number(line[2]):
                        if line[1] == '1':
                            count_single += 1
                        elif line[1] == '0':
                            count_multiple += 1
                    count_total += 1
        fin.close()
        count_div = count_total - count_single - count_multiple
        return [count_single, count_multiple, count_div], True
    except (OSError, IOError):
        return [0, 0, 0], False
    

def count_lines_file(filein):
    try:
        fin = open(filein, 'r')
        count = 0
        while True:
            j = fin.readline()
            if not j:
                break
            elif j[0] != '#':
                count += 1
        fin.close()
        return count
    except (OSError, IOError):
        return 0



def find_transition(path, str_file_1, str_file_2, sigma0, dsigma, sigmaf, ngraphs_attempt):
    sigma = sigma0
    ind_max_prev = 0
    trans = []
    sigma_prev = sigma0
    ngraphs = 0
    ngraphs_prev = 0
    ngraphs_trans = []
    while sigma <= sigmaf and len(trans) < 2:
        filein = path + "/AllData/PhaseDiagram/nsamples_" + str(ngraphs_attempt) + "/" + str_file_1 + \
                 str("{0:.2f}".format(sigma)) + str_file_2
        ngraphs = count_lines_file(filein)
        if ngraphs > 0:
            count_list, found = get_counts(filein)
            if found:
                ind_max = count_list.index(max(count_list))
                if ind_max == 1 and ind_max_prev == 0 or ind_max == 2 and ind_max_prev == 1:
                    trans.append([sigma_prev, sigma])
                    ngraphs_trans.append([ngraphs_prev, ngraphs])
                    ind_max_prev += 1
                elif ind_max == 2 and ind_max_prev == 0:
                    trans.append([sigma_prev, sigma])
                    ngraphs_trans.append([ngraphs_prev, ngraphs])
                    trans.append([sigma_prev, sigma])
                    ngraphs_trans.append([ngraphs_prev, ngraphs])
                    ind_max_prev += 2
                sigma_prev = sigma
                ngraphs_prev = ngraphs
        sigma += dsigma
    return trans, ngraphs_trans


def find_all_trans(lda, av0, tol, maxiter, path, pars_list, sigma0, dsigma, sigmaf, ngraphs_attempt,
                   str_graph, nmin, nmax, npoints, N, c):
    fileout = path + "/" + "PBMF_Lotka_Volterra_transitions_" + str_graph + "_lambda_" + lda + \
              "_av0_" + av0 + "_tol_" + tol + "_maxiter_" + maxiter + \
              "_ngr_att_" + str(ngraphs_attempt) + ".txt"
    fo = open(fileout, 'w')
    fo.write("#T\teps\tmu\tsigma_min_1\tsigma_max_1\tsigma_min_2\tsigma_max_2\n")
    for T, eps, mu in pars_list:
        str_file_1 = "PBMF_PD_Lotka_Volterra_final_T_" + str("{0:.2f}".format(T)) + \
                     "_lambda_" + lda + "_av0_" + av0 + "_tol_" + tol + \
                     "_maxiter_" + maxiter + "_eps_" + str("{0:.2f}".format(eps)) + \
                     "_mu_" + str("{0:.2f}".format(mu)) + "_sigma_"
        str_file_2 = "_nmin_" + nmin + "_nmax_" + nmax + "_npoints_" + npoints + "_N_" + N + "_c_" + c + ".txt"
        trans, ngraph_trans = find_transition(path, str_file_1, str_file_2, sigma0, dsigma, sigmaf, 
                                             ngraphs_attempt)
        fo.write(str(int(T * 100)) + "\t" + str(int(eps * 100)) + "\t" \
                     + str(int(mu * 100)))
        for i in range(len(trans)):
            smin, smax = trans[i]
            ngrmin, ngrmax = ngraph_trans[i]
            fo.write("\t" + str(smin) + "\t" + str(smax))
            fo.write("\t" + str(ngrmin) + "\t" + str(ngrmax))
        if len(trans) == 0:
            print("No transitions found for T = " + str(T) + ", eps = " + str(eps) + ", mu = " + str(mu))
        else:
            fo.write("\n")
    fo.close()


def parse_arg(sched_val):
    if isinstance(sched_val, float) or isinstance(sched_val, int):
        return np.round(np.array([sched_val]), 2)
    elif isinstance(sched_val, list):
        if isinstance(sched_val[0], list):
            return np.round(np.array(sched_val[0]), 2)
        elif is_number(sched_val[0]):
            return np.round(np.arange(sched_val[0], sched_val[2] + sched_val[1] / 2, sched_val[1]), 2)
        else:
            print("Error: Invalid schedule value")
    else:
        print("Error: Invalid schedule value")
    

def create_pars_list(sched_list):
    pars_list = []
    for sched in sched_list:
        vals_T = parse_arg(sched["T"])
        vals_eps = parse_arg(sched["eps"])
        vals_mu = parse_arg(sched["mu"])
        for T in vals_T:
            for eps in vals_eps:
                for mu in vals_mu:
                    if [T, eps, mu] not in pars_list:
                        pars_list.append([T, eps, mu])
    pars_list = sorted(pars_list, key=lambda x: (x[0], x[1], x[2]))
    return pars_list


def main():
    sched_eps = {"T":0.5, "eps":[0, 0.1, 1], "mu":0.0}
    sched_mu = {"T":0.5, "eps":1.0, "mu":[0.05, 0.05, 0.5]}
    lda = "0.01"
    av0 = "0.9"
    tol = "1e-4"
    maxiter = "1000"
    ngraphs_attempt = 100
    str_graph = "gr_inside_RRG"

    nmin = "0.001"
    nmax = "5"
    npoints = "500"
    N = "512"
    c = "3"


    sigma_0 = 0.26
    dsigma = 0.01
    sigma_f = 0.68

    path = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/PBMF"
    
    pars_list = create_pars_list([sched_eps, sched_mu])

    find_all_trans(lda, av0, tol, maxiter, path, pars_list,
                   sigma_0, dsigma, sigma_f, ngraphs_attempt, str_graph, nmin, nmax, npoints, N, c)

    return 0


if __name__ == '__main__':
    main()
