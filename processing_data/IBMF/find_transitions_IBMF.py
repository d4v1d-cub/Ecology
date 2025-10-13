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
    

def get_id_list(filein):
    id_list = []
    try:
        fin = open(filein, 'r')
        while True:
            j = fin.readline()
            if not j:
                break
            line = j.split()
            id_list.append(int(line[2]))
        fin.close()
        return id_list, True
    except (OSError, IOError):
        return [], False


def get_counts(filein, id_list):
    count_single = 0
    count_multiple = 0
    count_div_catched = 0
    try:
        fin = open(filein, 'r')
        while True:
            j = fin.readline()
            if not j:
                break
            elif j[0] != '#':
                line = j.split()
                if int(line[4]) in id_list:
                    if line[1] == 'diverges':
                        count_div_catched += 1
                    elif is_number(line[2]):
                        if line[1] == '1':
                            count_single += 1
                        elif line[1] == '0':
                            count_multiple += 1
                    else:
                        count_div_catched += 1
        count_div_not_catched = len(id_list) - count_div_catched - count_single - count_multiple 
        fin.close()
        return [count_single, count_multiple, count_div_catched, count_div_not_catched], True
    except (OSError, IOError):
        return [0, 0, 0, 0], False
    

def find_transition(path, str_file_1, str_file_2, str_donefile_1, str_donefile_2, 
                    sigma0, dsigma, sigmaf, ngraphs_total):
    sigma = sigma0
    ind_max_prev = 0
    trans = []
    sigma_prev = sigma0
    while sigma <= sigmaf and len(trans) < 2:
        filein = path + "/AllData/PhaseDiagram/nsamples_" + str(ngraphs_total) + "/" + str_file_1 + \
                 str("{0:.2f}".format(sigma)) + str_file_2 + '.txt'
        donefile = path + "/AllData/PhaseDiagram/nsamples_" + str(ngraphs_total) + "/done/" + str_donefile_1 + \
                   str("{0:.2f}".format(sigma)) + str_donefile_2 + '.txt'
        id_list, found_done = get_id_list(donefile)
        if found_done:
            count_list, found = get_counts(filein, id_list)
            if found:
                ind_max = count_list.index(max(count_list))
                if ind_max == 1 and ind_max_prev == 0 or ind_max == 2 and ind_max_prev == 1:
                    trans.append([sigma_prev, sigma])
                    ind_max_prev += 1
                elif ind_max == 2 and ind_max_prev == 0:
                    trans.append([sigma_prev, sigma])
                    trans.append([sigma_prev, sigma])
                    ind_max_prev += 2
                sigma_prev = sigma
        sigma += dsigma
    return trans


def find_all_trans(lda, av0, tol, maxiter, path, pars_list, sigma0, dsigma, sigmaf,
                   str_graph, ngraphs_total, N, c):
    fileout = path + "/" + "IBMF_Lotka_Volterra_transitions_" + str_graph + "_lambda_" + str(lda) + \
              "_av0_" + str(av0) + "_tol_" + tol + "_maxiter_" + str(maxiter) + "_nsamples_" + str(ngraphs_total) \
              + "_N_" + str(N) + "_c_" + str(c) + ".txt"
    fo = open(fileout, 'w')
    fo.write("#T\teps\tmu\ttrans_1\ttrans_2\n")
    for T, eps, mu in pars_list:
        str_file_1 = "IBMF_PD_Lotka_Volterra_final_T_" + str("{0:.2f}".format(T)) + \
                     "_lambda_" + str(lda) + "_av0_" + str(av0) + "_tol_" + tol + \
                     "_maxiter_" + str(maxiter) + "_eps_" + str("{0:.2f}".format(eps)) + \
                     "_mu_" + str("{0:.2f}".format(mu)) + "_sigma_"
        str_file_2 = "_N_" + str(N) + "_c_" + str(c)
        str_donefile_1 = "Done_IBMF_PD_T_" + str("{0:.2f}".format(T)) + "_eps_" + str("{0:.2f}".format(eps)) + \
                         "_mu_" + str("{0:.2f}".format(mu)) + "_sigma_"
        str_donefile_2 = "_avn0_" + str(av0)
        trans = find_transition(path, str_file_1, str_file_2, str_donefile_1, str_donefile_2, sigma0, dsigma, sigmaf, ngraphs_total)
        fo.write(str(int(T * 100)) + "\t" + str(int(eps * 100)) + "\t" \
                     + str(int(mu * 100)))
        for i in range(len(trans)):
            smin, smax = trans[i]
            fo.write("\t" + str(smin) + "\t" + str(smax))
        if len(trans) == 0:
            print("No transitions found for T = " + str(T) + ", eps = " + str(eps) + ", mu = " + str(mu))
        else:
            fo.write("\n")
            print("Transitions found for T = " + str(T) + ", eps = " + str(eps) + ", mu = " + str(mu))
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
    sched_eps = {"T":0.5, "eps":[0.00, 0.05, 1], "mu":0.0}
    # sched_mu = {"T":0.5, "eps":0.0, "mu":[0.05, 0.05, 0.5]}
    # sched_T = {"T":[0.05, 0.05, 1.20], "eps":0.0, "mu":0.0}
    lda = 0.01
    av0_list = [2.0, 0.08]
    tol = "1e-4"
    maxiter = 10000
    ngraphs_total = 10000
    str_graph = "gr_inside_RRG"

    N = 1000
    c = 3

    sigma_0 = 0.01
    dsigma = 0.01
    sigma_f = 0.90

    path = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF"
    
    # pars_list = create_pars_list([sched_eps, sched_mu, sched_T])
    pars_list = create_pars_list([sched_eps])

    for av0 in av0_list:
        find_all_trans(lda, av0, tol, maxiter, path, pars_list,
                       sigma_0, dsigma, sigma_f, str_graph, ngraphs_total, N, c)

    return 0


if __name__ == '__main__':
    main()
