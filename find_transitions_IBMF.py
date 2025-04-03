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


def get_counts(filein, ngraphs):
    count_single = 0
    count_multiple = 0
    try:
        fin = open(filein, 'r')
        while True:
            j = fin.readline()
            if not j:
                break
            elif j[0] != '#':
                line = j.split()
                if is_number(line[2]):
                    if line[1] == '1':
                        count_single += 1
                    elif line[1] == '0':
                        count_multiple += 1
        fin.close()
        count_div = ngraphs - count_single - count_multiple
        return [count_single, count_multiple, count_div], True
    except (OSError, IOError):
        return [0, 0, 0], False
    

def find_transition(path, str_file_1, sigma0, dsigma, sigmaf, ngraphs):
    sigma = sigma0
    ind_max_prev = 0
    trans = []
    sigma_prev = sigma0
    while sigma <= sigmaf and len(trans) < 2:
        filein = path + "/AllData/nsamples_" + str(ngraphs) + "/" + str_file_1 + \
                 str("{0:.2f}".format(sigma)) + '.txt'
        count_list, found = get_counts(filein, ngraphs)
        if found:
            ind_max = count_list.index(max(count_list))
            if ind_max == 1 and ind_max_prev == 0 or ind_max == 2 and ind_max_prev == 1:
                trans.append((sigma + sigma_prev) / 2)
                ind_max_prev += 1
            elif ind_max == 2 and ind_max_prev == 0:
                trans.append((sigma + sigma_prev) / 2)
                trans.append((sigma + sigma_prev) / 2)
                ind_max_prev += 2
            sigma_prev = sigma
        sigma += dsigma
    return trans


def find_all_trans(lda, av0, tol, maxiter, path, pars_list, sigma0, dsigma, sigmaf, ngraphs,
                   str_graph):
    fileout = path + "/" + "IBMF_Lotka_Volterra_ss_transitions_" + str_graph + "_lambda_" + str(lda) + \
              "_av0_" + str(av0) + "_tol_" + tol + "_maxiter_" + str(maxiter) + "txt"
    fo = open(fileout, 'w')
    fo.write("#T\teps\tmu\ttrans_1\ttrans_2\n")
    for T, eps, mu in pars_list:
        str_file_1 = "IBMF_Lotka_Volterra_steady_state_" + str_graph + "_T_" + str("{0:.2f}".format(T)) + \
                     "_lambda_" + str(lda) + "_av0_" + str(av0) + "_tol_" + tol + \
                     "_maxiter_" + str(maxiter) + "_eps_" + str("{0:.2f}".format(eps)) + \
                     "_mu_" + str("{0:.2f}".format(mu)) + "_sigma_"
        trans = find_transition(path, str_file_1, sigma0, dsigma, sigmaf, ngraphs)
        if len(trans) == 1:
            fo.write(str(int(T * 100)) + "\t" + str(int(eps * 100)) + "\t" \
                     + str(int(mu * 100)) + "\t" + str(trans[0]) + "\n")
        elif len(trans) == 2:
            fo.write(str(int(T * 100)) + "\t" + str(int(eps * 100)) + "\t" \
                     + str(int(mu * 100)) + "\t" + str(trans[0]) + "\t" \
                     + str(trans[1]) + "\n")
        else:
            print("No transitions found for T = " + str(T) + ", eps = " + str(eps) + ", mu = " + str(mu))
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
    sched_mu = {"T":0.5, "eps":0.0, "mu":[0.05, 0.05, 0.5]}
    sched_T = {"T":[0.05, 0.05, 1.20], "eps":0.0, "mu":0.0}
    lda = 0.01
    av0 = 2
    tol = "1e-6"
    maxiter = 1000
    ngraphs = 1000
    str_graph = "gr_inside_RRG"

    sigma_0 = 0.00
    dsigma = 0.01
    sigma_f = 0.80

    path = "../Results/IBMF"
    
    pars_list = create_pars_list([sched_eps, sched_mu, sched_T])

    find_all_trans(lda, av0, tol, maxiter, path, pars_list,
                   sigma_0, dsigma, sigma_f, ngraphs, str_graph)

    return 0


if __name__ == '__main__':
    main()
