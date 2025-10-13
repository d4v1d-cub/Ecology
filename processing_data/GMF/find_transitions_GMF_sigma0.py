__author__ = 'david'

import numpy as np


def find_specific_transition(all_lines, position, safe, T):
    index = 1
    while index < len(all_lines) - 1:
        mu = float(all_lines[index].split()[0])
        mu_prev = float(all_lines[index - 1].split()[0])
        line1 = all_lines[index].split()
        if line1[position] == '0':
            safe_counter = 0
            line1 = all_lines[index + safe_counter].split()
            while safe_counter < safe and line1[position] == '0':
                safe_counter += 1
                if index + safe_counter >= len(all_lines):
                    break
                line1 = all_lines[index + safe_counter].split()
            if index + safe_counter >= len(all_lines):
                break
            else:
                if safe_counter == safe:
                    return [mu_prev, mu], True
                index += safe_counter + 1
        else:
            index += 1
    return [], False
    

def find_transition(path, T, lda, av0, tol, max_iter, c, mu0, dmu, muf, ndigits_T):
    format_str = "{0:." + str(ndigits_T) + "f}"
    str_T = format_str.format(T)
    filein = f'{path}/AllData/PhaseDiagram/sigma0/GMF_LV_sigma_0_T_{str_T}_lambda_{lda}_av0_{av0}_tol_{tol}_maxiter_{max_iter}_c_{c}_mu0_{"{0:.3f}".format(mu0)}_dmu_{"{0:.3f}".format(dmu)}_muf_{"{0:.3f}".format(muf)}.txt'
    fin = open(filein, 'r')
    fin.readline()  # Skip header line
    all_lines = fin.readlines()
    safe = 5
    
    trans_m, found_trans_m = find_specific_transition(all_lines, 2, safe, T)
    trans_Q, found_trans_Q = find_specific_transition(all_lines, 3, safe, T)
    return trans_m, trans_Q, found_trans_m, found_trans_Q


def find_all_trans(path, T0, dT, Tf, lda, av0, tol, max_iter, c, mu0, dmu, muf, 
                   ndigits_T):
    fileout = f'{path}/GMF_LV_sigma_0_transitions_T0_{T0}_dT_{dT}_Tf_{Tf}_lambda_{lda}_av0_{av0}_tol_{tol}_maxiter_{max_iter}_c_{c}_mu0_{"{0:.3f}".format(mu0)}_dmu_{"{0:.3f}".format(dmu)}_muf_{"{0:.3f}".format(muf)}.txt'
    fo = open(fileout, 'w')
    fo.write("#T\ttrans_1\ttrans_2\n")
    format_str = "{0:." + str(ndigits_T) + "f}"
    for T in np.arange(T0, Tf + dT / 2, dT):
        trans_m, trans_Q, found_trans_m, found_trans_Q = \
            find_transition(path, T, lda, av0, tol, max_iter, c, mu0, dmu, muf, ndigits_T)
        str_T = format_str.format(T)
        fo.write(str_T)
        
        if found_trans_m and found_trans_Q:
            fo.write("\t" + str("{0:.3f}".format(trans_m[0])) + "\t" + str("{0:.3f}".format(trans_m[1])))
            fo.write("\t" + str("{0:.3f}".format(trans_Q[0])) + "\t" + str("{0:.3f}".format(trans_Q[1])))
        elif found_trans_m:
            fo.write("\t" + str("{0:.3f}".format(trans_m[0])) + "\t" + str("{0:.3f}".format(trans_m[1])))
        else:
            print("No transitions found for T = " + str("{0:.3f}".format(T)), "mu0 = " + str("{0:.3f}".format(mu0)), "dmu = " + str("{0:.3f}".format(dmu)), "muf = " + str("{0:.3f}".format(muf)))
        fo.write("\n")
    fo.close()


def main():
    lda = "1e-6"
    av0 = "0.08"
    tol = "1e-4"
    max_iter = "100000"
    c = "3"

    path = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/"
    

    T0 = 0.001
    dT = 0.001
    Tf = 0.1

    ndigits_T = 3

    mu0 = 0.010
    dmu = 0.002
    muf = 1.000

    find_all_trans(path, T0, dT, Tf, lda, av0, tol, max_iter, c, mu0, dmu, muf, ndigits_T)
    return 0


if __name__ == '__main__':
    main()
