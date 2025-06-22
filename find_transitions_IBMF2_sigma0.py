__author__ = 'david'

import numpy as np



    

def find_transition(path, T, lda, av0, tol, tol_asymp, max_iter, c, mu0, dmu, muf, ndigits_T):
    mu = mu0
    ind_prev = 0
    trans = []
    mu_prev = mu0
    format_str = "{0:." + str(ndigits_T) + "f}"
    str_T = format_str.format(T)
    filein = f'{path}/AllData/PhaseDiagram/sigma0/IBMF2_around_m_LV_sigma_0_T_{str_T}_lambda_{lda}_av0_{av0}_tol_{tol}_tolasymp_{tol_asymp}_maxiter_{max_iter}_c_{c}_mu0_{"{0:.3f}".format(mu0)}_dmu_{"{0:.3f}".format(dmu)}_muf_{"{0:.3f}".format(muf)}.txt'
    fin = open(filein, 'r')
    fin.readline()
    while mu <= muf and len(trans) < 2:
        j = fin.readline()
        if not j:
            break
        line = j.split()
        if line[2] == 'diverges':
            if ind_prev == 1:
                trans.append([mu_prev, mu])
                ind_prev += 1
            elif ind_prev == 0:
                trans.append([mu_prev, mu])
                trans.append([mu_prev, mu])
                ind_prev += 2
            break
        elif line[2] == '0':
            if ind_prev == 0:
                trans.append([mu_prev, mu])
                ind_prev += 1
        mu_prev = mu
        mu += dmu
    return trans


def find_all_trans(path, T0, dT, Tf, lda, av0, tol, tol_asymp, max_iter, c, mu0, dmu, muf, 
                   ndigits_T):
    fileout = f'{path}/IBMF2_around_m_LV_sigma_0_transitions_T0_{T0}_dT_{dT}_Tf_{Tf}_lambda_{lda}_av0_{av0}_tol_{tol}_tolasymp_{tol_asymp}_maxiter_{max_iter}_c_{c}_mu0_{"{0:.3f}".format(mu0)}_dmu_{"{0:.3f}".format(dmu)}_muf_{"{0:.3f}".format(muf)}.txt'
    fo = open(fileout, 'w')
    fo.write("#T\ttrans_1\ttrans_2\n")
    format_str = "{0:." + str(ndigits_T) + "f}"
    for T in np.arange(T0, Tf + dT / 2, dT):
        trans = find_transition(path, T, lda, av0, tol, tol_asymp, max_iter, c, mu0, dmu, muf, ndigits_T)
        str_T = format_str.format(T)
        fo.write(str_T)
        for i in range(len(trans)):
            mu_min, mu_max = trans[i]
            fo.write("\t" + str("{0:.3f}".format(mu_min)) + "\t" + str("{0:.3f}".format(mu_max)))
        if len(trans) == 0:
            print("No transitions found for T = " + str("{0:.3f}".format(T)), "mu0 = " + str("{0:.3f}".format(mu0)), "dmu = " + str("{0:.3f}".format(dmu)), "muf = " + str("{0:.3f}".format(muf)))
        else:
            fo.write("\n")
    fo.close()


def main():
    lda = "0.01"
    av0 = "0.08"
    tol = "1e-4"
    tol_asymp = "1e-6"
    max_iter = "10000"
    c = "3"

    path = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/IBMF2/"
    
    # T0 = 0.01
    # dT = 0.01
    # Tf = 1.0

    T0 = 0.001
    dT = 0.001
    Tf = 0.01

    ndigits_T = 3

    mu0 = 0.11
    dmu = 0.002
    muf = 2.0

    find_all_trans(path, T0, dT, Tf, lda, av0, tol, tol_asymp, max_iter, c, mu0, dmu, muf, ndigits_T)
    return 0


if __name__ == '__main__':
    main()
