__author__ = 'david'

import numpy as np


def read_transitions(filein, temp_list_langevin, ndigits_T):
    fin = open(filein, 'r')
    fin.readline()  # Skip header line
    all_lines = fin.readlines()
    fin.close()
    temp_list_langevin = sorted(temp_list_langevin)
    transitions = {}
    for temp in temp_list_langevin:
        str_temp = "{0:.{1}f}".format(temp, ndigits_T)
        found = False
        for line in all_lines:
            line_split = line.split()
            if line_split[0] == str_temp:
                mu_min = float(line_split[1])
                mu_max = float(line_split[2])
                transitions[temp] = (mu_min, mu_max)
                found = True
                break
        if not found:
            print(f"No transition found for T = {temp} in file {filein}")
    return transitions



def find_mu_central(path_GMF, path_IBMF, T0, dT, Tf, temp_list_langevin, lda, av0, tol, 
                    max_iter, c, mu0, dmu, muf, ndigits_T):
    fileGMF = f'{path_GMF}/GMF_LV_sigma_0_transitions_T0_{T0}_dT_{dT}_Tf_{Tf}_lambda_{lda}_av0_{av0}_tol_{tol}_maxiter_{max_iter}_c_{c}_mu0_{"{0:.3f}".format(mu0)}_dmu_{"{0:.3f}".format(dmu)}_muf_{"{0:.3f}".format(muf)}.txt'
    fileIBMF = f'{path_IBMF}/IBMF_LV_sigma_0_transitions_T0_{T0}_dT_{dT}_Tf_{Tf}_lambda_{lda}_av0_{av0}_tol_{tol}_maxiter_{max_iter}_c_{c}_mu0_{"{0:.3f}".format(mu0)}_dmu_{"{0:.3f}".format(dmu)}_muf_{"{0:.3f}".format(muf)}.txt'
    transitions_GMF = read_transitions(fileGMF, temp_list_langevin, ndigits_T)
    transitions_IBMF = read_transitions(fileIBMF, temp_list_langevin, ndigits_T)
    mu_central = {}
    for i in range(len(temp_list_langevin)):
        T = temp_list_langevin[i]
        if T in transitions_GMF:
            mu_min_GMF, mu_max_GMF = transitions_GMF[T]
            mu_GMF = (mu_min_GMF + mu_max_GMF) / 2
            if T in transitions_IBMF:
                mu_min_IBMF, mu_max_IBMF = transitions_IBMF[T]
                mu_IBMF = (mu_min_IBMF + mu_max_IBMF) / 2
                mu_central[T] = (mu_GMF + mu_IBMF) / 2
            else:
                mu_central[T] = mu_GMF
        else:
            if T in transitions_IBMF:
                mu_min_IBMF, mu_max_IBMF = transitions_IBMF[T]
                mu_IBMF = (mu_min_IBMF + mu_max_IBMF) / 2
                mu_central[T] = mu_IBMF
            else:
                print(f"No transition found for T = {T} in both files.")
                continue
    return mu_central
        

        
def print_params(temp_list, mu_central, interval_len_langevin, dmu_langevin, path, filename):
    with open(f'{path}/{filename}', 'w') as fo:
        for T in temp_list:
            if T in mu_central:
                mu_c = mu_central[T]
                mu_left = mu_c - interval_len_langevin / 2
                mu_list = np.arange(mu_left, mu_left + interval_len_langevin, dmu_langevin)
                for mu in mu_list:
                    if mu >= 0:
                        fo.write(f"{T:.3f}\t{mu:.3f}\n")
            else:
                print(f"No central mu found for T = {T}")
    print(f"Parameters saved to {path}/{filename}")


def read_transitions_temps(path_in, file_transitions):
    fin = open(f'{path_in}/{file_transitions}', 'r')
    fin.readline()  # Skip header line
    all_lines = fin.readlines()
    fin.close()
    temp_list = []
    for line in all_lines:
        line_split = line.split()
        temp = line_split[0]
        temp_list.append(temp)
    return temp_list


def build_params_2(path_in, filein, file_transitions, path_out, fileout, max_mu, dmu):
    temp_transitions = read_transitions_temps(path_in, file_transitions)
    
    fin = open(f'{path_in}/{filein}', 'r')
    fin.readline()  # Skip header line
    all_lines = fin.readlines()
    fin.close()

    counter = 0
    fo = open(f'{path_out}/{fileout}', 'w')
    for line in all_lines:
        line_split = line.split()
        T = line_split[0]
        if T not in temp_transitions:
            min_mu = float(line_split[2]) + dmu
            for mu in np.arange(min_mu, max_mu + dmu / 2, dmu):
                fo.write(f"{T}\t{mu:.3f}\n")
                counter += 1
    fo.close()
    print(counter)


def main():
    # lda = "1e-6"
    # av0 = "0.08"
    # tol = "1e-4"
    # max_iter = "10000"
    # c = "3"

    # ndigits_T = 3

    # path_GMF = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/"
    # path_IBMF = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Results/GMF/"
    

    # T0 = 0.001
    # dT = 0.001
    # Tf = 0.1


    # mu0 = 0.010
    # dmu = 0.002
    # muf = 1.000

    # T0_langevin = 0.002
    # dT_langevin = 0.002
    # Tf_langevin = 0.0500

    # temp_list_langevin = np.arange(T0_langevin, Tf_langevin + dT_langevin / 2, dT_langevin)
    # temp_list_langevin = np.insert(temp_list_langevin, 0, T0)

    # mu_central = find_mu_central(path_GMF, path_IBMF, T0, dT, Tf, temp_list_langevin, lda, av0,
    #                              tol, max_iter, c, mu0, dmu, muf, ndigits_T)
    
    # path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Lecce/"
    # fileout = "params_Langevin_1.txt"

    # interval_len_langevin = 0.16
    # dmu_langevin = 0.002

    # print_params(temp_list_langevin, mu_central, interval_len_langevin, dmu_langevin, path_out, fileout)


    eps = "1.0"
    lda = "1e-06"
    N = "256"
    c = "3.00"
    h = "0.001"
    sigma = "0"

    path_in = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Langevin/Results/"

    file_transitions = f'Lotka-Volterra_transitions_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_h_{h}_N_{N}_c_{c}_sigma_{sigma}.txt'
    file_minmax = f'Lotka-Volterra_min_max_mu_epsilon_{eps}_Partially_AsymGauss_lambda_{lda}_h_{h}_N_{N}_c_{c}_sigma_{sigma}.txt'

    path_out = "/media/david/Data/UH/Grupo_de_investigacion/Ecology/Scripts/Dresden/"
    fileout = "params_Langevin_2.txt"
    max_mu = 0.352
    dmu = 0.002

    build_params_2(path_in, file_minmax, file_transitions, path_out, fileout, max_mu, dmu)

    return 0


if __name__ == '__main__':
    main()
