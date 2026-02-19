import os
import numpy as np
import fnmatch


def read_file(path, filein):
    try:
        fin = open(f'{path}/{filein}', "r")
    except (IOError, OSError):
        return []
    
    lines = fin.readlines()
    fin.close()
    
    return lines


def filter_files(path, pattern, pos_N=28, pos_eps=22, pos_mu=24, pos_sigma=26):
    files = []

    # Iterate through the files in the specified directory
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            # Extract the unique strings from the filename
            parts = filename.split('_') 
            files.append((filename, int(parts[pos_N]), float(parts[pos_eps]), float(parts[pos_mu]), float(parts[pos_sigma])))
    sorted_data = sorted(files, key=lambda x: (x[1], x[2], x[3], x[4]))  # Sort by mu value
    return sorted_data


def get_all_files(path, pattern_beginning, seed0, ngr_batch, ngr_total, pos_N=28, pos_eps=22, pos_mu=24, pos_sigma=26):
    files = []
    for seed in range(seed0, ngr_total, ngr_batch):
        pattern = f'{pattern_beginning}_seedblock_{seed}_nsampl_{ngr_batch}.txt'
        files_seed0 = filter_files(path, pattern, pos_N, pos_eps, pos_mu, pos_sigma)
        files.extend(files_seed0)
    return files


def group_files_by_params(files):
    grouped_files = {}
    for file_info in files:
        filename, N, eps, mu, sigma = file_info
        key = (N, eps, mu, sigma)
        if key not in grouped_files:
            grouped_files[key] = []
        grouped_files[key].append(filename)
    return grouped_files


def read_previous_data(file_prev_data):
    try:
        fin = open(file_prev_data, "r")
    except (IOError, OSError):
        return []
    
    print(f'Reading previous data from {file_prev_data}')
    lines = fin.readlines()
    fin.close()
    
    return lines


def gather(path_to_files, path_out, path_previous_data, T, lda, avn0, dn, ninitconds, tol, 
           max_iter, c, damping, nseq, seed0, ngr_batch, ngr_total, ndigits=3):
    pattern_beginning = f'IBMF_seq_RRG_T_{T}_lambda_{lda}_PD_Lotka_Volterra_final_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_*_mu_*_sigma_*_N_*_c_{c}_damping_{damping}_nseq_{nseq}'    
    files = get_all_files(path_to_files, pattern_beginning, seed0, ngr_batch, ngr_total)
    grouped_files = group_files_by_params(files)
    for key, file_list in grouped_files.items():
        N, eps, mu, sigma = key
        fout = open(f'{path_out}/IBMF_seq_RRG_T_{T}_lambda_{lda}_PD_Lotka_Volterra_final_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_{eps:.{ndigits}f}_mu_{mu:.{ndigits}f}_sigma_{sigma:.{ndigits}f}_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt', "w")
        file_prev_data = f'{path_previous_data}/IBMF_seq_RRG_T_{T}_lambda_{lda}_PD_Lotka_Volterra_final_av0_{avn0}_dn_{dn}_ninitconds_{ninitconds}_tol_{tol}_maxiter_{max_iter}_eps_{eps:.{ndigits}f}_mu_{mu:.{ndigits}f}_sigma_{sigma:.{ndigits}f}_N_{N}_c_{c}_damping_{damping}_nseq_{nseq}.txt'
        lines_prev = read_previous_data(file_prev_data)
        if len(lines_prev) > 0:
            for line in lines_prev:
                fout.write(line)
        ngraphs_prev = len(lines_prev)
        lines_to_print = []
        for filein in file_list:
            lines = read_file(path_to_files, filein)
            lines_to_print.extend(lines)
        ngraphs = len(lines_to_print) + ngraphs_prev
            
        for line in lines_to_print:
            fout.write(line)
        print(f'Wrote file for N={N}, eps={eps}, mu={mu}, sigma={sigma}, ngraphs={ngraphs}')
        fout.close()


def gather_all_ngrbatch(path_to_files, path_out, path_previous_data, T, lda, avn0, dn, ninitconds, tol, max_iter, c, damping, nseq, seed0, ngr_batch_list, ngr_total):
    for ngr_batch in ngr_batch_list:
        print(f'Gathering with ngr_batch={ngr_batch}')
        gather(path_to_files, path_out, path_previous_data, T, lda, avn0, dn, ninitconds, tol, max_iter, c, damping, nseq, seed0, ngr_batch, ngr_total)


def main():
    T = "0.000"
    lda = "0.000"
    avn0 = "0.5"
    dn = "0.5"
    ninitconds = "10"
    tol = "1e-6"
    max_iter = "10000"
    c = "3"
    damping = "0.2"
    nseq = "1"

    path_to_files = "./"
    path_out = './ToDownload'
    path_previous_data = '../../../Saved/IBMF/PhaseDiagram/'

    seed0 = 1
    ngr_total = 10000
    ngr_batch_list = [10000, 5000, 2000, 1000, 500, 200, 100]

    gather_all_ngrbatch(path_to_files, path_out, path_previous_data, T, lda, avn0, dn, ninitconds, tol, max_iter, c, damping, nseq, seed0, ngr_batch_list, ngr_total)
      
    return 0


if __name__ == '__main__':
    main()