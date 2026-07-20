import os
import fnmatch
import statistics


def read_file(path):
    try:
        fin = open(path, "r")
    except (IOError, OSError):
        return []

    lines = fin.readlines()
    fin.close()

    return lines


def count_fixed_points(path):
    lines = read_file(path)
    count = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        count += 1
    return count


def filter_files(path, pattern, pos_mu, pos_N, pos_ninitcond, pos_seedgraph):
    files = []

    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            parts = filename.split('_')
            mu = float(parts[pos_mu])
            N = int(parts[pos_N])
            ninitcond = int(parts[pos_ninitcond])
            seedgraph = int(parts[pos_seedgraph])
            nfp = count_fixed_points(os.path.join(path, filename))
            files.append((mu, N, ninitcond, seedgraph, nfp))

    return files


def gather_files(base_path, pattern, pos_mu=9, pos_N=13, pos_ninitcond=26, pos_seedgraph=17):
    results = []
    for entry in sorted(os.listdir(base_path)):
        subdir = os.path.join(base_path, entry)
        if not os.path.isdir(subdir):
            continue
        results.extend(filter_files(subdir, pattern, pos_mu, pos_N, pos_ninitcond, pos_seedgraph))

    return results


def average_over_seedgraphs(results):
    grouped = {}
    for mu, N, ninitcond, seedgraph, nfp in results:
        key = (mu, N, ninitcond)
        grouped.setdefault(key, []).append(nfp)

    averaged = []
    for (mu, N, ninitcond), values in grouped.items():
        ngraphs = len(values)
        mean_nfp = statistics.fsum(values) / ngraphs
        if ngraphs > 1:
            sem_nfp = statistics.stdev(values) / (ngraphs ** 0.5)
        else:
            sem_nfp = 0.0
        averaged.append((mu, N, ninitcond, mean_nfp, sem_nfp, ngraphs))

    sorted_averaged = sorted(averaged, key=lambda x: (x[0], x[1], x[2]))
    return sorted_averaged


def write_summary(path_out, filename_out, results):
    fout = open(f'{path_out}/{filename_out}', "w")
    fout.write('# mu N ninitcond mean_number_of_fixed_points sem_number_of_fixed_points ngraphs\n')
    for mu, N, ninitcond, mean_nfp, sem_nfp, ngraphs in results:
        fout.write(f'{mu:.3f} {N} {ninitcond} {mean_nfp:.6f} {sem_nfp:.6f} {ngraphs}\n')
    fout.close()
    print(f'Wrote summary file {filename_out} with {len(results)} rows')


def main():
    T = "0"
    eps = "0.000"
    sigma = "0.000"
    c = "3"
    avn0 = "0.500"
    dn = "0.500"
    tol = "1.0e-06"
    max_iter = "10000"
    damping = "1.00"

    base_path = "./"
    path_out = "./ToDownload"

    pattern = (f'IBMF_T{T}_seq_gr_inside_RRG_eps_{eps}_mu_*_sigma_{sigma}_N_*_c_{c}'
               f'_seedgraph_*_Lotka_Volterra_final_av0_{avn0}_dn_{dn}'
               f'_ninitcond_*_tol_{tol}_maxiter_{max_iter}_damping_{damping}_summary.txt')

    filename_out = (f'IBMF_T{T}_seq_gr_inside_RRG_eps_{eps}_sigma_{sigma}_c_{c}'
                     f'_Lotka_Volterra_final_av0_{avn0}_dn_{dn}'
                     f'_tol_{tol}_maxiter_{max_iter}_damping_{damping}_nfixedpoints_avg.txt')

    results = gather_files(base_path, pattern)
    averaged_results = average_over_seedgraphs(results)
    write_summary(path_out, filename_out, averaged_results)

    return 0


if __name__ == '__main__':
    main()
