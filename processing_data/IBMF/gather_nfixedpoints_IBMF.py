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


def filter_files(path, pattern, source, pos_mu, pos_N, pos_ninitcond, pos_seedgraph):
    files = []

    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, pattern):
            parts = filename.split('_')
            mu = float(parts[pos_mu])
            N = int(parts[pos_N])
            ninitcond = int(parts[pos_ninitcond])
            seedgraph = int(parts[pos_seedgraph])
            nfp = count_fixed_points(os.path.join(path, filename))
            files.append((mu, N, ninitcond, seedgraph, nfp, source))

    return files


def gather_files(base_path, pattern, source, pos_mu=9, pos_N=13, pos_ninitcond=26, pos_seedgraph=17):
    results = []

    if not os.path.isdir(base_path):
        print(f'Warning: base path {base_path} does not exist, skipping {source}')
        return results

    for entry in sorted(os.listdir(base_path)):
        subdir = os.path.join(base_path, entry)
        if not os.path.isdir(subdir):
            continue
        results.extend(filter_files(subdir, pattern, source, pos_mu, pos_N, pos_ninitcond, pos_seedgraph))

    return results


def deduplicate_seedgraphs(results):
    # Keep a single sample per (mu, N, ninitcond, seedgraph); a seedgraph found
    # both in scratch and home (e.g. left behind by a prior, later-completed
    # run) is counted once, preferring the home copy as canonical.
    chosen = {}
    for mu, N, ninitcond, seedgraph, nfp, source in results:
        key = (mu, N, ninitcond, seedgraph)
        if key not in chosen or (chosen[key][5] != 'home' and source == 'home'):
            chosen[key] = (mu, N, ninitcond, seedgraph, nfp, source)
    return list(chosen.values())


def average_over_seedgraphs(results):
    grouped = {}
    for mu, N, ninitcond, _, nfp, source in results:
        key = (mu, N, ninitcond)
        grouped.setdefault(key, []).append((nfp, source))

    averaged = []
    for (mu, N, ninitcond), values in grouped.items():
        ngraphs = len(values)
        nfps = [nfp for nfp, _ in values]
        mean_nfp = statistics.fsum(nfps) / ngraphs
        if ngraphs > 1:
            sem_nfp = statistics.stdev(nfps) / (ngraphs ** 0.5)
        else:
            sem_nfp = 0.0
        n_home = sum(1 for _, source in values if source == 'home')
        n_scratch = sum(1 for _, source in values if source == 'scratch')
        averaged.append((mu, N, ninitcond, mean_nfp, sem_nfp, ngraphs, n_home, n_scratch))

    sorted_averaged = sorted(averaged, key=lambda x: (x[0], x[1], x[2]))
    return sorted_averaged


def print_sample_report(averaged):
    print('\nSamples found per (mu, N, ninitcond):')
    total_home = 0
    total_scratch = 0
    for mu, N, ninitcond, _, _, ngraphs, n_home, n_scratch in averaged:
        print(f'  mu={mu:.3f} N={N} ninitcond={ninitcond}: '
              f'{ngraphs} samples (home: {n_home}, scratch: {n_scratch})')
        total_home += n_home
        total_scratch += n_scratch
    print(f'Total: {total_home + total_scratch} samples '
          f'(home: {total_home}, scratch: {total_scratch})\n')


def write_summary(path_out, filename_out, results):
    os.makedirs(path_out, exist_ok=True)
    fout = open(f'{path_out}/{filename_out}', "w")
    fout.write('# mu N ninitcond mean_number_of_fixed_points sem_number_of_fixed_points ngraphs\n')
    for mu, N, ninitcond, mean_nfp, sem_nfp, ngraphs, _, _ in results:
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

    home_path = "./"
    scratch_path = "/mnt/beegfs/2a/dm27124/Ecology"
    path_out = "./ToDownload"

    pattern = (f'IBMF_T{T}_seq_gr_inside_RRG_eps_{eps}_mu_*_sigma_{sigma}_N_*_c_{c}'
               f'_seedgraph_*_Lotka_Volterra_final_av0_{avn0}_dn_{dn}'
               f'_ninitcond_*_tol_{tol}_maxiter_{max_iter}_damping_{damping}_summary.txt')

    filename_out = (f'IBMF_T{T}_seq_gr_inside_RRG_eps_{eps}_sigma_{sigma}_c_{c}'
                     f'_Lotka_Volterra_final_av0_{avn0}_dn_{dn}'
                     f'_tol_{tol}_maxiter_{max_iter}_damping_{damping}_nfixedpoints_avg.txt')

    results = gather_files(home_path, pattern, 'home')
    results += gather_files(scratch_path, pattern, 'scratch')

    results = deduplicate_seedgraphs(results)

    averaged_results = average_over_seedgraphs(results)
    print_sample_report(averaged_results)
    write_summary(path_out, filename_out, averaged_results)

    return 0


if __name__ == '__main__':
    main()
