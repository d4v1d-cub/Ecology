#ifndef __IBMF_CONVERGENCE_T0_H_INCLUDED__
#define __IBMF_CONVERGENCE_T0_H_INCLUDED__
// to do: correggere il numero alla riga 240
/**
 * @file IBMF_convergence_T0_seq.h
 * @brief Implementation of zero-temperature IBMF convergence
 * 
 * This file implements the convergence algorithm for the Individual Based Mean Field
 * approach at zero temperature (T=0). At T=0, the stationary solution reduces to
 * a simpler form where species abundances are directly proportional to their local fields
 * when positive, and zero otherwise.
 */

#include "IBMF_common_countFP.h"
#include <chrono>
#include <cstring>

using namespace std;

/**
 * @brief Update node abundances using zero-temperature IBMF equations
 * @param N Number of species
 * @param nodes Array of species nodes
 * @param tol Convergence tolerance
 * @param iter Current iteration number
 * @param sequence Array defining update order
 * @param damping Damping factor for updates (1.0 = no damping)
 * @param normfactor Small number to prevent division by zero
 * @return Maximum change in abundance across all nodes
 * 
 * At T=0, the update rule is:
 * n_i = max(0, 1 - sum_j a_ij n_j)
 * with optional damping to aid convergence
 */
double new_averages(long N, Tnode *nodes, double tol, int iter, long sequence[],
                    double damping, double normfactor = 1e-14){
    double var = 0, var_i;
    double av_new;
    long pos;
    for (long i = 0; i < N; i++){
        pos = sequence[i];
        nodes[pos].field = field_in(pos, nodes);
        if (nodes[pos].field > 0){
            av_new = damping * nodes[pos].field + (1 - damping) * nodes[pos].av;
        }else{
            av_new = (1 - damping) * nodes[pos].av;               
        }

        if (std::isnan(av_new) || std::isinf(av_new)){
            cerr << "Error: av_new is nan or inf at site i=" << pos << "   iter=" << iter << endl;
            return sqrt(-1);
        }
        
        var_i = fabs(av_new - nodes[pos].av);
        if (var_i > var){
            var = var_i;
        }
        if (var_i < tol){
            nodes[pos].converged = true;
        }
        else{
            nodes[pos].converged = false;
        }

        nodes[pos].av = av_new;
    }
    return var;
}


int convergence(long N, Tnode *nodes, double tol, int max_iter, bool &divergence, 
                long sequence[], double damping, double maximum=1e10, int min_consecutive=5){
    double var = tol + 1;
    int iter = 0;

    int consecutive = 0;
    while (consecutive < min_consecutive && iter < max_iter){
        var = new_averages(N, nodes, tol, iter, sequence, damping);
        iter++;
        if (std::isinf(var) || std::isnan(var) || var > maximum){
            divergence = true;
            return iter;
        }
        if (var < tol){
            consecutive++;
        }else{
            consecutive = 0;
        }
    }

    divergence = false;
    return iter;
}


size_t IBMF_single_try(unsigned long seed_seq, long N, Tnode *nodes, double tol,
                       int max_iter, double avn_0, double damping, bool random_init, double dn,
                       unsigned long seed_condinit, long sequence[], bool &divergence, int &iter,
                       gsl_rng * r_seq, gsl_rng * r_condinit){
    produce_random_seq(seed_seq, N, sequence, r_seq);
    init_avgs(N, nodes, avn_0, random_init, dn, seed_condinit, r_condinit);
    auto start = std::chrono::high_resolution_clock::now();
    iter = convergence(N, nodes, tol, max_iter, divergence, sequence, damping);
    auto end = std::chrono::high_resolution_clock::now();
    size_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return elapsed;
}

void print_fixed_points_summary(Tnode *nodes, long N, char *fileout_base, unsigned long seed_graph,
                                int attempts, bool print_avgs, const vector<int> &fixed_point_counts) {
    int num_fixed_points = nodes[0].fixed_points.size();
    char filesummary[300];
    sprintf(filesummary, "%s_summary.txt", fileout_base);
    
    FILE *fsummary = fopen(filesummary, "w");
    fprintf(fsummary,"# number count n_ave n_sq n[0] n[1]\n");   
    
    printf("\n========================================\n");
    printf("RIEPILOGO PUNTI FISSI TROVATI:\n");
    printf("Numero totale di punti fissi distinti: %d\n", num_fixed_points);
    printf("========================================\n");
    
    for (int fp_idx = 0; fp_idx < num_fixed_points; fp_idx++) {
        int count = fixed_point_counts[fp_idx];
        printf("\nPunto fisso %d:\n", fp_idx + 1);
        printf("  Trovato %d volte\n", count);
        printf("  Valori: ");
        for (long i = 0; i < min(N, 5L); i++) { // Stampa solo i primi 5 valori
            printf("%.6f ", nodes[i].fixed_points[fp_idx]);
        }
        if (N > 5) printf("...");
        printf("\n");

	fprintf(fsummary,"%d %g %g %g %g %g\n",fp_idx+1,count/(double)attempts,average_FP(N,nodes,fp_idx),average_sqr_FP(N,nodes,fp_idx),nodes[0].fixed_points[fp_idx],nodes[1].fixed_points[fp_idx]);   
	
    if (print_avgs){
        // Salva su file
        char filename[300];
        sprintf(filename, "%s_fixedpoint_%d.txt", fileout_base, fp_idx + 1);
        
        FILE *fp = fopen(filename, "w");
        if (fp) {
	  fprintf(fp, "# Punto fisso %d di %d - Trovato %d volte su %d prove\n", fp_idx + 1, num_fixed_points, count, attempts );
            for (long i = 0; i < N; i++) {
                fprintf(fp, "%ld %.10f\n", i, nodes[i].fixed_points[fp_idx]);
            }
            fclose(fp);
            printf("  Salvato in: %s\n", filename);
        }
    }
    }
    
    printf("\n========================================\n");
    fprintf(fsummary,"\n");
    fclose(fsummary);
}

/**
 * @brief Write pairwise distances between all distinct fixed points, with multiplicity
 * @param nodes Array of species nodes (nodes[i].fixed_points[k] holds species i's abundance at fixed point k)
 * @param N Number of species
 * @param fileout_base Base file name; output is written to "<fileout_base>_pairwise_dist.txt"
 * @param fixed_point_counts Number of times each distinct fixed point was found
 *
 * For every pair of distinct fixed points (k, l), k < l, writes a row with the
 * normalized Euclidean distance ||n_k - n_l|| / sqrt(N) and multiplicity
 * count_k * count_l. Pairs of attempts that converged to the SAME fixed point
 * (distance 0) are also included, with multiplicity count_k * (count_k - 1) / 2,
 * so the file directly gives the weights needed for a P(q) histogram.
 */
void print_pairwise_distances(Tnode *nodes, long N, char *fileout_base,
                              const vector<int> &fixed_point_counts) {
    int num_fixed_points = nodes[0].fixed_points.size();
    char filedist[300];
    sprintf(filedist, "%s_pairwise_dist.txt", fileout_base);

    FILE *fdist = fopen(filedist, "w");
    fprintf(fdist, "# distance multiplicity\n");

    double sqrtN = sqrt((double)N);

    for (int k = 0; k < num_fixed_points; k++) {
        long long Mk = fixed_point_counts[k];
        if (Mk > 1) {
            long long mult0 = Mk * (Mk - 1) / 2;
            fprintf(fdist, "%.10f %lld\n", 0.0, mult0);
        }
        for (int l = k + 1; l < num_fixed_points; l++) {
            double sumsq = 0.0;
            for (long i = 0; i < N; i++) {
                double diff = nodes[i].fixed_points[k] - nodes[i].fixed_points[l];
                sumsq += diff * diff;
            }
            double dist = sqrt(sumsq) / sqrtN;
            long long mult = Mk * (long long)fixed_point_counts[l];
            fprintf(fdist, "%.10f %lld\n", dist, mult);
        }
    }

    fclose(fdist);
    printf("Pairwise distances saved in: %s\n", filedist);
}

/**
 * @brief Build a checkpoint file base name by replacing the "_ninitcond_<N>" token
 * @param fileout_base Original file base, containing "_ninitcond_<num_init_conds>"
 * @param num_init_conds Total number of initial conditions the original base was built with
 * @param checkpoint_count Number of initial conditions explored so far (replaces num_init_conds)
 * @param out_buffer Buffer to receive the checkpoint file base
 * @param out_size Size of out_buffer
 */
void build_checkpoint_fileout_base(const char *fileout_base, int num_init_conds,
                                   long checkpoint_count, char *out_buffer, size_t out_size) {
    char search_str[50];
    char replace_str[50];
    sprintf(search_str, "_ninitcond_%d", num_init_conds);
    sprintf(replace_str, "_ninitcond_%ld", checkpoint_count);

    const char *pos = strstr(fileout_base, search_str);
    if (pos == nullptr) {
        snprintf(out_buffer, out_size, "%s", fileout_base);
        return;
    }
    int prefix_len = (int)(pos - fileout_base);
    snprintf(out_buffer, out_size, "%.*s%s%s", prefix_len, fileout_base, replace_str, pos + strlen(search_str));
}

bool is_power_of_two(long n) {
    return n > 0 && (n & (n - 1)) == 0;
}

void several_seq_IBMF_T0(unsigned long seed_graph, unsigned long seed_seq_init,
                         long N, Tnode *nodes, double tol,
                         int max_iter, unsigned long num_seq, double tol_fixed_point,
                         double avn_0, double damping, 
                         bool print_avgs,
                         char * fileout_base, bool random_init, double dn,
                         unsigned long id_0, int num_init_conds, bool pairwise_dist,
                         bool no_checkpoints) {

    long *sequence;
    sequence = new long[N];

    bool divergence;
    char fileavgs[300];
    size_t elapsed;

    // RNGs are allocated once and reseeded per attempt (via gsl_rng_set),
    // instead of being allocated/freed on every single attempt.
    gsl_rng * r_seq;
    gsl_rng * r_condinit;
    init_ran(r_seq, seed_seq_init);
    init_ran(r_condinit, id_0);

    // Inizializza lo storage dei punti fissi
    vector<int> fixed_point_counts;
    init_fixed_points_storage(nodes, N, fixed_point_counts);

    divergence = false;
    unsigned long seed_seq, seed_condinit;
    bool make_other_tries;
    int iter;
    bool same_fixed_point = true;
    bool cond = true;
    int attempts=0;
    int fp_index;
    
    seed_seq = seed_seq_init;
    seed_condinit = id_0;    
    
    while (seed_condinit < id_0 + num_init_conds && cond){
      seed_seq = seed_seq_init;
      while (seed_seq < seed_seq_init + num_seq && cond){
	elapsed = IBMF_single_try(seed_seq, N, nodes, tol, max_iter, avn_0, damping,
				  random_init, dn, seed_condinit, sequence, divergence,
				  iter, r_seq, r_condinit);
	attempts++;
	if (!divergence && iter < max_iter) {
	  fp_index = find_fixed_point_index(nodes, N, tol_fixed_point);
	  
	  if (fp_index >= 0) {
	    // Punto fisso già trovato in precedenza
	    update_fixed_point_counts(fixed_point_counts, fp_index);
	    same_fixed_point = true;
	  } else {
	    // Nuovo punto fisso
	    add_new_fixed_point(nodes, N, 1, fixed_point_counts);
	    same_fixed_point = false;
	    
	  }
	} else {
	  same_fixed_point = false;
	}
        
	
	seed_seq++;
      }

      long ninitcond_done = seed_condinit - id_0 + 1;
      if (!no_checkpoints && is_power_of_two(ninitcond_done) && ninitcond_done != num_init_conds && ninitcond_done >= 256){
	char fileout_checkpoint[300];
	build_checkpoint_fileout_base(fileout_base, num_init_conds, ninitcond_done,
				      fileout_checkpoint, sizeof(fileout_checkpoint));
	print_fixed_points_summary(nodes, N, fileout_checkpoint, seed_graph, attempts, print_avgs, fixed_point_counts);
      }

      seed_condinit++;
    }

    
    // Stampa i risultati finali
    print_fixed_points_summary(nodes, N, fileout_base, seed_graph, attempts, print_avgs, fixed_point_counts);

    if (pairwise_dist) {
        print_pairwise_distances(nodes, N, fileout_base, fixed_point_counts);
    }

    delete [] sequence;
    gsl_rng_free(r_seq);
    gsl_rng_free(r_condinit);
}


#endif
