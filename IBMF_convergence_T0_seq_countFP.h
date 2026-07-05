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
                       unsigned long seed_condinit, long sequence[], bool &divergence, int &iter){
    produce_random_seq(seed_seq, N, sequence);
    init_avgs(N, nodes, avn_0, random_init, dn, seed_condinit);
    auto start = std::chrono::high_resolution_clock::now();
    iter = convergence(N, nodes, tol, max_iter, divergence, sequence, damping);
    auto end = std::chrono::high_resolution_clock::now();
    size_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return elapsed;
}

void print_fixed_points_summary(Tnode *nodes, long N, char *fileout_base, unsigned long seed_graph, int attempts) {
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
        int count = nodes[0].fixed_point_counts[fp_idx];
        printf("\nPunto fisso %d:\n", fp_idx + 1);
        printf("  Trovato %d volte\n", count);
        printf("  Valori: ");
        for (long i = 0; i < min(N, 5L); i++) { // Stampa solo i primi 5 valori
            printf("%.6f ", nodes[i].fixed_points[fp_idx]);
        }
        if (N > 5) printf("...");
        printf("\n");

	fprintf(fsummary,"%d %g %g %g %g %g\n",fp_idx+1,count/(double)attempts,average_FP(N,nodes,fp_idx),average_sqr_FP(N,nodes,fp_idx),nodes[0].fixed_points[fp_idx],nodes[1].fixed_points[fp_idx]);   
	
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
    
    printf("\n========================================\n");
    fprintf(fsummary,"\n");
    fclose(fsummary);
}

void several_seq_IBMF_T0(unsigned long seed_graph, unsigned long seed_seq_init, 
                         long N, Tnode *nodes, double tol,
                         int max_iter, unsigned long num_seq, double tol_fixed_point,
                         double avn_0, double damping, 
                         bool print_only_last, bool print_avgs, 
                         char * fileout_base, bool random_init, double dn, 
                         unsigned long id_0, int num_init_conds) {

    long *sequence;
    sequence = new long[N];

    bool divergence;
    char fileavgs[300];
    size_t elapsed;

    // Inizializza lo storage dei punti fissi
    init_fixed_points_storage(nodes, N);

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
				  iter);
	attempts++;
	if (!divergence && iter < max_iter) {
	  fp_index = find_fixed_point_index(nodes, N, tol_fixed_point);
	  
	  if (fp_index >= 0) {
	    // Punto fisso già trovato in precedenza
	    update_fixed_point_counts(nodes, N, fp_index);
	    same_fixed_point = true;
	  } else {
	    // Nuovo punto fisso
	    add_new_fixed_point(nodes, N, 1);
	    same_fixed_point = false;
	    
	  }
	} else {
	  same_fixed_point = false;
	}
        
	if (!print_only_last){
	  print_results_short(iter, nodes, N, seed_graph, seed_seq, seed_condinit, max_iter, divergence, same_fixed_point, elapsed);
	} else {
	  if (!same_fixed_point || divergence || iter >= max_iter){
	    cond = false;
	  }
	}
	seed_seq++;
      }
      seed_condinit++;
    }
    
    if (print_only_last){
      print_results_short(iter, nodes, N, seed_graph, seed_seq-1, seed_condinit-1, max_iter, divergence, same_fixed_point, elapsed);
    }

    
    // Stampa i risultati finali
    print_fixed_points_summary(nodes, N, fileout_base, seed_graph, attempts);
    
    delete [] sequence;
}


#endif
