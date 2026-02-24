#ifndef __IBMF_CONVERGENCE_T0_H_INCLUDED__
#define __IBMF_CONVERGENCE_T0_H_INCLUDED__

/**
 * @file IBMF_convergence_T0_seq.h
 * @brief Implementation of zero-temperature IBMF convergence
 * 
 * This file implements the convergence algorithm for the Individual Based Mean Field
 * approach at zero temperature (T=0). At T=0, the stationary solution reduces to
 * a simpler form where species abundances are directly proportional to their local fields
 * when positive, and zero otherwise.
 */

#include "IBMF_common_popdyn.h"
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

template <typename Func>
void new_averages(long S, Tnode *nodes, double tol, int iter, double damping, double av_c, 
                  double mu, double sigma, gsl_rng * r, Func draw_connectivity, double normfactor = 1e-14){
    double var = 0, var_i;
    double av_new;
    long pos;
    int c;
    for (long i = 0; i < S; i++){
        pos = gsl_rng_uniform_int(r, S);
        c = draw_connectivity(av_c, r);
        nodes[pos].field = field_in_pop(S, nodes, c, mu, sigma, r);

        if (nodes[pos].field > 0){
            av_new = damping * nodes[pos].field + (1 - damping) * nodes[pos].av;
        }else{
            av_new = (1 - damping) * nodes[pos].av;               
        }

        if (std::isnan(av_new) || std::isinf(av_new)){
            cerr << "Error: av_new is nan or inf at site i=" << pos << "   iter=" << iter << endl;
        }

        nodes[pos].av = av_new;
    }
}


template <typename Func>
int convergence(long S, Tnode *nodes, double tol, int max_iter, bool &divergence, 
                double damping, double av_c, double mu, double sigma, 
                gsl_rng * r, Func draw_connectivity, int print_every, 
                double maximum=1e10, int min_consecutive=5){
    double var = tol + 1, av_pop, av_sqr_pop, av_pop_new, av_sqr_pop_new;
    int iter = 0;
    int consecutive = 0;
    av_pop = average(S, nodes);
    av_sqr_pop = average_sqr(S, nodes);
    if (print_every > 0){
        cerr << "# iteration   av(n)   av(n^2)   sqrt(av(n^2) - av(n)^2)   delta" << endl;
    }
    while (consecutive < min_consecutive && iter < max_iter){
        new_averages(S, nodes, tol, iter, damping, av_c, mu, sigma, r, draw_connectivity);
        iter++;
        av_pop_new = average(S, nodes);
        av_sqr_pop_new = average_sqr(S, nodes);
        var = max(fabs(av_pop_new - av_pop), fabs(av_sqr_pop_new - av_sqr_pop));
        if (std::isinf(var) || std::isnan(var) || var > maximum){
            divergence = true;
            return iter;
        }
        if (var < tol){
            consecutive++;
        }else{
            consecutive = 0;
        }

        av_pop = av_pop_new;
        av_sqr_pop = av_sqr_pop_new;

        if (print_every > 0 && iter % print_every == 0){
            cerr << iter << "\t" << av_pop << "\t" << av_sqr_pop << "\t" << sqrt(av_sqr_pop - av_pop*av_pop) << "\t" << var << endl;
        }
    }

    divergence = false;
    return iter;
}



template <typename Func>
void measure_observables(long S, Tnode *nodes, double tol, 
                         bool &divergence, double damping, double av_c, double mu, double sigma, 
                         gsl_rng * r, int N_measurements, double &av_pop, double &av_sqr_pop,
                         double &av_counter_dead, Func draw_connectivity, double maximum=1e10){
    av_pop = 0;
    av_sqr_pop = 0;
    av_counter_dead = 0;
    

    int consecutive = 0;
    for (int i = 0; i < N_measurements; i++){
        new_averages(S, nodes, tol, i, damping, av_c, mu, sigma, r, draw_connectivity);
        av_pop += average(S, nodes);
        av_sqr_pop += average_sqr(S, nodes);
        av_counter_dead += count_dead(S, nodes);
    }

    av_pop /= N_measurements;
    av_sqr_pop /= N_measurements;
    av_counter_dead /= N_measurements;
}



size_t IBMF(long S, Tnode *nodes, double tol, int max_iter, double avn_0, 
            double damping, bool random_init, double dn, unsigned long seed_condinit, 
            bool &divergence, int &iter, double av_c, double mu, double sigma, char * graph_type, 
            unsigned long seed_choose, int N_measurements, double &av_pop,
            double &av_sqr_pop, double &av_counter_dead, int print_every){
    gsl_rng * r;
    init_avgs(S, nodes, avn_0, random_init, dn, seed_condinit);
    init_ran(r, seed_choose);
    auto start = std::chrono::high_resolution_clock::now();
    if (graph_type == string("RRG")) {
        iter = convergence(S, nodes, tol, max_iter, divergence, damping, av_c, mu, sigma, r, 
                           connectivity_RRG, print_every);
        measure_observables(S, nodes, tol, divergence, damping, av_c, mu, sigma, r, 
                            N_measurements, av_pop, av_sqr_pop, av_counter_dead, connectivity_RRG);
    }else if (graph_type == string("ER")){
        iter = convergence(S, nodes, tol, max_iter, divergence, damping, av_c, mu, sigma, r, 
                           connectivity_ER, print_every);
        measure_observables(S, nodes, tol, divergence, damping, av_c, mu, sigma, r, N_measurements, av_pop, 
                            av_sqr_pop, av_counter_dead, connectivity_ER);
    }else{
        cerr << "graph_type must be RRG or ER" << endl;
        exit(1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    size_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return elapsed;
}


void run_IBMF_T0(unsigned long seed_choose, long S, Tnode *nodes, double tol, int max_iter, 
                 double avn_0, double damping, int print_every, bool print_avgs, char * fileout_base, 
                 bool random_init, double dn, unsigned long id_0, double av_c, double mu, 
                 double sigma, char * graph_type, int N_measurements){

    bool divergence;

    char fileavgs[300];
    
    
    divergence = false;
    bool make_other_tries;
    int iter;
    bool same_fixed_point = true;

    double av_pop, av_sqr_pop, av_counter_dead;
    size_t elapsed = IBMF(S, nodes, tol, max_iter, avn_0, damping, random_init, dn, id_0, divergence, 
                          iter, av_c, mu, sigma, graph_type, seed_choose, N_measurements, 
                          av_pop, av_sqr_pop, av_counter_dead, print_every);
    print_results_short(iter, nodes, S, id_0, max_iter, divergence, elapsed, 
                        av_counter_dead, av_pop, av_sqr_pop);
    if (print_avgs){
        sprintf(fileavgs, "%s.txt", fileout_base);
        print_avgs_to_file(nodes, S, fileavgs);
    }

}

#endif