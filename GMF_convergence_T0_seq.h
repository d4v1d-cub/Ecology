#ifndef __GMF_CONVERGENCE_T0_H_INCLUDED__
#define __GMF_CONVERGENCE_T0_H_INCLUDED__

#include "GMF_common.h"
#include <chrono>

using namespace std;



double new_averages(long M, Tedge *edges, double tol, int iter, long sequence[], 
                    double damping, double normfactor = 1e-14, double maximum=1e6){
    double delta = 0, delta_av, delta_chi_cav, h, den, av_new, 
           chi_cav_new, var_cav, field_cav;

    long pos;
    for (long e = 0; e < M; e++){
        pos = sequence[e];
        for (int k = 0; k < 2; k++){
            field_cav = field_cav_in(pos, k, edges);
            var_cav = var_cav_in(pos, k, edges);
            if (var_cav > 0){
                edges[pos].var_cav_positive[k] = true;   
                h = field_cav * var_cav;
                if (h > 0){
                    av_new = damping * h + (1 - damping) * edges[pos].cond_av[k];
                    chi_cav_new = damping * var_cav + (1 - damping) * edges[pos].chi_cav[k];
                }else if (h < 0){
                    av_new = (1 - damping) * edges[pos].cond_av[k];
                    chi_cav_new = (1 - damping) * edges[pos].chi_cav[k];
                }
            }else{
                edges[pos].var_cav_positive[k] = false;
                av_new = damping * maximum + (1 - damping) * edges[pos].cond_av[k];
                chi_cav_new = damping * maximum + (1 - damping) * edges[pos].chi_cav[k];
            }
            
            if (isnan(av_new) || isinf(av_new) || isnan(chi_cav_new) || isinf(chi_cav_new)){
                cerr << "Error: averages are nan or inf at site e=" << pos << "  node=" << edges[pos].nodes_in[k] << "   iter=" << iter << endl;
                return sqrt(-1);
            }

            delta_av = fabs(av_new - edges[pos].cond_av[k]);
            if (edges[pos].var_cav_positive[k]){
                delta_chi_cav = fabs(chi_cav_new - edges[pos].chi_cav[k]);
            }else{
                delta_chi_cav = 1;
            }
            
            
            if (delta_av > delta){
                delta = delta_av;
            }
            if (delta_chi_cav > delta){
                delta = delta_chi_cav;
            }

            if (delta_chi_cav < tol){
                edges[pos].chi_cav_converged[k] = true;
            }else{
                edges[pos].chi_cav_converged[k] = false;
            }

            edges[pos].cond_av[k] = av_new;
            edges[pos].chi_cav[k] = chi_cav_new;

            if (delta_av < tol && delta_chi_cav < tol){
                edges[pos].converged[k] = true;
            }else{
                edges[pos].converged[k] = false;
            }
        }

    }
    return delta;
}



double average(long N, Tnode *nodes, Tedge *edges, double normfactor = 1e-14){
    double av = 0;
    double h, den;
    for (long i = 0; i < N; i++){
        nodes[i].field = field_in(i, nodes, edges);
        nodes[i].var = var_in(i, nodes, edges);
        if (nodes[i].var > 0){
            h = nodes[i].field * nodes[i].var;
            if (h > 0){
                nodes[i].av = h;
                nodes[i].chi = nodes[i].var;
            } else{
                nodes[i].av = 0;
                nodes[i].chi = 0;
            } 
        }else{
            nodes[i].av = 0;
            nodes[i].chi = 0;
        }
        
        av += nodes[i].av;
        
    }
    return av / N;
}


int convergence(long M, Tedge *edges, double tol, int max_iter, 
                bool &divergence, long sequence[], double damping, 
                double maximum=1e10, int min_consecutive=5){
    double delta = tol + 1;
    int iter = 0;

    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            edges[e].chi_cav_converged[k] = false;
        }
    }

    int consecutive = 0;
    while (consecutive < min_consecutive && iter < max_iter){
        delta = new_averages(M, edges, tol, iter, sequence, damping);
        iter++;
        if (isinf(delta) || isnan(delta) || delta > maximum){
            divergence = true;
            return iter;
        }
        if (delta < tol){
            consecutive++;
        }else{
            consecutive = 0;
        }
    }
    divergence = false;
    return iter;
}


size_t GMF_single_try(unsigned long seed_seq, long M, Tedge *edges, double tol,
                      int max_iter, double avn_0, double chi_0, double damping, bool random_init, 
                      double dn, double dchi, unsigned long seed_condinit, long sequence[], 
                      bool &divergence, int &iter){ 
    produce_random_seq(seed_seq, M, sequence);
    init_avgs(M, edges, avn_0, chi_0, random_init, dn, dchi, seed_condinit);
    auto start = std::chrono::high_resolution_clock::now();
    iter = convergence(M, edges, tol, max_iter, divergence, sequence, damping);
    auto end = std::chrono::high_resolution_clock::now();
    size_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return elapsed;
}


void several_seq_GMF_T0(unsigned long seed_graph, unsigned long seed_seq_init, 
                        long N, long M, Tnode *nodes, Tedge *edges, double tol,
                        int max_iter, unsigned long num_seq, double tol_fixed_point,
                        double avn_0, double chi_0, double damping, bool print_only_last, bool print_avgs, 
                        char * fileout_base, bool random_init, double dn, double dchi, unsigned long id_0, 
                        int num_init_conds){

    long *sequence;
    sequence = new long[M];

    bool divergence;

    char fileavgs[300];


    divergence = false;
    unsigned long seed_seq, seed_condinit;
    bool make_other_tries;
    int iter;
    bool same_fixed_point = true;

    seed_seq = seed_seq_init;
    seed_condinit = id_0;
    
    size_t elapsed = GMF_single_try(seed_seq, M, edges, tol, max_iter, avn_0, chi_0, damping,
                                    random_init, dn, dchi, seed_condinit, sequence, divergence, 
                                    iter);
    double av = average(N, nodes, edges);
        
    if (!print_only_last){
        print_results(av, iter, nodes, edges, N, M, seed_graph, seed_seq, seed_condinit, max_iter, divergence, true, elapsed);
        if (print_avgs){
            sprintf(fileavgs, "%s_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
            print_avgs_to_file(nodes, N, fileavgs);
        }
    }else if(divergence || iter >= max_iter){
        print_results(av, iter, nodes, edges, N, M, seed_graph, seed_seq, seed_condinit, max_iter, divergence, true, elapsed);
        if (print_avgs){
            sprintf(fileavgs, "%s_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
            print_avgs_to_file(nodes, N, fileavgs);
        }
    }

    
    make_other_tries = !print_only_last || (!divergence && iter < max_iter);

    if (make_other_tries){
        set_av_prev(nodes, N);
        bool cond = true;

        seed_seq = seed_seq_init + 1;
        while (seed_seq < seed_seq_init + num_seq && cond){
            elapsed = GMF_single_try(seed_seq, M, edges, tol, max_iter, avn_0, chi_0, damping,
                                     random_init, dn, dchi, seed_condinit, sequence, divergence, 
                                     iter);
            av = average(N, nodes, edges);
            same_fixed_point = compare_fixed_points(nodes, N, tol_fixed_point);
            if (!print_only_last){
                print_results(av, iter, nodes, edges, N, M, seed_graph, seed_seq, seed_condinit, max_iter, divergence, same_fixed_point, elapsed);
                if (print_avgs){
                    sprintf(fileavgs, "%s_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
                    print_avgs_to_file(nodes, N, fileavgs);
                }
            }else{
                if (!same_fixed_point || divergence || iter >= max_iter){
                    cond = false;
                }
            }
            seed_seq++;
        }
        
        seed_condinit++;
        
        while (seed_condinit < id_0 + num_init_conds && cond){
            seed_seq = seed_seq_init;
            while (seed_seq < seed_seq_init + num_seq && cond){
                elapsed = GMF_single_try(seed_seq, M, edges, tol, max_iter, avn_0, chi_0, damping,
                                         random_init, dn, dchi, seed_condinit, sequence, divergence, 
                                         iter);
                av = average(N, nodes, edges);
                same_fixed_point = compare_fixed_points(nodes, N, tol_fixed_point);
                if (!print_only_last){
                    print_results(av, iter, nodes, edges, N, M, seed_graph, seed_seq, seed_condinit, max_iter, divergence, same_fixed_point, elapsed);
                    if (print_avgs){
                        sprintf(fileavgs, "%s_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
                        print_avgs_to_file(nodes, N, fileavgs);
                    }
                }else{
                    if (!same_fixed_point || divergence || iter >= max_iter){
                        cond = false;
                    }
                }
                seed_seq++;
            }
            seed_condinit++;
        }

        if (print_only_last){
            print_results(av, iter, nodes, edges, N, M, seed_graph, seed_seq-1, seed_condinit-1, max_iter, divergence, same_fixed_point, elapsed);
            if (print_avgs){
                sprintf(fileavgs, "%s_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq-1, seed_condinit-1);
                print_avgs_to_file(nodes, N, fileavgs);
            }
        }
    }
    delete [] sequence;

}


#endif