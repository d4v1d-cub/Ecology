#ifndef __IBMF_CONVERGENCE_T0_H_INCLUDED__
#define __IBMF_CONVERGENCE_T0_H_INCLUDED__

#include "IBMF_common.h"
#include <chrono>

using namespace std;

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

        if (isnan(av_new) || isinf(av_new)){
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
        if (isinf(var) || isnan(var) || var > maximum){
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


void several_seq_IBMF_T0(unsigned long seed_graph, unsigned long seed_seq_init, 
                         long N, Tnode *nodes, double tol,
                         int max_iter, unsigned long num_seq, double tol_fixed_point,
                         double avn_0, double damping, 
                         bool print_only_last, bool print_avgs, 
                         char * fileout_base, bool random_init, double dn, unsigned long id_0, int num_init_conds){

    long sequence[N];

    bool divergence;

    char fileavgs[300];


    divergence = false;
    unsigned long seed_seq, seed_condinit;
    bool make_other_tries;
    int iter;
    bool same_fixed_point = true;

    seed_seq = seed_seq_init;
    seed_condinit = id_0;
    
    size_t elapsed = IBMF_single_try(seed_seq, N, nodes, tol, max_iter, avn_0, damping,
                                     random_init, dn, seed_condinit, sequence, divergence, 
                                     iter);
        
    if (!print_only_last){
        print_results_short(iter, nodes, N, seed_graph, seed_seq, seed_condinit, max_iter, divergence, true, elapsed);
        if (print_avgs){
            sprintf(fileavgs, "%s_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
            print_avgs_to_file(nodes, N, fileavgs);
        }
    }else if(divergence || iter >= max_iter){
        print_results_short(iter, nodes, N, seed_graph, seed_seq, seed_condinit, max_iter, divergence, true, elapsed);
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
            elapsed = IBMF_single_try(seed_seq, N, nodes, tol, max_iter, avn_0, damping,
                                      random_init, dn, seed_condinit, sequence, divergence, 
                                      iter);
            same_fixed_point = compare_fixed_points(nodes, N, tol_fixed_point);
            if (!print_only_last){
                print_results_short(iter, nodes, N, seed_graph, seed_seq, seed_condinit, max_iter, divergence, same_fixed_point, elapsed);
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
                elapsed = IBMF_single_try(seed_seq, N, nodes, tol, max_iter, avn_0, damping,
                                          random_init, dn, seed_condinit, sequence, divergence, 
                                          iter);
                same_fixed_point = compare_fixed_points(nodes, N, tol_fixed_point);
                if (!print_only_last){
                    print_results_short(iter, nodes, N, seed_graph, seed_seq, seed_condinit, max_iter, divergence, same_fixed_point, elapsed);
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
            print_results_short(iter, nodes, N, seed_graph, seed_seq-1, seed_condinit-1, max_iter, divergence, same_fixed_point, elapsed);
            if (print_avgs){
                sprintf(fileavgs, "%s_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq-1, seed_condinit-1);
                print_avgs_to_file(nodes, N, fileavgs);
            }
        }
    }

}

#endif