#ifndef __IBMF_CONVERGENCE_FINITE_T_H_INCLUDED__
#define __IBMF_CONVERGENCE_FINITE_T_H_INCLUDED__

/**
 * @file IBMF_convergence_finite_T_seq.h
 * @brief Implementation of finite-temperature IBMF convergence
 * 
 * This file implements the convergence algorithm for the Individual Based Mean Field
 * approach at finite temperature (T>0). At finite T, the stationary solution involves
 * confluent hypergeometric functions that arise from the Fokker-Planck equation.
 * The implementation handles both the analytical solution using special functions
 * and asymptotic approximations when numerical evaluation becomes unstable.
 */

#include "IBMF_common.h"
#include <chrono>

using namespace std;


/**
 * @brief Compute coefficients for the finite temperature IBMF solution
 * @param beta Inverse temperature (1/T)
 * @param lambda Immigration rate
 * @param coefficients Output matrix of coefficients for the hypergeometric functions
 * @param gamma_vals Output array of gamma function values
 * @param maximum Maximum allowed value before switching to asymptotic form
 * @return True if gamma functions diverge and asymptotic form is used
 * 
 * The stationary solution at finite T involves ratios of confluent hypergeometric
 * functions with coefficients determined by beta and lambda. This function computes
 * these coefficients, handling both the regular case and the asymptotic approximation
 * when the gamma functions become too large to evaluate directly.
 */
bool comp_coefficients(double beta, double lambda, double **&coefficients, double *&gamma_vals, 
                       double maximum=1e10){
    bool gamma_diverges = false;
    gamma_vals = new double[2];
    // Check if gamma functions can be evaluated directly
    if (isnan(gsl_sf_gamma((1 + beta * lambda) / 2)) || isinf(gsl_sf_gamma((1 + beta * lambda) / 2)) || 
        gsl_sf_gamma((1 + beta * lambda) / 2) > maximum){
        gamma_diverges = true;
        gamma_vals[0] = sqrt(2 * M_PI / beta / lambda) * pow(beta * lambda / 2 / M_E, beta * lambda / 2);
        gamma_vals[1] = sqrt(4 * M_PI / (1 + beta * lambda)) * pow((1 + beta * lambda) / 2 / M_E, (1 + beta * lambda) / 2);
    }else{
        gamma_vals[0] = gsl_sf_gamma(beta * lambda / 2);
        gamma_vals[1] = gsl_sf_gamma((1 + beta * lambda) / 2);
    }

    coefficients = new double *[2];
    for (int i = 0; i < 2; i++){
        coefficients[i] = new double[2];
    }

    if (gamma_diverges){
        coefficients[0][0] = 1;
        coefficients[0][1] = beta * sqrt(lambda) * (1 - 1.0 / 4 / beta / lambda);

        coefficients[1][0] = sqrt(lambda) * (1 - 1.0 / 4 / beta / lambda);
        coefficients[1][1] = lambda * beta;
    }else{
        double gammabl2 = gsl_sf_gamma(beta * lambda / 2);
        double gammabl12 = gsl_sf_gamma((1 + beta * lambda) / 2);
        
        coefficients[0][0] = sqrt(beta / 2) * gammabl2;
        coefficients[0][1] = beta * gammabl12;

        coefficients[1][0] = gammabl12;
        coefficients[1][1] = sqrt(beta / 2) * beta * lambda * gammabl2;
    }

    return gamma_diverges;
}


double find_divergence_max(double beta, double alpha, double hmax=100, double precision=1e-4, double maximum=1e10){
    double val1, val2;
    val1 = gsl_sf_hyperg_1F1(alpha, 0.5, beta * hmax * hmax / 2);
    val2 = gsl_sf_hyperg_1F1(alpha + 0.5, 1.5, beta * hmax * hmax / 2);
    while (!(isnan(val1) || isinf(val1) || isnan(val2) || isinf(val2) || 
             val1 > maximum || val2 > maximum)){
        hmax *= 2;
        val1 = gsl_sf_hyperg_1F1(alpha, 0.5, beta * hmax * hmax / 2);
        val2 = gsl_sf_hyperg_1F1(alpha + 0.5, 1.5, beta * hmax * hmax / 2);   
    }

    double hmin = 0;
    double h = (hmax + hmin) / 2;
    while (hmax - hmin > precision){
        val1 = gsl_sf_hyperg_1F1(alpha, 0.5, beta * h * h / 2);
        val2 = gsl_sf_hyperg_1F1(alpha + 0.5, 1.5, beta * h * h / 2);
        if (isnan(val1) || isinf(val1) || isnan(val2) || isinf(val2) || 
            val1 > maximum || val2 > maximum){
            hmax = h;
        }else{
            hmin = h;
        }
        h = (hmax + hmin) / 2;
    }

    cerr << "Divergence found at h = " << hmax << endl;
    cerr << "Last value to converge: " << hmin << endl;
    return hmin;
}


double numerator_av(double beta, double lambda, double hi, double *coefficients){
    return coefficients[0] * gsl_sf_hyperg_1F1((1 + beta * lambda) / 2, 0.5, beta * hi * hi / 2) +
           coefficients[1] * hi * gsl_sf_hyperg_1F1(1 + beta * lambda / 2, 1.5, beta * hi * hi / 2);
}


double denominator(double beta, double lambda, double hi, double *coefficients, double normfactor = 1e-14){
    return coefficients[0] * gsl_sf_hyperg_1F1(beta * lambda / 2, 0.5, beta * hi * hi / 2) + 
           coefficients[1] * hi * gsl_sf_hyperg_1F1((1 + beta * lambda) / 2, 1.5, beta * hi * hi / 2)
           + normfactor;
}


double find_divergence_min(double beta, double lambda, double **coefficients, double hmin=-100, double precision=1e-4, double maximum=1e10){
    double num, den;
    num = numerator_av(beta, lambda, hmin, coefficients[1]);
    den = denominator(beta, lambda, hmin, coefficients[0]);
    
    while (!(isnan(num) || isinf(num) || isnan(den) || isinf(den) || 
             num > maximum || den > maximum || num < 0 || den < 0)){
        hmin *= 2;
        num = numerator_av(beta, lambda, hmin, coefficients[1]);
        den = denominator(beta, lambda, hmin, coefficients[0]);   
    }

    double hmax = 0;
    double h = (hmax + hmin) / 2;
    while (hmax - hmin > precision){
        num = numerator_av(beta, lambda, h, coefficients[1]);
        den = denominator(beta, lambda, h, coefficients[0]); 
        if (isnan(num) || isinf(num) || isnan(den) || isinf(den) || 
             num > maximum || den > maximum || num < 0 || den < 0){
            hmin = h;
        }else{
            hmax = h;
        }
        h = (hmax + hmin) / 2;
    }

    cerr << "Divergence found at h = " << hmin << endl;
    cerr << "Last value to converge: " << hmax << endl;
    return hmin;
}

double new_averages(long N, double beta, double lambda, Tnode *nodes, double tol, 
                    double hmin, double hmax, double **coefficients, double *gamma_vals, 
                    int iter, long sequence[], double damping, double normfactor = 1e-14){
    double var = 0, var_i;
    double av_new;
    long pos;
    for (long i = 0; i < N; i++){
        pos = sequence[i];
        nodes[pos].field = field_in(pos, nodes);
        if (nodes[pos].field > hmax){
            av_new = damping * nodes[pos].field * (1 - 1.0 / beta / nodes[pos].field / nodes[pos].field + 
                                                   lambda / nodes[pos].field / nodes[pos].field) + 
                     (1 - damping) * nodes[pos].av;      
        }else if (nodes[pos].field < hmin)
        {
            av_new = damping * lambda / fabs(nodes[pos].field) + (1 - damping) * nodes[pos].av;
        }else if (nodes[pos].field == 0){
            av_new = damping * sqrt(2.0 / beta) * gamma_vals[1] / gamma_vals[0] + (1 - damping) * nodes[pos].av;
        }else {
            av_new = damping * numerator_av(beta, lambda, nodes[pos].field, coefficients[1]) /
                     denominator(beta, lambda, nodes[pos].field, coefficients[0], normfactor) + 
                     (1 - damping) * nodes[pos].av;
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


int convergence(long N, double beta, double lambda, Tnode *nodes, double tol, 
                 int max_iter, bool &divergence, double hmin, double hmax, double **coefficients,
                 double *gamma_vals, long sequence[], double damping, 
                 double maximum=1e10, int min_consecutive=5){
    double var = tol + 1;
    int iter = 0;

    int consecutive = 0;
    while (consecutive < min_consecutive && iter < max_iter){
        var = new_averages(N, beta, lambda, nodes, tol, hmin, hmax, coefficients, gamma_vals,
                           iter, sequence, damping);
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


size_t IBMF_single_try(unsigned long seed_seq, long N, Tnode *nodes, double beta, double lambda, double tol,
                       int max_iter, double avn_0, double damping, bool random_init, double dn, 
                       unsigned long seed_condinit, long sequence[], bool &divergence, int &iter, 
                       double hmin, double hmax, double **coefficients, double *gamma_vals){
    produce_random_seq(seed_seq, N, sequence);
    init_avgs(N, nodes, avn_0, random_init, dn, seed_condinit);
    auto start = std::chrono::high_resolution_clock::now();
    iter = convergence(N, beta, lambda, nodes, tol, max_iter, divergence, 
                       hmin, hmax, coefficients, gamma_vals, sequence, damping);
    auto end = std::chrono::high_resolution_clock::now();
    size_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return elapsed;
}


void several_seq_IBMF(unsigned long seed_graph, unsigned long seed_seq_init, 
                      long N, Tnode *nodes, double T, double lambda, double tol,
                      int max_iter, unsigned long num_seq, double tol_fixed_point,
                      double avn_0, double damping, 
                      bool print_only_last, bool print_avgs, 
                      char * fileout_base, bool random_init, double dn, unsigned long id_0, int num_init_conds){
    double beta = 1.0 / T;

    double hmax = find_divergence_max(beta, (1 + beta * lambda) / 2);
    double **coefficients, *gamma_vals;
    comp_coefficients(beta, lambda, coefficients, gamma_vals);
    double hmin = find_divergence_min(beta, lambda, coefficients);

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
    size_t elapsed = IBMF_single_try(seed_seq, N, nodes, beta, lambda, tol, max_iter, avn_0, 
                                     damping, random_init, dn, seed_condinit, sequence, divergence, 
                                     iter, hmin, hmax, coefficients, gamma_vals);
        
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
            elapsed = IBMF_single_try(seed_seq, N, nodes, beta, lambda, tol, max_iter, avn_0, 
                                      damping, random_init, dn, seed_condinit, sequence, divergence, 
                                      iter, hmin, hmax, coefficients, gamma_vals);
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
                elapsed = IBMF_single_try(seed_seq, N, nodes, beta, lambda, tol, max_iter, avn_0, 
                                          damping, random_init, dn, seed_condinit, sequence, divergence, 
                                          iter, hmin, hmax, coefficients, gamma_vals);
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