#ifndef __GMF_CONVERGENCE_FINITE_T_H_INCLUDED__
#define __GMF_CONVERGENCE_FINITE_T_H_INCLUDED__

#include "GMF_common.h"
#include <chrono>

using namespace std;



bool comp_coefficients(double beta, double lambda, double **&coefficients, double *&gamma_vals, 
                       double maximum=1e10){
    bool gamma_diverges = false;
    gamma_vals = new double[2];
    if (std::isnan(gsl_sf_gamma((1 + beta * lambda) / 2)) || std::isinf(gsl_sf_gamma((1 + beta * lambda) / 2)) || 
        gsl_sf_gamma((1 + beta * lambda) / 2) > maximum){
        gamma_diverges = true;
        gamma_vals[0] = sqrt(2 * M_PI / beta / lambda) * pow(beta * lambda / 2 / M_E, beta * lambda / 2);
        gamma_vals[1] = sqrt(4 * M_PI / (1 + beta * lambda)) * pow((1 + beta * lambda) / 2 / M_E, (1 + beta * lambda) / 2);
    }else{
        gamma_vals[0] = gsl_sf_gamma(beta * lambda / 2);
        gamma_vals[1] = gsl_sf_gamma((1 + beta * lambda) / 2);
    }

    coefficients = new double *[3];
    for (int i = 0; i < 3; i++){
        coefficients[i] = new double[2];
    }

    if (gamma_diverges){
        coefficients[0][0] = 1;
        coefficients[0][1] = beta * sqrt(lambda) * (1 - 1.0 / 4 / beta / lambda);

        coefficients[1][0] = sqrt(lambda) * (1 - 1.0 / 4 / beta / lambda);
        coefficients[1][1] = lambda * beta;

        coefficients[2][0] = lambda;
        coefficients[2][1] = lambda * beta * sqrt(lambda) * (1 + 3.0 / 4 / beta / lambda);

    }else{
        double gammabl2 = gsl_sf_gamma(beta * lambda / 2);
        double gammabl12 = gsl_sf_gamma((1 + beta * lambda) / 2);
        
        coefficients[0][0] = sqrt(beta / 2) * gammabl2;
        coefficients[0][1] = beta * gammabl12;

        coefficients[1][0] = gammabl12;
        coefficients[1][1] = sqrt(beta / 2) * beta * lambda * gammabl2;

        coefficients[2][0] = sqrt(beta / 2) * lambda * gammabl2;
        coefficients[2][1] = (1 + beta * lambda) * gammabl12;
    }

    return gamma_diverges;
}


double find_divergence_max(double beta, double alpha, double hmax=100, double precision=1e-4, double maximum=1e10){
    double val1, val2;
    val1 = gsl_sf_hyperg_1F1(alpha, 0.5, beta * hmax * hmax / 2);
    val2 = gsl_sf_hyperg_1F1(alpha + 0.5, 1.5, beta * hmax * hmax / 2);
    while (!(std::isnan(val1) || std::isinf(val1) || std::isnan(val2) || std::isinf(val2) || 
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
        if (std::isnan(val1) || std::isinf(val1) || std::isnan(val2) || std::isinf(val2) || 
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

double numerator_av(double beta, double lambda, double hi_div_Q, double *coefficients){
    return coefficients[0] * gsl_sf_hyperg_1F1((1 + beta * lambda) / 2, 0.5, beta * hi_div_Q * hi_div_Q / 2) +
           coefficients[1] * hi_div_Q * gsl_sf_hyperg_1F1(1 + beta * lambda / 2, 1.5, beta * hi_div_Q * hi_div_Q / 2);
}


double numerator_q_sqr(double beta, double lambda, double hi_div_Q, double *coefficients){
    return coefficients[0] * gsl_sf_hyperg_1F1(1 + beta * lambda / 2, 0.5, beta * hi_div_Q * hi_div_Q / 2) + 
           coefficients[1] * hi_div_Q * gsl_sf_hyperg_1F1((3 + beta * lambda) / 2, 1.5, beta * hi_div_Q * hi_div_Q / 2);
}


double denominator(double beta, double lambda, double hi_div_Q, double *coefficients, double normfactor = 1e-14){
    return coefficients[0] * gsl_sf_hyperg_1F1(beta * lambda / 2, 0.5, beta * hi_div_Q * hi_div_Q / 2) + 
           coefficients[1] * hi_div_Q * gsl_sf_hyperg_1F1((1 + beta * lambda) / 2, 1.5, beta * hi_div_Q * hi_div_Q / 2)
           + normfactor;
}


double find_divergence_min(double beta, double lambda, double **coefficients, double hmin=-100, double precision=1e-4, double maximum=1e10){
    double num, num_q, den;
    num = numerator_av(beta, lambda, hmin, coefficients[1]);
    num_q = numerator_q_sqr(beta, lambda, hmin, coefficients[2]);
    den = denominator(beta, lambda, hmin, coefficients[0]);
    
    while (!(std::isnan(num) || std::isinf(num) || std::isnan(num_q) || std::isinf(num_q) 
             || std::isnan(den) || std::isinf(den) || num > maximum || num_q > maximum
             || den > maximum || num < 0 || num_q < 0 || den < 0)){
        hmin *= 2;
        num = numerator_av(beta, lambda, hmin, coefficients[1]);
        num_q = numerator_q_sqr(beta, lambda, hmin, coefficients[2]);
        den = denominator(beta, lambda, hmin, coefficients[0]);   
    }

    double hmax = 0;
    double h = (hmax + hmin) / 2;
    while (hmax - hmin > precision){
        num = numerator_av(beta, lambda, h, coefficients[1]);
        num_q = numerator_q_sqr(beta, lambda, h, coefficients[2]);
        den = denominator(beta, lambda, h, coefficients[0]); 
        if (std::isnan(num) || std::isinf(num) || std::isnan(num_q) || std::isinf(num_q) 
             || std::isnan(den) || std::isinf(den) || num > maximum || num_q > maximum
             || den > maximum || num < 0 || num_q < 0 || den < 0){
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


double new_averages(long M, double beta, double lambda, Tedge *edges, double tol, 
                    double hmin, double hmax, double **coefficients, double *gamma_vals, 
                    int iter, long sequence[], double damping, 
                    double normfactor=1e-14, double maximum=1e6){
    double delta = 0, delta_av, delta_chi_cav, Q, h, h_div_Q, den, av_new, av_new_not_damp, 
           q_sqr_new, chi_cav_new, var_cav, field_cav;

    long pos;
    for (long e = 0; e < M; e++){
        pos = sequence[e];
        for (int k = 0; k < 2; k++){
            field_cav = field_cav_in(pos, k, edges);
            var_cav = var_cav_in(pos, k, edges);
            if (var_cav >= 0){
                edges[pos].chi_cav_finite[k] = true;   
                Q = sqrt(var_cav);
                h = field_cav * var_cav;
                h_div_Q = h / Q;
                if (h_div_Q > hmax){
                    av_new = damping * h * (1 - 1.0 / beta / h_div_Q / h_div_Q + lambda / h_div_Q / h_div_Q) + 
                             (1 - damping) * edges[pos].cond_av[k];
                    chi_cav_new = damping * var_cav * (1 + 1.0 / beta / h_div_Q / h_div_Q - lambda / h_div_Q / h_div_Q) + 
                                  (1 - damping) * edges[pos].chi_cav[k];
                }else if (h_div_Q < hmin){
                    av_new = damping * lambda * Q / fabs(h_div_Q) + (1 - damping) * edges[pos].cond_av[k];
                    chi_cav_new = damping * var_cav * lambda / h_div_Q / h_div_Q + 
                                  (1 - damping) * edges[pos].chi_cav[k];
                }else if(h_div_Q == 0){
                    av_new = damping * Q * sqrt(2.0 / beta) * gamma_vals[1] / gamma_vals[0] + 
                             (1 - damping) * edges[pos].cond_av[k];
                    chi_cav_new = damping * var_cav * (beta * lambda - 
                                  2 * gamma_vals[1] / gamma_vals[0] * gamma_vals[1] / gamma_vals[0]) + 
                                  (1 - damping) * edges[pos].chi_cav[k];
                }else{
                    den = denominator(beta, lambda, h_div_Q, coefficients[0], normfactor);
                    av_new_not_damp = Q * numerator_av(beta, lambda, h_div_Q, coefficients[1]) / den;
                    av_new = damping * av_new_not_damp + (1 - damping) * edges[pos].cond_av[k];
                    q_sqr_new = var_cav * numerator_q_sqr(beta, lambda, h_div_Q, coefficients[2]) / den;
                    chi_cav_new = damping * beta * (q_sqr_new - av_new_not_damp * av_new_not_damp) + (1 - damping) * edges[pos].chi_cav[k];
                }
            }else{
                edges[pos].chi_cav_finite[k] = false;
                if (field_cav > hmax){
                    av_new = damping * (1 - 1.0 / beta / field_cav + lambda / field_cav) + 
                             (1 - damping) * edges[pos].cond_av[k];
                }else if (field_cav < hmin){
                    av_new = damping * lambda / fabs(field_cav) + (1 - damping) * edges[pos].cond_av[k];
                }else if(field_cav == 0){
                    av_new = damping * sqrt(2.0 / beta) * gamma_vals[1] / gamma_vals[0] + 
                             (1 - damping) * edges[pos].cond_av[k];
                }else{
                    den = denominator(beta, lambda, field_cav, coefficients[0], normfactor);
                    av_new_not_damp = numerator_av(beta, lambda, field_cav, coefficients[1]) / den;
                    av_new = damping * av_new_not_damp + (1 - damping) * edges[pos].cond_av[k];
                }
                chi_cav_new = -damping + (1 - damping) * edges[pos].chi_cav[k];
            }
            
            if (std::isnan(av_new) || std::isinf(av_new) || std::isnan(chi_cav_new) || std::isinf(chi_cav_new)){
                cerr << "Error: averages are nan or inf at site e=" << pos << "  node=" << edges[pos].nodes_in[k] << "   iter=" << iter << endl;
                return sqrt(-1);
            }

            delta_av = fabs(av_new - edges[pos].cond_av[k]);
            delta_chi_cav = fabs(chi_cav_new - edges[pos].chi_cav[k]);
            

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



double average(long N, Tnode *nodes, Tedge *edges, double beta, double lambda, 
               double hmin, double hmax, double **coefficients, double *gamma_vals, 
               double normfactor = 1e-14, double maximum = 1e10){
    double av = 0;
    double h, h_div_Q, Q, den, q_sqr_new;
    for (long i = 0; i < N; i++){
        nodes[i].field = field_in(i, nodes, edges);
        nodes[i].var = var_in(i, nodes, edges);
        if (nodes[i].var >= 0){
            h = nodes[i].field * nodes[i].var;
            Q = sqrt(nodes[i].var);
            h_div_Q = h / Q;
            if (h_div_Q > hmax){
                nodes[i].av = h * (1 - 1.0 / beta / h_div_Q / h_div_Q + lambda / h_div_Q / h_div_Q);
                nodes[i].chi = nodes[i].var;
            }else if (h_div_Q < hmin){
                nodes[i].av = lambda * Q / fabs(h_div_Q);
                nodes[i].chi = nodes[i].var * lambda / h_div_Q / h_div_Q;
            }else if(h_div_Q == 0){
                nodes[i].av = Q * sqrt(2.0 / beta) * gamma_vals[1] / gamma_vals[0];
                nodes[i].chi = nodes[i].var * (beta * lambda - 
                                  2 * gamma_vals[1] / gamma_vals[0] * gamma_vals[1] / gamma_vals[0]);
            }else{
                den = denominator(beta, lambda, h_div_Q, coefficients[0], normfactor);
                nodes[i].av = Q * numerator_av(beta, lambda, h_div_Q, coefficients[1]) / den;
                q_sqr_new = nodes[i].var * numerator_q_sqr(beta, lambda, h_div_Q, coefficients[2]) / den;
                nodes[i].chi = beta * (q_sqr_new - nodes[i].av * nodes[i].av);
            }
        }else{
            h = nodes[i].field;
            if (h > hmax){
                nodes[i].av = (1 - 1.0 / beta / h + lambda / h);
            }else if (h < hmin){
                nodes[i].av = lambda / fabs(h);
            }else if(h == 0){
                nodes[i].av = sqrt(2.0 / beta) * gamma_vals[1] / gamma_vals[0];
            }else{
                den = denominator(beta, lambda, h, coefficients[0], normfactor);
                nodes[i].av = numerator_av(beta, lambda, h, coefficients[1]) / den;
            }
            nodes[i].chi = -1;
        }
        
        av += nodes[i].av;
        
    }
    return av / N;
}




int convergence(long M, double beta, double lambda, Tedge *edges, double tol, 
                int max_iter, bool &divergence, double hmin, double hmax, 
                double **coefficients, double *gamma_vals, long sequence[], 
                double damping, double maximum=1e10, int min_consecutive=5){
    double delta = tol + 1;
    int iter = 0;


    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            edges[e].chi_cav_converged[k] = false;
        }
    }

    int consecutive = 0;
    while (consecutive < min_consecutive && iter < max_iter){
        delta = new_averages(M, beta, lambda, edges, tol, hmin, hmax, coefficients, 
                             gamma_vals, iter, sequence, damping);
        iter++;
        if (std::isinf(delta) || std::isnan(delta) || delta > maximum){
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


size_t GMF_single_try(unsigned long seed_seq, long M, Tedge *edges, double beta, double lambda, double tol,
                     int max_iter, double avn_0, double chi_0, double damping, bool random_init, double dn,
                     double dchi, unsigned long seed_condinit, long sequence[], bool &divergence, int &iter, 
                     double hmin, double hmax, double **coefficients, double *gamma_vals){
    produce_random_seq(seed_seq, M, sequence);
    init_avgs(M, edges, avn_0, chi_0, random_init, dn, dchi, seed_condinit);
    auto start = std::chrono::high_resolution_clock::now();
    iter = convergence(M, beta, lambda, edges, tol, max_iter, divergence, 
                       hmin, hmax, coefficients, gamma_vals, sequence, damping);
    auto end = std::chrono::high_resolution_clock::now();
    size_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return elapsed;
}


void several_seq_GMF(unsigned long seed_graph, unsigned long seed_seq_init, 
                     long N, long M, Tnode *nodes, Tedge *edges, double T, double lambda, double tol,
                     int max_iter, unsigned long num_seq, double tol_fixed_point,
                     double avn_0, double chi_0, double damping, bool print_only_last, bool print_avgs, 
                     char * fileout_base, bool random_init, double dn, double dchi, unsigned long id_0, 
                     int num_init_conds){
    double beta = 1.0 / T;

    double hmax = find_divergence_max(beta, 1 + beta * lambda / 2);
    double **coefficients, *gamma_vals;
    comp_coefficients(beta, lambda, coefficients, gamma_vals);
    double hmin = find_divergence_min(beta, lambda, coefficients);

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
    size_t elapsed = GMF_single_try(seed_seq, M, edges, beta, lambda, tol, max_iter, avn_0, chi_0, damping, 
                                    random_init, dn, dchi, seed_condinit, sequence, divergence, iter, hmin, 
                                    hmax, coefficients, gamma_vals);
    double av = average(N, nodes, edges, beta, lambda, hmin, hmax, coefficients, 
                        gamma_vals);
        
    if (!print_only_last){
        print_results(av, iter, nodes, edges, N, M, seed_graph, seed_seq, seed_condinit, max_iter, divergence, 
                      true, elapsed);
        if (print_avgs){
            sprintf(fileavgs, "%s_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
            print_avgs_to_file(nodes, N, fileavgs);
        }
    }else if(divergence || iter >= max_iter){
        print_results(av, iter, nodes, edges, N, M, seed_graph, seed_seq, seed_condinit, max_iter, divergence, 
                      true, elapsed);
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
            elapsed = GMF_single_try(seed_seq, M, edges, beta, lambda, tol, max_iter, avn_0, chi_0, damping, 
                                    random_init, dn, dchi, seed_condinit, sequence, divergence, iter, hmin, 
                                    hmax, coefficients, gamma_vals);
            av = average(N, nodes, edges, beta, lambda, hmin, hmax, coefficients, 
                         gamma_vals);
            same_fixed_point = compare_fixed_points(nodes, N, tol_fixed_point);
            if (!print_only_last){
                print_results(av, iter, nodes, edges, N, M, seed_graph, seed_seq, seed_condinit, max_iter, 
                              divergence, same_fixed_point, elapsed);
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
                elapsed = GMF_single_try(seed_seq, M, edges, beta, lambda, tol, max_iter, avn_0, chi_0, damping, 
                                    random_init, dn, dchi, seed_condinit, sequence, divergence, iter, hmin, 
                                    hmax, coefficients, gamma_vals);
                av = average(N, nodes, edges, beta, lambda, hmin, hmax, coefficients, 
                             gamma_vals);
                same_fixed_point = compare_fixed_points(nodes, N, tol_fixed_point);
                if (!print_only_last){
                    print_results(av, iter, nodes, edges, N, M, seed_graph, seed_seq, seed_condinit, max_iter, 
                                  divergence, same_fixed_point, elapsed);
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
            print_results(av, iter, nodes, edges, N, M, seed_graph, seed_seq-1, seed_condinit-1, max_iter, 
                              divergence, same_fixed_point, elapsed);
            if (print_avgs){
                sprintf(fileavgs, "%s_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq-1, seed_condinit-1);
                print_avgs_to_file(nodes, N, fileavgs);
            }
        }
    }
    delete [] sequence;
}


#endif