#ifndef __PBMF_ANSATZ_CONVERGENCE_H_INCLUDED__
#define __PBMF_ANSATZ_CONVERGENCE_H_INCLUDED__

#include "PBMF_ansatz_common.h"
#include <chrono>

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
    if (std::isnan(gsl_sf_gamma((1 + beta * lambda) / 2)) || std::isinf(gsl_sf_gamma((1 + beta * lambda) / 2)) || 
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
    
    while (!(std::isnan(num) || std::isinf(num) || std::isnan(den) || std::isinf(den) || 
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
        if (std::isnan(num) || std::isinf(num) || std::isnan(den) || std::isinf(den) || 
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






double field_cav_in_from_single_avgs(long e, int k, Tnode *nodes, Tedge *edges, double n_neigh){
    double sum = 0;
    long edge_neigh;
    int pos_there;
    for (long j = 0; j < edges[e].edges_except[k].size(); j++){
        edge_neigh = edges[e].edges_except[k][j];
        pos_there = edges[e].pos_there[k][j];
        sum += edges[edge_neigh].links[pos_there] * nodes[edges[edge_neigh].nodes_in[1 - pos_there]].av_abundance;
    }
    sum += edges[e].links[k] * n_neigh;
    return 1 - sum;
}


void cond_av_from_single_avgs(double beta, double lambda, Tnode *nodes, Tedge *edges, long edge_index, int k,
                              vector <double> n_grid, double hmin, double hmax, double **coefficients, double *gamma_vals, 
                              double normfactor = 1e-14){
    double field_cav, cond_av, den;
    for (long n_index = 0; n_index < n_grid.size(); n_index++){
        field_cav = field_cav_in_from_single_avgs(edge_index, k, nodes, edges, n_grid[n_index]);
        if (field_cav > hmax){
            edges[edge_index].cond_av[k][n_index] = (1 - 1.0 / beta / field_cav + lambda / field_cav);
        }else if (field_cav < hmin){
            edges[edge_index].cond_av[k][n_index] = lambda / fabs(field_cav);
        }else if(field_cav == 0){
            edges[edge_index].cond_av[k][n_index] = sqrt(2.0 / beta) * gamma_vals[1] / gamma_vals[0];
        }else{
            den = denominator(beta, lambda, field_cav, coefficients[0], normfactor);
            edges[edge_index].cond_av[k][n_index] = numerator_av(beta, lambda, field_cav, coefficients[1]) / den;           
        }
    }
}

void update_integrated_cond_av(vector <double> cond_av, vector <double> &integrated_cond_av, vector <double> n_grid, long M){
    for (int index_n = 1; index_n < n_grid.size(); index_n++){
        integrated_cond_av[index_n] = integrated_cond_av[index_n - 1] + (cond_av[index_n] + cond_av[index_n - 1]) * (n_grid[index_n] - n_grid[index_n - 1]) / 2;
    }
}


void init_cond_av_single_avgs(long M, double beta, double lambda, Tnode *nodes, Tedge *edges,
                   vector <double> n_grid, double hmin, double hmax, double **coefficients, double *gamma_vals){
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            cond_av_from_single_avgs(beta, lambda, nodes, edges, e, k, n_grid, hmin, hmax, 
                                     coefficients, gamma_vals);
            update_integrated_cond_av(edges[e].cond_av[k], edges[e].integrated_cond_av[k], n_grid, M);
        }
    }
}








double sum_over_neighs_except(long e, int k, Tedge *edges, double n_neigh, int n_index_node){
    double sum = 0;
    long edge_neigh;
    int pos_there;
    for (long j = 0; j < edges[e].edges_except[k].size(); j++){
        edge_neigh = edges[e].edges_except[k][j];
        pos_there = edges[e].pos_there[k][j];
        sum += edges[edge_neigh].links[pos_there] * edges[edge_neigh].integrated_cond_av[1 - pos_there][n_index_node];
    }
    return sum;
}


double integrate(vector <double> fixed_integrand, double beta, double lambda, int p, vector <double> n_grid, 
                 Tedge *edges, double n_neigh, long edge_index, int k, vector <double> simpson_weights){
    double sum_cav = sum_over_neighs_except(edge_index, k, edges, n_neigh, 1) + edges[edge_index].links[k] * n_neigh * n_grid[1];
    double integral = fixed_integrand[0] * n_grid[1] / (beta * lambda + p) * exp(-beta * sum_cav);
    for (int n_index = 1; n_index < n_grid.size(); n_index++){
        sum_cav = sum_over_neighs_except(edge_index, k, edges, n_neigh, n_index) + edges[edge_index].links[k] * n_neigh * n_grid[n_index];
        integral += simpson_weights[n_index - 1] * fixed_integrand[n_index - 1] * exp(-beta * sum_cav);
    }
    return integral;
}


double update_cond_avgs(Tedge *edges, vector <double> n_grid, vector <double> simpson_weights, vector <double> fixed_integrand_num,
                      vector <double> fixed_integrand_den, double beta, double lambda, long M, double dn, double damping, double tol){
    double num, den, cond_av_new, variation;
    double max_variation = 0;
    for (long e=0; e < M; e++){
        for (int k = 0; k < 2; k++){
            variation = 0;
            for (int n_index = 0; n_index < n_grid.size(); n_index++){
                num = integrate(fixed_integrand_num, beta, lambda, 1, n_grid, edges, n_grid[n_index], 
                                e, k, simpson_weights);
                den = integrate(fixed_integrand_den, beta, lambda, 0, n_grid, edges, n_grid[n_index], 
                                e, k, simpson_weights);
                cond_av_new = damping * num / den + (1 - damping) * edges[e].cond_av[k][n_index];
                variation += fabs(cond_av_new - edges[e].cond_av[k][n_index]);
                edges[e].cond_av[k][n_index] = cond_av_new;
            }
            update_integrated_cond_av(edges[e].cond_av[k], edges[e].integrated_cond_av[k], n_grid, M);
            variation *= dn;
            if (variation > max_variation){
                max_variation = variation;
            }
            if (variation > tol){
                edges[e].converged[k] = true;
            }
        }
    }
    return max_variation;
}


int convergence(Tedge *edges, vector <double> n_grid, vector <double> simpson_weights, vector <double> fixed_integrand_num,
                vector <double> fixed_integrand_den, double beta, double lambda, long M, double dn, double damping, double tol,
                int max_iter, bool &divergence, double min_consecutive=5, double maximum=1e10){
    double delta = tol + 1;
    int iter = 0;


    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            edges[e].converged[k] = false;
        }
    }

    int consecutive = 0;
    while (consecutive < min_consecutive && iter < max_iter){
        delta = update_cond_avgs(edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta,
                                 lambda, M, dn, damping, tol);
        iter++;
        cerr << "Iteration " << iter << ", delta = " << delta << endl;
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



double sum_over_neighs(long node, Tnode *nodes, Tedge *edges, int n_index_node){
    double sum = 0;
    long edge_neigh;
    int pos_there;
    for (long j = 0; j < nodes[node].edges_in.size(); j++){
        edge_neigh = nodes[node].edges_in[j];
        pos_there = nodes[node].pos_there[j];
        sum += edges[edge_neigh].links[pos_there] * edges[edge_neigh].integrated_cond_av[1 - pos_there][n_index_node];
    }
    return sum;
}


void compute_averages(Tnode *nodes, Tedge *edges, vector <double> n_grid, vector <double> simpson_weights, 
                      vector <double> fixed_integrand_num, vector <double> fixed_integrand_den, double beta, 
                      double lambda, long N, double &av_n_graph, double &av_var_n_graph,
                      double &error_av_n_graph, double &error_var_n_graph){
    double num_P, av_sqr_site, sum_neighs;
    av_n_graph = 0;
    av_var_n_graph = 0;
    double av_sqr_av_n_graph = 0;
    double av_sqr_var_n_graph = 0;
    for (long i=0; i < N; i++){
        sum_neighs = sum_over_neighs(i, nodes, edges, 1);
        nodes[i].av_abundance = fixed_integrand_num[0] * n_grid[1] / (beta * lambda + 1) * exp(-beta * sum_neighs);
        av_sqr_site = fixed_integrand_num[0] * n_grid[1] * n_grid[1] / (beta * lambda + 2) * exp(-beta * sum_neighs);
        nodes[i].normalization_Psingle = fixed_integrand_den[0] * n_grid[1] / (beta * lambda) * exp(-beta * sum_neighs);
        for (int n_index = 1; n_index < n_grid.size(); n_index++){
            sum_neighs = sum_over_neighs(i, nodes, edges, n_index);
            num_P = fixed_integrand_den[n_index - 1] * exp(-beta * sum_neighs);
            nodes[i].av_abundance += simpson_weights[n_index - 1] * num_P * n_grid[n_index];
            av_sqr_site += simpson_weights[n_index - 1] * num_P * n_grid[n_index] * n_grid[n_index];
            nodes[i].normalization_Psingle += simpson_weights[n_index - 1] * num_P;
        }

        nodes[i].av_abundance /= nodes[i].normalization_Psingle;
        av_sqr_site /= nodes[i].normalization_Psingle;
        nodes[i].var_abundance = av_sqr_site - nodes[i].av_abundance * nodes[i].av_abundance;
        av_n_graph += nodes[i].av_abundance;
        av_var_n_graph += nodes[i].var_abundance;
        av_sqr_av_n_graph += nodes[i].av_abundance * nodes[i].av_abundance;
        av_sqr_var_n_graph += nodes[i].var_abundance * nodes[i].var_abundance;
    }
    av_n_graph /= N;
    av_var_n_graph /= N;
    av_sqr_av_n_graph /= N;
    av_sqr_var_n_graph /= N;
    error_av_n_graph = sqrt((av_sqr_av_n_graph - av_n_graph * av_n_graph) / N);
    error_var_n_graph = sqrt((av_sqr_var_n_graph - av_var_n_graph * av_var_n_graph) / N);
}



void compute_responses(Tnode *nodes, Tedge *edges, vector <double> n_grid, double beta, long M){
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            edges[e].response_around_zero[k] = -(edges[e].cond_av[k][1] - edges[e].cond_av[k][0]) / beta / n_grid[1];
            for (int n_index = 1; n_index < n_grid.size(); n_index++){
                if(nodes[edges[e].nodes_in[k]].av_abundance >= n_grid[n_index - 1] && nodes[edges[e].nodes_in[k]].av_abundance < n_grid[n_index]){
                    edges[e].response_around_average[k] = -(edges[e].cond_av[k][n_index] - edges[e].cond_av[k][n_index - 1]) / beta / (n_grid[n_index] - n_grid[n_index - 1]);
                    break;
                }
            }
        }
    }
}


void compute_distributions(Tnode *nodes, Tedge *edges, vector <double> n_grid, vector <double> fixed_integrand_den, double beta, 
                           long N){
    double sum_neighs;
    for (long i=0; i < N; i++){
        nodes[i].Psingle = vector <double> (n_grid.size() - 1, 0);
        sum_neighs = sum_over_neighs(i, nodes, edges, 1);
        for (int n_index = 1; n_index < n_grid.size(); n_index++){
            sum_neighs = sum_over_neighs(i, nodes, edges, n_index);
            nodes[i].Psingle[n_index - 1] = fixed_integrand_den[n_index - 1] * exp(-beta * sum_neighs) / nodes[i].normalization_Psingle;
        }
    }
}


size_t PBMF_ansatz_single_try(Tnode *nodes, Tedge *edges, vector <double> n_grid, vector <double> simpson_weights, vector <double> fixed_integrand_num,
                              vector <double> fixed_integrand_den, double beta, double lambda, long N, long M, double dn, double damping, 
                              double tol, int max_iter, long sequence[], unsigned long seed_seq, double avn_0, bool random_init, 
                              double std_n0, unsigned long seed_condinit, int &iter, double hmin, double hmax, double **coefficients,
                              double *gamma_vals, bool &divergence){
    produce_random_seq(seed_seq, M, sequence);
    init_avgs(N, nodes, avn_0, random_init, std_n0, seed_condinit);
    init_cond_av_single_avgs(M, beta, lambda, nodes, edges, n_grid, hmin, hmax, coefficients, gamma_vals);
    auto start = std::chrono::high_resolution_clock::now();
    iter = convergence(edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, 
                       M, dn, damping, tol, max_iter, divergence);
    auto end = std::chrono::high_resolution_clock::now();
    size_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return elapsed;
}



void several_seq_PBMF(unsigned long seed_graph, unsigned long seed_seq_init, 
                      long N, long M, Tnode *nodes, Tedge *edges, double T, double lambda, double tol,
                      int max_iter, unsigned long num_seq, double tol_fixed_point,
                      double avn_0, double damping, bool print_only_last, bool print_avgs, 
                      bool print_responses, bool print_distributions, char * fileout_base, 
                      bool random_init, double std_n0, unsigned long id_0, int num_init_conds,
                      double n1, double dn, double nmax){
    double beta = 1.0 / T;

    double hmax = find_divergence_max(beta, 1 + beta * lambda / 2);
    double **coefficients, *gamma_vals;
    comp_coefficients(beta, lambda, coefficients, gamma_vals);
    double hmin = find_divergence_min(beta, lambda, coefficients);

    long *sequence;
    sequence = new long[M];
    bool divergence;

    char fileavgs[300];
    char fileresponses[300];
    char filedistributions[300];
    
    
    divergence = false;
    unsigned long seed_seq, seed_condinit;
    bool make_other_tries;
    int iter;
    bool same_fixed_point = true;
    double av_n_graph, av_var_n_graph, error_av_n_graph, error_var_n_graph;
    
    vector <double> n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den;
    init_auxiliary_vectors(n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, n1, dn, nmax);
    init_cond_av(M, edges, n_grid.size());

    seed_seq = seed_seq_init;
    seed_condinit = id_0;
    size_t elapsed = PBMF_ansatz_single_try(nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, 
                                            N, M, dn, damping, tol, max_iter, sequence, seed_seq, avn_0, random_init, std_n0, 
                                            seed_condinit, iter, hmin, hmax, coefficients, gamma_vals, divergence);
    compute_averages(nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, 
                     lambda, N, av_n_graph, av_var_n_graph, error_av_n_graph, error_var_n_graph);
        
    if (!print_only_last){
        print_results(av_n_graph, error_av_n_graph, av_var_n_graph, error_av_n_graph, iter, nodes, edges, N, M, seed_graph, seed_seq,
                     seed_condinit, max_iter, divergence, true, elapsed, lambda);
        if (print_avgs){
            sprintf(fileavgs, "%s_average_abundances_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
            print_node_avgs_to_file(nodes, N, fileavgs);
        }
        if (print_responses){
            sprintf(fileresponses, "%s_responses_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
            compute_responses(nodes, edges, n_grid, beta, M);
            print_responses_to_file(edges, M, fileresponses);
        }
        if (print_distributions){
            sprintf(filedistributions, "%s_distributions_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
            compute_distributions(nodes, edges, n_grid, fixed_integrand_den, beta, N);
            print_distributions_to_file(nodes, n_grid, N, filedistributions);
        }
    }else if(divergence || iter >= max_iter){
        print_results(av_n_graph, error_av_n_graph, av_var_n_graph, error_av_n_graph, iter, nodes, edges, N, M, seed_graph, seed_seq,
                     seed_condinit, max_iter, divergence, true, elapsed, lambda);
        if (print_avgs){
            sprintf(fileavgs, "%s_average_abundances_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
            print_node_avgs_to_file(nodes, N, fileavgs);
        }
        if (print_responses){
            sprintf(fileresponses, "%s_responses_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
            compute_responses(nodes, edges, n_grid, beta, M);
            print_responses_to_file(edges, M, fileresponses);
        }
        if (print_distributions){
            sprintf(filedistributions, "%s_distributions_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
            compute_distributions(nodes, edges, n_grid, fixed_integrand_den, beta, N);
            print_distributions_to_file(nodes, n_grid, N, filedistributions);
        }
    }

    make_other_tries = !print_only_last || (!divergence && iter < max_iter);
    
    if (make_other_tries){
        set_av_prev(nodes, N);
        bool cond = true;

        seed_seq = seed_seq_init + 1;
        while (seed_seq < seed_seq_init + num_seq && cond){
            elapsed = PBMF_ansatz_single_try(nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, 
                                             N, M, dn, damping, tol, max_iter, sequence, seed_seq, avn_0, random_init, std_n0, 
                                             seed_condinit, iter, hmin, hmax, coefficients, gamma_vals, divergence);
            compute_averages(nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, 
                             lambda, N, av_n_graph, av_var_n_graph, error_av_n_graph, error_var_n_graph);
            same_fixed_point = compare_fixed_points(nodes, N, tol_fixed_point);
            if (!print_only_last){
                print_results(av_n_graph, error_av_n_graph, av_var_n_graph, error_av_n_graph, iter, nodes, edges, N, M, seed_graph, seed_seq,
                     seed_condinit, max_iter, divergence, same_fixed_point, elapsed, lambda);
                if (print_avgs){
                    sprintf(fileavgs, "%s_average_abundances_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
                    print_node_avgs_to_file(nodes, N, fileavgs);
                }
                if (print_responses){
                    sprintf(fileresponses, "%s_responses_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
                    compute_responses(nodes, edges, n_grid, beta, M);
                    print_responses_to_file(edges, M, fileresponses);
                }
                if (print_distributions){
                    sprintf(filedistributions, "%s_distributions_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
                    compute_distributions(nodes, edges, n_grid, fixed_integrand_den, beta, N);
                    print_distributions_to_file(nodes, n_grid, N, filedistributions);
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
                elapsed = PBMF_ansatz_single_try(nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, 
                                                N, M, dn, damping, tol, max_iter, sequence, seed_seq, avn_0, random_init, std_n0, seed_condinit, 
                                                iter, hmin, hmax, coefficients, gamma_vals, divergence);
                compute_averages(nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, 
                                lambda, N, av_n_graph, av_var_n_graph, error_av_n_graph, error_var_n_graph);
                same_fixed_point = compare_fixed_points(nodes, N, tol_fixed_point);
                if (!print_only_last){
                    print_results(av_n_graph, error_av_n_graph, av_var_n_graph, error_av_n_graph, iter, nodes, edges, N, M, seed_graph, seed_seq,
                     seed_condinit, max_iter, divergence, same_fixed_point, elapsed, lambda);
                    if (print_avgs){
                        sprintf(fileavgs, "%s_average_abundances_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
                        print_node_avgs_to_file(nodes, N, fileavgs);
                    }
                    if (print_responses){
                        sprintf(fileresponses, "%s_responses_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
                        compute_responses(nodes, edges, n_grid, beta, M);
                        print_responses_to_file(edges, M, fileresponses);
                    }
                    if (print_distributions){
                        sprintf(filedistributions, "%s_distributions_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq, seed_condinit);
                        compute_distributions(nodes, edges, n_grid, fixed_integrand_den, beta, N);
                        print_distributions_to_file(nodes, n_grid, N, filedistributions);
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
            print_results(av_n_graph, error_av_n_graph, av_var_n_graph, error_av_n_graph, iter, nodes, edges, N, M, seed_graph, seed_seq-1,
                     seed_condinit-1, max_iter, divergence, true, elapsed, lambda);
            if (print_avgs){
                sprintf(fileavgs, "%s_average_abundances_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq-1, seed_condinit-1);
                print_node_avgs_to_file(nodes, N, fileavgs);
            }
            if (print_responses){
                sprintf(fileresponses, "%s_responses_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq-1, seed_condinit-1);
                compute_responses(nodes, edges, n_grid, beta, M);
                print_responses_to_file(edges, M, fileresponses);
            }
            if (print_distributions){
                sprintf(filedistributions, "%s_distributions_seedseq_%li_seedinit_%li.txt", fileout_base, seed_seq-1, seed_condinit-1);
                compute_distributions(nodes, edges, n_grid, fixed_integrand_den, beta, N);
                print_distributions_to_file(nodes, n_grid, N, filedistributions);
            }
        }
    }
    delete [] sequence;
}



#endif