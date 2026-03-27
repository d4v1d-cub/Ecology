#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include "math.h"
#include <chrono>

using namespace std;

void init_ran(gsl_rng * &r, unsigned long s){
    const gsl_rng_type * T;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, s);
}


typedef struct{
    vector <long> edges_in; // edges that contain the node
    vector <int> pos_there; // position occupied by the node in those edges
    double av_abundance; // average value of n in that node
}Tnode;




typedef struct{
    vector <long> nodes_in; // nodes inside the edge. nodes_in[i], with i={0, 1}.
    vector <double> links; // links[i], with i={0, 1}. links[i] is the one pointing to the variable in nodes_in[i]
    vector < vector <double> > cond_av; // cond_av[i], with i = {0, 1}. cond_av[i] is the average of the variable in nodes_in[i]
    // conditioned to the variable in nodes_in[1 - i] being fixed. cond_av[i] is computed as an integral over the conditional distribution.
    vector < vector <double> > integrated_cond_av; // integrated_cond_av[i][n_index], with i = {0, 1}, is the integral of cond_av[i] 
    // from n0 to n=n_grid[n_index]
    vector < vector <long> > edges_except; // edges that contain the node in nodes_in[i] excepting this edge
    vector < vector <int> > pos_there; // position occupied by the node in nodes_in[i] in those edges
    vector <int> edge_index; // position of the edge in the list of edges that contain the node
    vector <bool> converged; // converged[i], with i = {0, 1}. converged[i] is true if the conditional average cond_av[i] arrived to convergence.
}Tedge;


void fill_except(Tnode *nodes, Tedge *edges, long M){
    for (long e = 0; e < M; e++){
        edges[e].edges_except = vector < vector <long> > (2, vector <long> ());
        edges[e].pos_there = vector < vector <int> > (2, vector <int> ());
        for (int i = 0; i < 2; i++){
            for (int k = 0; k < edges[e].edge_index[i]; k++){
                edges[e].edges_except[i].push_back(nodes[edges[e].nodes_in[i]].edges_in[k]);
                edges[e].pos_there[i].push_back(nodes[edges[e].nodes_in[i]].pos_there[k]);
            }
            for (int k = edges[e].edge_index[i] + 1; k < nodes[edges[e].nodes_in[i]].edges_in.size(); k++){
                edges[e].edges_except[i].push_back(nodes[edges[e].nodes_in[i]].edges_in[k]);
                edges[e].pos_there[i].push_back(nodes[edges[e].nodes_in[i]].pos_there[k]);
            }
        }
    }
}


void init_graph_from_input(Tnode *&nodes, Tedge *&edges, long &N, long &M){
    scanf("%ld %ld", &N, &M);
    nodes = new Tnode[N];
    edges = new Tedge[M];
    long i, j;
    double aij, aji;
    for (long e = 0; e < M; e++){
        scanf("%ld %ld %lf %lf", &i, &j, &aij, &aji);
        edges[e].nodes_in.push_back(i);
        edges[e].nodes_in.push_back(j);
        edges[e].links.push_back(aji);
        edges[e].links.push_back(aij);

        edges[e].edge_index = vector <int> (2);
        edges[e].edge_index[0] = nodes[i].edges_in.size();
        edges[e].edge_index[1] = nodes[j].edges_in.size();
        

        nodes[i].edges_in.push_back(e);
        nodes[j].edges_in.push_back(e);
        nodes[i].pos_there.push_back(0);
        nodes[j].pos_there.push_back(1);
    }

    fill_except(nodes, edges, M);
}


void init_nodes(Tnode *nodes, long N){
    for (long i = 0; i < N; i++){
        nodes[i].edges_in = vector <long> ();
        nodes[i].pos_there = vector <int> ();
    }
}

void init_graph_inside_RRG(Tnode *&nodes, Tedge *&edges, long N, int c, double eps,
                           double mu, double sigma, gsl_rng * r){
    // eps is the degree of symmetry of the graph
    if (N * c % 2 != 0){
        cout << "N*c must be even to create a random regular graph" << endl;
        exit(1);
    }else{
        long M = N * c / 2;
        long pos_i, pos_j, i, j;
        double aij, aji;
        nodes = new Tnode[N];
        edges = new Tedge[M];

        for (long e = 0; e < M; e++){
            edges[e].nodes_in = vector <long> (2);
            edges[e].links = vector <double> (2);
            edges[e].edge_index = vector <int> (2);
        }

        bool success = false;
        while (!success){
            init_nodes(nodes, N);

            vector < long > copies = vector < long > (c * N);
            for (long i = 0; i < N; i++){
                for (int k = 0; k < c; k++){
                    copies[i * c + k] = i;
                }
            }

            for (long e = 0; e < M - 1; e++){
                pos_i = gsl_rng_uniform_int(r, copies.size());
                i = copies[pos_i];
                copies.erase(copies.begin() + pos_i);
                pos_j = gsl_rng_uniform_int(r, copies.size());
                j = copies[pos_j];
                while (j == i){
                    pos_j = gsl_rng_uniform_int(r, copies.size());
                    j = copies[pos_j];
                }
                copies.erase(copies.begin() + pos_j);

                edges[e].nodes_in.push_back(i);
                edges[e].nodes_in.push_back(j);
                
                edges[e].edge_index[0] = nodes[i].edges_in.size();
                edges[e].edge_index[1] = nodes[j].edges_in.size();

                aij = mu + gsl_ran_gaussian(r, sigma);
                if (gsl_rng_uniform_pos(r) < eps){
                    aji = aij;
                }else{
                    aji = mu + gsl_ran_gaussian(r, sigma);
                }
                edges[e].links.push_back(aji);
                edges[e].links.push_back(aij);

                nodes[i].edges_in.push_back(e);
                nodes[j].edges_in.push_back(e);
                nodes[i].pos_there.push_back(0);
                nodes[j].pos_there.push_back(1);
            }
            
            pos_i = 0;
            i = copies[pos_i];
            pos_j = 1;
            j = copies[pos_j];
            if (i != j){
                success = true;
                edges[M - 1].nodes_in.push_back(i);
                edges[M - 1].nodes_in.push_back(j);
                
                edges[M - 1].edge_index[0] = nodes[i].edges_in.size();
                edges[M - 1].edge_index[1] = nodes[j].edges_in.size();

                aij = mu + gsl_ran_gaussian(r, sigma);
                if (gsl_rng_uniform_pos(r) < eps){
                    aji = aij;
                }else{
                    aji = mu + gsl_ran_gaussian(r, sigma);
                }
                edges[M - 1].links.push_back(aji);
                edges[M - 1].links.push_back(aij);

                nodes[i].edges_in.push_back(M - 1);
                nodes[j].edges_in.push_back(M - 1);
                nodes[i].pos_there.push_back(0);
                nodes[j].pos_there.push_back(1);
            }
        }
        fill_except(nodes, edges, M);
    }
}


vector <double> init_grid(double n0, double n1, double dn, double nmax){
    vector <double> n_grid;
    n_grid.push_back(n0);
    n_grid.push_back(n1);
    for (double n = n1 + dn; n <= nmax; n += dn){
        n_grid.push_back(n);
    }
    return n_grid;
}


vector <double> compute_simpson_weights(int npoints, double dn){
    vector <double> simpson_weights = vector <double> (npoints, dn / 3);
    for (int i = 1; i < npoints - 1; i+=2){
        simpson_weights[i] *= 4;
    }
    for (int i = 2; i < npoints - 1; i+=2){
        simpson_weights[i] *= 2;
    }
}


vector <double> compute_fixed_integrand(double beta, double lambda, int p, vector <double> n_grid){
    vector <double> integrand = vector <double> (n_grid.size() - 1, 0);
    for (int i = 1; i < n_grid.size(); i++){
        integrand[i - 1] = pow(n_grid[i], beta * lambda - 1 + p) * exp(-beta * (n_grid[i] * n_grid[i] - 2 * n_grid[i]) / 2);
    }
    return integrand;
}

void init_cond_av(long M, Tedge *edges, long npoints){
    for (long e = 0; e < M; e++){
        edges[e].cond_av = vector < vector <double> > (2, vector <double> (npoints, 0));
        edges[e].integrated_cond_av = vector < vector <double> > (2, vector <double> (npoints, 0));
    }
}


void init_auxiliary_vectors(vector <double> &n_grid, vector <double> &simpson_weights, vector <double> &fixed_integrand_num, 
                            vector <double> &fixed_integrand_den, double beta, double lambda, double n0, double n1, double dn,
                            double nmax){
    n_grid = init_grid(n0, n1, dn, nmax);
    simpson_weights = compute_simpson_weights(n_grid.size() - 1, dn);
    fixed_integrand_num = compute_fixed_integrand(beta, lambda, 1, n_grid);
    fixed_integrand_den = compute_fixed_integrand(beta, lambda, 0, n_grid);
}



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


void init_avgs(long N, Tnode *nodes, double avn_0, bool random_init, double dn, unsigned long id_0){
    if (random_init){
        gsl_rng * r;
        init_ran(r, id_0);
        double n_i;
        for (long i = 0; i < N; i++){
            n_i = avn_0 - dn + 2 * dn * gsl_rng_uniform(r);
            if (n_i < 0){
                n_i = 0;
            }
            nodes[i].av_abundance = n_i;
        }
        gsl_rng_free(r);
    }else{
        for (long i = 0; i < N; i++){
            nodes[i].av_abundance = avn_0;
        }
    }
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




void init_cond_av_single_avgs(long M, double beta, double lambda, Tnode *nodes, Tedge *edges,
                   vector <double> n_grid, double hmin, double hmax, double **coefficients, double *gamma_vals){
    for (long e = 0; e < M; e++){
        edges[e].cond_av = vector < vector <double> > (2, vector <double> (n_grid.size(), 0));
        edges[e].integrated_cond_av = vector < vector <double> > (2, vector <double> (n_grid.size(), 0));
        for (int k = 0; k < 2; k++){
            cond_av_from_single_avgs(beta, lambda, nodes, edges, e, k, n_grid, hmin, hmax, 
                                     coefficients, gamma_vals);
        }
    }
}




void update_integrals_cond_av(Tedge *edges, vector <double> n_grid, long M){
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            for (int index_n = 1; index_n < n_grid.size(); index_n++){
                edges[e].integrated_cond_av[k][index_n] = (edges[e].cond_av[k][index_n] + edges[e].cond_av[k][index_n - 1]) * 
                                                          (n_grid[index_n] - n_grid[index_n - 1]) / 2;
            }
        }
    }
}


// vector <double> derivative(vector <double> vals, double dx){
//     vector <double> der = vector <double> (vals.size(), 0);
//     for (long i = 0; i < vals.size() - 1; i++){
//         der[i] = (vals[i + 1] - vals[i]) / dx;
//     }
//     der[vals.size() - 1] = der[vals.size() - 2];
//     return der;
// }




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
        sum_cav = sum_over_neighs_except(edge_index, k, edges, n_neigh, n_index) + edges[edge_index].links[k] * n_neigh * n_grid[1];
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
                den = integrate(fixed_integrand_num, beta, lambda, 0, n_grid, edges, n_grid[n_index], 
                                e, k, simpson_weights);
                cond_av_new = damping * num / den + (1 - damping) * edges[e].cond_av[k][n_index];
                variation += fabs(cond_av_new - edges[e].cond_av[k][n_index]);
                edges[e].cond_av[k][n_index] = cond_av_new;
            }
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


void produce_random_seq(unsigned long seed_seq, long M, long sequence[]){
    gsl_rng * r;
    init_ran(r, seed_seq);
    vector <long> elements(M);
    for (long i = 0; i < M; i++){
        elements[i] = i;
    }
    long pos;
    for (long i = 0; i < M; i++){
        pos = gsl_rng_uniform_int(r, M - i);
        sequence[i] = elements[pos];
        elements.erase(elements.begin() + pos);
    }
    gsl_rng_free(r);
}


size_t PBMF_ansatz_single_try(Tnode *nodes, Tedge *edges, vector <double> n_grid, vector <double> simpson_weights, vector <double> fixed_integrand_num,
                              vector <double> fixed_integrand_den, double beta, double lambda, long M, double dn, double damping, 
                              double tol, int max_iter, long sequence[], unsigned long seed_seq, double avn_0, bool random_init, 
                              double std_n_0, unsigned long seed_condinit, int &iter, double hmin, double hmax, double **coefficients,
                              double *gamma_vals, bool &divergence){
    produce_random_seq(seed_seq, M, sequence);
    init_avgs(M, nodes, avn_0, random_init, std_n_0, seed_condinit);
    init_cond_av_single_avgs(M, beta, lambda, nodes, edges, n_grid, hmin, hmax, coefficients, gamma_vals);
    auto start = std::chrono::high_resolution_clock::now();
    iter = convergence(edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, 
                       M, dn, damping, tol, max_iter, divergence);
    auto end = std::chrono::high_resolution_clock::now();
    size_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return elapsed;
}


int main(int argc, char *argv[]) {
    
    
    return 0;
}