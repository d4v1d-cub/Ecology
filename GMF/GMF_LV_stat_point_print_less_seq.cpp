#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include "math.h"
#include <cmath>

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
    double av; // average value of n in that node
    double chi; // beta * (q_sqr - av^2) in that node
    double var; // variance of the perturberd Gaussian in that node
    double field; // field in that node
    double av_prev_fixed_point;
    double chi_prev_fixed_point; // previous value of av and chi in the fixed point
}Tnode;




typedef struct{
    long nodes_in[2]; // nodes inside the edge. nodes_in[i], with i={0, 1}.
    double links[2]; // links[i], with i={0, 1}. links[i] is the one pointing to the variable in nodes[i]
    double cond_av[2]; // cond_av[i], with i={0,1}, is the average of nodes_in[i] given that node j is zero
    vector < vector <long> > edges_except; // edges that contain the node in nodes[i] excepting this edge
    vector < vector <int> > pos_there; // position occupied by the node in nodes[i] in those edges
    int edge_index[2]; // position of the edge in the list of edges that contain the node
    double chi_cav[2]; // chi_cav[i], with i={0, 1}. chi_cav[i] = beta(cond_q_sqr_i - cond_av_i^2)
    bool chi_cav_converged[2];
    double var_cav[2];
    double fields_cav[2]; // fields_cav[i], with i={0, 1}, is the field in nodes_in[i] given that node j is zero
    bool var_cav_positive[2];
    bool converged[2]; // converged[i], with i={0, 1}, is true if the averages in nodes_in[i] have converged
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
    double aij, aji; // aij is the coupling that node j sees from node i
    for (long e = 0; e < M; e++){
        scanf("%ld %ld %lf %lf", &i, &j, &aij, &aji);
        edges[e].nodes_in[0] = i;
        edges[e].nodes_in[1] = j;
        edges[e].links[0] = aji;
        edges[e].links[1] = aij;

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

long init_graph_inside_RRG(Tnode *&nodes, Tedge *&edges, long N, int c, double eps,
                           double mu, double sigma, gsl_rng * r){
    // eps is the degree of symmetry of the graph
    if (N * c % 2 != 0){
        cout << "N*c must be even to create a random regular graph" << endl;
        exit(1);
    }else{
        long M = N * c / 2;
        long pos_i, pos_j, i, j;
        double aij, aji; // aij is the coupling that node j sees from node i
        nodes = new Tnode[N];
        edges = new Tedge[M];

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

                edges[e].nodes_in[0] = i;
                edges[e].nodes_in[1] = j;
                
                edges[e].edge_index[0] = nodes[i].edges_in.size();
                edges[e].edge_index[1] = nodes[j].edges_in.size();

                aij = mu + gsl_ran_gaussian(r, sigma);
                if (gsl_rng_uniform_pos(r) < eps){
                    aji = aij;
                }else{
                    aji = mu + gsl_ran_gaussian(r, sigma);
                }
                edges[e].links[0] = aji;
                edges[e].links[1] = aij;

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
                edges[M - 1].nodes_in[0] = i;
                edges[M - 1].nodes_in[1] = j;
                
                edges[M - 1].edge_index[0] = nodes[i].edges_in.size();
                edges[M - 1].edge_index[1] = nodes[j].edges_in.size();

                aij = mu + gsl_ran_gaussian(r, sigma);
                if (gsl_rng_uniform_pos(r) < eps){
                    aji = aij;
                }else{
                    aji = mu + gsl_ran_gaussian(r, sigma);
                }
                edges[M - 1].links[0] = aji;
                edges[M - 1].links[1] = aij;

                nodes[i].edges_in.push_back(M - 1);
                nodes[j].edges_in.push_back(M - 1);
                nodes[i].pos_there.push_back(0);
                nodes[j].pos_there.push_back(1);
            }
        }
        fill_except(nodes, edges, M);
        return M;
    }
}


void init_avgs(long M, Tedge *edges, double avn_0){
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            edges[e].cond_av[k] = avn_0;
            edges[e].chi_cav[k] = 0;
            edges[e].var_cav[k] = 1;
        }
    }
}


double field_cav_in(long e, int k, Tedge *edges){
    double sum = 0;
    long edge_neigh;
    int pos_there;
    for (long j = 0; j < edges[e].edges_except[k].size(); j++){
        edge_neigh = edges[e].edges_except[k][j];
        pos_there = edges[e].pos_there[k][j];
        sum += edges[edge_neigh].links[pos_there] * edges[edge_neigh].cond_av[1 - pos_there];
    }
    return 1 - sum;
}


double var_cav_in(long e, int k, Tedge *edges){
    double sum = 0;
    long edge_neigh;
    int pos_there;
    for (long j = 0; j < edges[e].edges_except.size(); j++){
        edge_neigh = edges[e].edges_except[k][j];
        pos_there = edges[e].pos_there[k][j];
        sum += edges[edge_neigh].links[0] * edges[edge_neigh].links[1] * 
               edges[edge_neigh].chi_cav[1 - pos_there];
    }
    return 1.0 / (1 - sum);
}


bool comp_coefficients(double beta, double lambda, double **&coefficients, double *&gamma_vals, 
                       double maximum=1e10){
    bool gamma_diverges = false;
    gamma_vals = new double[2];
    if (isnan(gsl_sf_gamma((1 + beta * lambda) / 2)) || isinf(gsl_sf_gamma((1 + beta * lambda) / 2)) || 
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
    
    while (!(isnan(num) || isinf(num) || isnan(num_q) || isinf(num_q) 
             || isnan(den) || isinf(den) || num > maximum || num_q > maximum
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
        if (isnan(num) || isinf(num) || isnan(num_q) || isinf(num_q) 
             || isnan(den) || isinf(den) || num > maximum || num_q > maximum
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
           q_sqr_new, chi_cav_new;

    long pos;
    for (long e = 0; e < M; e++){
        pos = sequence[e];
        for (int k = 0; k < 2; k++){
            edges[pos].fields_cav[k] = field_cav_in(pos, k, edges);
            edges[pos].var_cav[k] = var_cav_in(pos, k, edges);
            if (edges[pos].var_cav[k] > 0){
                edges[pos].var_cav_positive[k] = true;   
                Q = sqrt(edges[pos].var_cav[k]);
                h = edges[pos].fields_cav[k] * edges[pos].var_cav[k];
                h_div_Q = h / Q;
                if (h_div_Q > hmax){
                    av_new = damping * h * (1 - 1.0 / beta / h_div_Q / h_div_Q + lambda / h_div_Q / h_div_Q) + 
                             (1 - damping) * edges[pos].cond_av[k];
                    chi_cav_new = damping * edges[pos].var_cav[k] * (1 + 1.0 / beta / h_div_Q / h_div_Q - lambda / h_div_Q / h_div_Q) + 
                                  (1 - damping) * edges[pos].chi_cav[k];
                }else if (h_div_Q < hmin){
                    av_new = damping * lambda * Q / fabs(h_div_Q) + (1 - damping) * edges[pos].cond_av[k];
                    chi_cav_new = damping * edges[pos].var_cav[k] * lambda / h_div_Q / h_div_Q + 
                                  (1 - damping) * edges[pos].chi_cav[k];
                }else if(h_div_Q == 0){
                    av_new = damping * Q * sqrt(2.0 / beta) * gamma_vals[1] / gamma_vals[0] + 
                             (1 - damping) * edges[pos].cond_av[k];
                    chi_cav_new = damping * edges[pos].var_cav[k] * (beta * lambda - 
                                  2 * gamma_vals[1] / gamma_vals[0] * gamma_vals[1] / gamma_vals[0]) + 
                                  (1 - damping) * edges[pos].chi_cav[k];
                }else{
                    den = denominator(beta, lambda, h_div_Q, coefficients[0], normfactor);
                    av_new_not_damp = Q * numerator_av(beta, lambda, h_div_Q, coefficients[1]) / den;
                    av_new = damping * av_new_not_damp + (1 - damping) * edges[pos].cond_av[k];
                    q_sqr_new = edges[pos].var_cav[k] * numerator_q_sqr(beta, lambda, h_div_Q, coefficients[2]) / den;
                    chi_cav_new = damping * beta * (q_sqr_new - av_new_not_damp * av_new_not_damp) + (1 - damping) * edges[pos].chi_cav[k];
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
                delta_chi_cav = maximum;
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


double field_in(long i, Tnode *nodes, Tedge *edges){
    double sum = 0;
    long e;
    int pos;
    for (long j = 0; j < nodes[i].edges_in.size(); j++){
        e = nodes[i].edges_in[j];
        pos = nodes[i].pos_there[j];
        sum += edges[e].links[pos] * edges[e].cond_av[1 - pos];
    }
    return 1 - sum;
}


double var_in(long i, Tnode *nodes, Tedge *edges){
    double sum = 0;
    long e;
    int pos;
    for (long j = 0; j < nodes[i].edges_in.size(); j++){
        e = nodes[i].edges_in[j];
        pos = nodes[i].pos_there[j];
        sum += edges[e].links[0] * edges[e].links[1] * edges[e].chi_cav[1 - pos];
    }
    return 1.0 / (1 - sum);
}


double average(long N, Tnode *nodes, Tedge *edges, double beta, double lambda, 
               double hmin, double hmax, double **coefficients, double *gamma_vals, 
               double normfactor = 1e-14, double maximum = 1e10){
    double av = 0;
    double h, h_div_Q, Q, den, q_sqr_new;
    for (long i = 0; i < N; i++){
        nodes[i].field = field_in(i, nodes, edges);
        nodes[i].var = var_in(i, nodes, edges);
        if (nodes[i].var > 0){
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
            nodes[i].av = maximum;
            nodes[i].chi = maximum;
        }
        
        av += nodes[i].av;
        
    }
    return av / N;
}

double average_sqr(long N, Tnode *nodes){
    double av_sqr = 0;
    for (long i = 0; i < N; i++){
        av_sqr += nodes[i].av * nodes[i].av;
    }
    return av_sqr / N;
}


double average_chi(long N, Tnode *nodes){
    double av = 0;
    for (long i = 0; i < N; i++){
        av += nodes[i].chi;
    }
    return av / N;
}

double average_chi_sqr(long N, Tnode *nodes){
    double av_sqr = 0;
    for (long i = 0; i < N; i++){
        av_sqr += nodes[i].chi * nodes[i].chi;
    }
    return av_sqr / N;
}


double average_cav(long M, Tedge *edges){
    double av = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            av += edges[e].cond_av[k];
        }
    }
    return av / (2 * M);
}


double average_cav_sqr(long M, Tedge *edges){
    double av_sqr = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            av_sqr += edges[e].cond_av[k] * edges[e].cond_av[k];
        }
    }
    return av_sqr / (2 * M);
}


double average_chi_cav(long M, Tedge *edges){
    double av = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            av += edges[e].chi_cav[k];
        }
    }
    return av / (2 * M);
}


double average_chi_cav_sqr(long M, Tedge *edges){
    double av_sqr = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            av_sqr += edges[e].chi_cav[k] * edges[e].chi_cav[k];
        }
    }
    return av_sqr / (2 * M);
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


void print_results(double av, int iter, Tnode *nodes, Tedge *edges, long N, long M, long seed, 
                   int max_iter, bool divergence, double beta, double lambda, 
                   double hmax, double **coefficients, bool same_fixed_point, double normfactor = 1e-14){
    long counter_diverged = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            if (!edges[e].converged[k]){
                counter_diverged++;
            }
        }
    }

    long counter_varneg = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            if (!edges[e].var_cav_positive[k]){
                counter_varneg++;
            }
        }
    }

    long counter_chi_cav_diverged = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            if (!edges[e].chi_cav_converged[k]){
                counter_chi_cav_diverged++;
            }
        }
    }

    long count_dead = 0;
    for (long i = 0; i < N; i++){
        if (nodes[i].av <= 0){
            count_dead++;
        }
    }

    if (iter >= max_iter || divergence){
        same_fixed_point = false;
    }

    double av_sqr = average_sqr(N, nodes);
    double av_chi = average_chi(N, nodes);
    double av_chi_sqr = average_chi_sqr(N, nodes);

    double av_cav = average_cav(M, edges);
    double av_cav_sqr = average_cav_sqr(M, edges);
    double av_chi_cav = average_chi_cav(M, edges);
    double av_chi_cav_sqr = average_chi_cav_sqr(M, edges);
    if (divergence){
        cout << iter << "\t" << "diverges" << "\t" << 
                av_cav << "\t" << sqrt(fabs(av_cav_sqr - av_cav * av_cav) / 2 / M) << "\t" << 
                av_chi_cav << "\t" << sqrt(fabs(av_chi_cav_sqr - av_chi_cav * av_chi_cav) / 2 / M) << "\t" << 
                av << "\t" << sqrt(fabs(av_sqr - av * av) / N) << "\t" << 
                av_chi << "\t" << sqrt(fabs(av_chi_sqr - av_chi * av_chi) / N) << "\t" << 
                counter_diverged << "\t" << counter_varneg << "\t" << counter_chi_cav_diverged << "\t" << 
                count_dead << "\t" << seed << "\t" << same_fixed_point << endl;
    }else{
        bool conv = iter < max_iter;
        cout << iter << "\t" << conv << "\t" << 
                av_cav << "\t" << sqrt(fabs(av_cav_sqr - av_cav * av_cav) / 2 / M) << "\t" << 
                av_chi_cav << "\t" << sqrt(fabs(av_chi_cav_sqr - av_chi_cav * av_chi_cav) / 2 / M) << "\t" << 
                av << "\t" << sqrt(fabs(av_sqr - av * av) / N) << "\t" << 
                av_chi << "\t" << sqrt(fabs(av_chi_sqr - av_chi * av_chi) / N) << "\t" << 
                counter_diverged << "\t" << counter_varneg << "\t" << counter_chi_cav_diverged << "\t" << 
                count_dead << "\t" << seed << "\t" << same_fixed_point << endl;
    }

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


void set_av_prev(Tnode *nodes, long N) {
    for (long i = 0; i < N; i++) {
        nodes[i].av_prev_fixed_point = nodes[i].av;
        nodes[i].chi_prev_fixed_point = nodes[i].chi;
    }
}


bool compare_fixed_points(Tnode *nodes, long N, double tol_fixed_point){
    double max_diff = 0.0;
    double diff;
    for (long i = 0; i < N; i++) {
        diff = fabs(nodes[i].av - nodes[i].av_prev_fixed_point);
        if (diff > max_diff) {
            max_diff = diff;
        }
        diff = fabs(nodes[i].chi - nodes[i].chi_prev_fixed_point);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    cerr << "Max difference in fixed points: " << max_diff << endl;
    return max_diff < tol_fixed_point;
}


int main(int argc, char *argv[]) {
    unsigned long seed = atoi(argv[1]);
    double avn_0 = atof(argv[2]);
    double T = atof(argv[3]);
    double lambda = atof(argv[4]);
    double tol = atof(argv[5]);
    int max_iter = atoi(argv[6]);
    double eps = atof(argv[7]);
    double mu = atof(argv[8]);
    double sigma = atof(argv[9]);
    unsigned long seed_seq_init = atoi(argv[10]);
    unsigned long num_seq = atoi(argv[11]);
    double tol_fixed_point = atof(argv[12]);
    double damping = atof(argv[13]);
    bool gr_inside = atoi(argv[14]);

    gsl_set_error_handler_off();

    Tnode *nodes;
    Tedge *edges;
    double beta = 1.0 / T;
    long N, M;

    if (gr_inside){
        N = atol(argv[15]);
        int c = atoi(argv[16]);
        gsl_rng * r;

        init_ran(r, seed);

        M = init_graph_inside_RRG(nodes, edges, N, c, eps, mu, sigma, r);
    }else{
        init_graph_from_input(nodes, edges, N, M);
    }

    
    double hmax = find_divergence_max(beta, 1 + beta * lambda / 2);
    double **coefficients, *gamma_vals;
    comp_coefficients(beta, lambda, coefficients, gamma_vals);
    double hmin = find_divergence_min(beta, lambda, coefficients);

    bool divergence;
    long sequence[M];

    divergence = false;
    produce_random_seq(seed_seq_init, M, sequence);
    init_avgs(M, edges, avn_0);
    int iter = convergence(M, beta, lambda, edges, tol, max_iter, divergence, 
                           hmin, hmax, coefficients, gamma_vals, sequence, damping);
    double av = average(N, nodes, edges, beta, lambda, hmin, hmax, coefficients, 
                        gamma_vals);


    bool same_fixed_point = true;
    if (!divergence && iter < max_iter){
        set_av_prev(nodes, N);
        unsigned long seed_seq = seed_seq_init + 1;
        while (seed_seq < seed_seq_init + num_seq && !divergence && iter < max_iter && same_fixed_point) {
            produce_random_seq(seed_seq, M, sequence);
            init_avgs(M, edges, avn_0);
            iter = convergence(M, beta, lambda, edges, tol, max_iter, divergence, 
                               hmin, hmax, coefficients, gamma_vals, sequence, 
                               damping);
            av = average(N, nodes, edges, beta, lambda, hmin, hmax, 
                         coefficients, gamma_vals);
            same_fixed_point = compare_fixed_points(nodes, N, tol_fixed_point);
            seed_seq++;
        }
    }
    

    print_results(av, iter, nodes, edges, N, M, seed, max_iter, divergence,
                  beta, lambda, hmax, coefficients, same_fixed_point);
    
    return 0;
}