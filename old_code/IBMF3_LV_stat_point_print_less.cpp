#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_integration.h>
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
    vector <long> neighs;
    vector <double> links_in;
    double field; // local field in that node
    double var; // variance of the perturbed gaussian that depends on the neighbors
    double W; // third moment of the perturbed gaussian that depends on the neighbors
    double av; // average value of n in that node
    double q_sqr; // average value of n^2 in that node
    double w3; // average value of n^3 in that node
}Tnode;


void init_graph_from_input(Tnode *&nodes, long &N){
    long M;
    scanf("%ld %ld", &N, &M);
    nodes = new Tnode[N];
    long i, j;
    double aij, aji;
    for (long e = 0; e < M; e++){
        scanf("%ld %ld %lf %lf", &i, &j, &aij, &aji);
        nodes[i].neighs.push_back(j);
        nodes[j].neighs.push_back(i);
        nodes[i].links_in.push_back(aji);
        nodes[j].links_in.push_back(aij);
    }
}


void init_nodes(Tnode *nodes, long N){
    for (long i = 0; i < N; i++){
        nodes[i].links_in = vector <double> ();
        nodes[i].neighs = vector <long> ();
    }
}

void init_graph_inside_RRG(Tnode *&nodes, long N, int c, double eps,
                           double mu, double sigma, gsl_rng * r){
    // eps is the degree of symmetry of the graph
    if (N * c % 2 != 0){
        cout << "N*c must be even to create a random regular graph" << endl;
        exit(1);
    }else{
        bool success = false;
        long M = N * c / 2;
        long pos_i, pos_j, i, j;
        double aij, aji;
        nodes = new Tnode[N];

        while (!success)
        {
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
                nodes[i].neighs.push_back(j);
                nodes[j].neighs.push_back(i);
                aij = mu + gsl_ran_gaussian(r, sigma);
                if (gsl_rng_uniform_pos(r) < eps){
                    aji = aij;
                }else{
                    aji = mu + gsl_ran_gaussian(r, sigma);
                }
                nodes[i].links_in.push_back(aji);
                nodes[j].links_in.push_back(aij);
            }

            pos_i = 0;
            pos_j = 1;
            i = copies[pos_i];
            j = copies[pos_j];
            if (i != j){
                success = true;
                nodes[i].neighs.push_back(j);
                nodes[j].neighs.push_back(i);
                aij = mu + gsl_ran_gaussian(r, sigma);
                if (gsl_rng_uniform_pos(r) < eps){
                    aji = aij;
                }else{
                    aji = mu + gsl_ran_gaussian(r, sigma);
                }
                nodes[i].links_in.push_back(aji);
                nodes[j].links_in.push_back(aij);
            }
        }
    }    
}


void init_avgs(long N, Tnode *nodes, double avn_0){
    for (long i = 0; i < N; i++){
        nodes[i].av = avn_0;
        nodes[i].q_sqr = avn_0 * avn_0;
        nodes[i].w3 = avn_0 * avn_0 * avn_0;
    }
}


double field_in(long i, Tnode *nodes, vector <long> neighs, vector <double> links_in){
    double sum = 0;
    for (long j = 0; j < neighs.size(); j++){
        sum += links_in[j] * nodes[neighs[j]].av;
    }
    return 1 - sum;
}


double var_in(long i, Tnode *nodes, vector <long> neighs, vector <double> links_in, double beta){
    double sum = 0;
    for (long j = 0; j < neighs.size(); j++){
        sum += links_in[j] * links_in[j] * (nodes[neighs[j]].q_sqr - nodes[neighs[j]].av * nodes[neighs[j]].av);
    }
    return 1.0 / (1 - beta * sum);
}


double W_in(long i, Tnode *nodes, vector <long> neighs, vector <double> links_in, double beta){
    double sum = 0;
    for (long j = 0; j < neighs.size(); j++){
        sum += links_in[j] * links_in[j] * links_in[j] * 
               (nodes[neighs[j]].w3 - 3 * nodes[neighs[j]].av * nodes[neighs[j]].q_sqr + 2 * nodes[neighs[j]].av * nodes[neighs[j]].av * nodes[neighs[j]].av);
    }
    return beta * beta * sum;
}


double numerator_av_IBMF(double beta, double lambda, double hi_div_s, double hi2_div_s){
    return 2 * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi2_div_s / 2) + 
           sqrt(2 * beta) * hi_div_s * beta * lambda * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi2_div_s / 2);
}


double numerator_q_sqr_IBMF2(double beta, double lambda, double hi_div_s, double hi2_div_s){
    return 4 * hi_div_s * gsl_sf_gamma((3 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 1.5, -beta * hi2_div_s / 2) + 
           sqrt(2 * beta) * lambda * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1(-(1 + beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2);
}


double denominator_IBMF(double beta, double lambda, double hi_div_s, double hi2_div_s){
    return sqrt(2 * beta) * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2) + 
           2 * beta * hi_div_s * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi2_div_s / 2);
}


double integrand_av(double ni, void *params){
    double *params_d = (double *) params;
    double beta = params_d[0];
    double lambda = params_d[1];
    double W = params_d[2];
    double S2 = params_d[3];
    double M = params_d[4];
    double m = params_d[5];
    double integrand_av = pow(ni, beta * lambda) * 
                          exp(-beta * (W * ni * ni * ni / 6 + ni * ni / 2 * (1.0 / S2 - m * W) - 
                                       ni * (M + m * (1 - S2) - W * m * m / 2)));
    return integrand_av;
}

double integrand_q2(double ni, void *params){
    return integrand_av(ni, params) * ni;
}

double integrand_w3(double ni, void *params){
    return integrand_q2(ni, params) * ni;
}


double numerator_IBMF3(double beta, double lambda, double W, double S2, double M, double m, 
                       gsl_function function, double epsabs, double epsrel, long limit, 
                       gsl_integration_workspace *workspace, double *params){
    double result, error;
    params[0] = beta;
    params[1] = lambda;
    params[2] = W;
    params[3] = S2;
    params[4] = M;
    params[5] = m;
    gsl_integration_qagiu(&function, 0, epsabs, epsrel, limit, workspace, &result, &error);
    return result;
}



void comp_fields(long N, Tnode *nodes){
    for (long i = 0; i < N; i++){
        nodes[i].field = field_in(i, nodes, nodes[i].neighs, nodes[i].links_in);
    }
}

void comp_vars(long N, Tnode *nodes, double beta){
    for (long i = 0; i < N; i++){
        nodes[i].var = var_in(i, nodes, nodes[i].neighs, nodes[i].links_in, beta);
    }
}


void comp_W(long N, Tnode *nodes, double beta){
    for (long i = 0; i < N; i++){
        nodes[i].W = W_in(i, nodes, nodes[i].neighs, nodes[i].links_in, beta);
    }
}


double new_averages(long N, double beta, double lambda, Tnode *nodes, double tol_asymp, 
                    gsl_function integrand_av, gsl_function integrand_q_sqr, gsl_function integrand_w3, 
                    double epsabs, double epsrel, long limit, gsl_integration_workspace *workspace, double *params){
    double delta = 0, delta_av, delta_q_sqr, delta_w3, s, hi_div_s, hi2_div_s, den, av_new, q_sqr_new, w3_new;
    
    for (long i = 0; i < N; i++){
        if (nodes[i].W > 0){
            av_new = numerator_IBMF3(beta, lambda, nodes[i].W, nodes[i].var, nodes[i].field, nodes[i].av, integrand_av, 
                                     epsabs, epsrel, limit, workspace, params);
            q_sqr_new = numerator_IBMF3(beta, lambda, nodes[i].W, nodes[i].var, nodes[i].field, nodes[i].av, integrand_q_sqr, 
                                        epsabs, epsrel, limit, workspace, params);
            w3_new = numerator_IBMF3(beta, lambda, nodes[i].W, nodes[i].var, nodes[i].field, nodes[i].av, integrand_w3, 
                                     epsabs, epsrel, limit, workspace, params);
            den = (nodes[i].W * w3_new / 2 + 
                   q_sqr_new * (1.0 / nodes[i].var - nodes[i].av * nodes[i].W) - 
                   av_new * (nodes[i].field + nodes[i].av * (1 - nodes[i].var) - nodes[i].W * nodes[i].av * nodes[i].av / 2)) / lambda;
            av_new /= den;
            q_sqr_new /= den;
            w3_new /= den;
        }else{
            if (nodes[i].var > 0){
                s = sqrt(nodes[i].var);
                hi_div_s = nodes[i].av * (1.0 / s - s) + s * nodes[i].field;
                hi2_div_s = hi_div_s * hi_div_s;
                if (exp(-beta * hi2_div_s / 2) / pow(hi2_div_s, beta * lambda / 2) < tol_asymp){
                    hi_div_s = nodes[i].field;
                    hi2_div_s = hi_div_s * hi_div_s;
                    den = denominator_IBMF(beta, lambda, hi_div_s, hi2_div_s);
                    av_new = numerator_av_IBMF(beta, lambda, hi_div_s, hi2_div_s) / den;
                    q_sqr_new = av_new * av_new;
                    w3_new = av_new * av_new * av_new;
                }else{
                    den = denominator_IBMF(beta, lambda, hi_div_s, hi2_div_s);
                    av_new = s * numerator_av_IBMF(beta, lambda, hi_div_s, hi2_div_s) / den; 
                    q_sqr_new = nodes[i].var * numerator_q_sqr_IBMF2(beta, lambda, hi_div_s, hi2_div_s) / den;
                    w3_new = av_new * av_new * av_new;
                }
            }else{
                hi_div_s = nodes[i].field;
                hi2_div_s = hi_div_s * hi_div_s;
                den = denominator_IBMF(beta, lambda, hi_div_s, hi2_div_s);
                av_new = numerator_av_IBMF(beta, lambda, hi_div_s, hi2_div_s) / den;
                q_sqr_new = av_new * av_new;
                w3_new = av_new * av_new * av_new;
            }
        }
        delta_av = fabs(av_new - nodes[i].av);
        delta_q_sqr = fabs(q_sqr_new - nodes[i].q_sqr);
        delta_w3 = fabs(w3_new - nodes[i].w3);
        if (delta_av > delta){
            delta = delta_av;
        }
        if (delta_q_sqr > delta){
            delta = delta_q_sqr;
        }
        if (delta_w3 > delta){
            delta = delta_w3;
        }
        nodes[i].av = av_new;
        nodes[i].q_sqr = q_sqr_new;
        nodes[i].w3 = w3_new;
    }
    return delta;
}




double average_field(long N, Tnode *nodes){
    double av = 0;
    for (long i = 0; i < N; i++){
        av += nodes[i].field;
    }
    return av / N;
}

double average_field_sqr(long N, Tnode *nodes){
    double av_sqr = 0;
    for (long i = 0; i < N; i++){
        av_sqr += nodes[i].field * nodes[i].field;
    }
    return av_sqr / N;
}


double average_var(long N, Tnode *nodes){
    double av = 0;
    for (long i = 0; i < N; i++){
        av += nodes[i].var;
    }
    return av / N;
}

double average_var_sqr(long N, Tnode *nodes){
    double av_sqr = 0;
    for (long i = 0; i < N; i++){
        av_sqr += nodes[i].var * nodes[i].var;
    }
    return av_sqr / N;
}

double average_W(long N, Tnode *nodes){
    double av = 0;
    for (long i = 0; i < N; i++){
        av += nodes[i].W;
    }
    return av / N;
}

double average_W_sqr(long N, Tnode *nodes){
    double av_sqr = 0;
    for (long i = 0; i < N; i++){
        av_sqr += nodes[i].W * nodes[i].W;
    }
    return av_sqr / N;
}

int convergence(long N, double beta, double lambda, Tnode *nodes, double tol, double tol_asymp, 
                int max_iter, bool &divergence, gsl_function integrand_av, gsl_function integrand_q_sqr, 
                gsl_function integrand_w3, double epsabs, double epsrel, long limit, 
                gsl_integration_workspace *workspace, double *params){
    double delta = tol + 1;
    int iter = 0;

    comp_fields(N, nodes);
    comp_vars(N, nodes, beta);
    comp_W(N, nodes, beta);

    while (delta > tol && iter < max_iter){
        delta = new_averages(N, beta, lambda, nodes, tol_asymp,
                             integrand_av, integrand_q_sqr, integrand_w3, 
                             epsabs, epsrel, limit, workspace, params);
        iter++;
        comp_fields(N, nodes);
        comp_vars(N, nodes, beta);
        comp_W(N, nodes, beta);
        if (isinf(delta)){
            divergence = true;
            return iter;
        }
    }
    divergence = false;
    return iter;
}


void print_results(int iter, Tnode *nodes, long N, long seed, int max_iter, bool divergence){
    double av_field = average_field(N, nodes);
    double av_field_sqr = average_field_sqr(N, nodes);
    double av_var = average_var(N, nodes);
    double av_var_sqr = average_var_sqr(N, nodes);
    double av_W = average_W(N, nodes);
    double av_W_sqr = average_W_sqr(N, nodes);
    if (divergence){
        cout << iter << "\t" << "diverges" << "\t" << 
                av_field << "\t" << sqrt((av_field_sqr - av_field * av_field) / N) << "\t" << 
                av_var << "\t" << sqrt((av_var_sqr - av_var * av_var) / N) << "\t" << 
                av_W << "\t" << sqrt((av_W_sqr - av_W * av_W) / N) << "\t" << 
                seed << endl;
    }else{
        bool conv = iter < max_iter;
        cout << iter << "\t" << conv << "\t" << 
                av_field << "\t" << sqrt((av_field_sqr - av_field * av_field) / N) << "\t" << 
                av_var << "\t" << sqrt((av_var_sqr - av_var * av_var) / N) << "\t" << 
                av_W << "\t" << sqrt((av_W_sqr - av_W * av_W) / N) << "\t" << 
                seed << endl;
    }

}



int main(int argc, char *argv[]) {
    unsigned long seed = atoi(argv[1]);
    double avn_0 = atof(argv[2]);
    double T = atof(argv[3]);
    double lambda = atof(argv[4]);
    double tol = atof(argv[5]);
    double tol_asymp = atof(argv[6]);
    int max_iter = atoi(argv[7]);
    double eps = atof(argv[8]);
    double mu = atof(argv[9]);
    double sigma = atof(argv[10]);
    double epsabs = atof(argv[11]);
    double epsrel = atof(argv[12]);
    long limit = atol(argv[13]);
    bool gr_inside = atoi(argv[14]);


    Tnode *nodes;
    double beta = 1.0 / T;
    long N;
    char gr_str[20];

    if (gr_inside){
        sprintf(gr_str, "gr_inside_RRG");
        N = atol(argv[15]);
        int c = atoi(argv[16]);
        gsl_rng * r;

        init_ran(r, seed);

        init_graph_inside_RRG(nodes, N, c, eps, mu, sigma, r);
    }else{
        sprintf(gr_str, "gr_from_input");
        init_graph_from_input(nodes, N);
    }

    init_avgs(N, nodes, avn_0);

    bool divergence;

    double params[5]; // beta, lambda, W, S2, M
    gsl_function integrand_av_gsl, integrand_q_sqr_gsl, integrand_w3_gsl;
    integrand_av_gsl.function = &integrand_av;
    integrand_q_sqr_gsl.function = &integrand_q2;
    integrand_w3_gsl.function = &integrand_w3;
    integrand_av_gsl.params = params;
    integrand_q_sqr_gsl.params = params;
    integrand_w3_gsl.params = params;
    gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(limit);

    int iter = convergence(N, beta, lambda, nodes, tol, tol_asymp, max_iter, divergence,
                           integrand_av_gsl, integrand_q_sqr_gsl, integrand_w3_gsl, 
                           epsabs, epsrel, limit, workspace, params);

    print_results(iter, nodes, N, seed, max_iter, divergence);

    gsl_integration_workspace_free(workspace);
    
    return 0;
}