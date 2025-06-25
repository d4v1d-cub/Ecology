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
    vector <long> neighs;
    vector <double> links_in;
    double field; // local field in that node
    double var; // variance of the perturbed gaussian that depends on the neighbors
    double av; // average value of n in that node
    double q_sqr; // average value of n^2 in that node
    bool converged; // whether the node converged or not
    bool var_positive; // whether the variance is positive or not
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


void init_graph_inside_RGER_full_asym(Tnode *&nodes, long N, double c,
                                      double mu, double sigma, gsl_rng * r){
    // eps is the degree of symmetry of the graph
    double aji;
    nodes = new Tnode[N];

    init_nodes(nodes, N);

    for (long i = 0; i < N; i++){
        for (long j = 0; j < i; j++){
            if (gsl_rng_uniform(r) < c / N){
                nodes[i].neighs.push_back(j);
                aji = mu + gsl_ran_gaussian(r, sigma);
                nodes[i].links_in.push_back(aji);
            }
        }
        for (long j = i + 1; j < N; j++){
            if (gsl_rng_uniform(r) < c / N){
                nodes[i].neighs.push_back(j);
                aji = mu + gsl_ran_gaussian(r, sigma);
                nodes[i].links_in.push_back(aji);
            }
        }
    }
}


void init_avgs(long N, Tnode *nodes, double avn_0){
    for (long i = 0; i < N; i++){
        nodes[i].av = avn_0;
        nodes[i].q_sqr = avn_0 * avn_0;
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

double numerator_av(double beta, double lambda, double hi_div_s, double hi2_div_s){
    return gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi2_div_s / 2) + 
           sqrt(beta / 2) * hi_div_s * beta * lambda * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi2_div_s / 2);
}


double numerator_q_sqr(double beta, double lambda, double hi_div_s, double hi2_div_s){
    return 2 * hi_div_s * gsl_sf_gamma((3 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 1.5, -beta * hi2_div_s / 2) + 
           sqrt(beta / 2) * lambda * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1(-(1 + beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2);
}


double denominator(double beta, double lambda, double hi_div_s, double hi2_div_s, double normfactor = 1e-14){
    return sqrt(beta / 2) * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2) + 
           beta * hi_div_s * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi2_div_s / 2) + 
           normfactor;
}


double numerator_av_asymp(double beta, double lambda, double hi_div_s, double hi2_div_s){
    return sqrt(lambda) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi2_div_s / 2) +
           beta * lambda * hi_div_s * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi2_div_s / 2);
}


double numerator_q_sqr_asymp(double beta, double lambda, double hi_div_s, double hi2_div_s){
    return lambda * gsl_sf_hyperg_1F1(-(1 + beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2) +
           beta * lambda * sqrt(lambda) * hi_div_s * gsl_sf_hyperg_1F1(-beta * lambda / 2, 1.5, -beta * hi2_div_s / 2);
}


double denominator_asymp(double beta, double lambda, double hi_div_s, double hi2_div_s, double normfactor = 1e-14){
    return gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2) + 
           beta * sqrt(lambda) * hi_div_s * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi2_div_s / 2) + 
           normfactor;
}


int check_wich_diverges(double beta, double lambda, double hi2_div_s, double limit = 1e+10){
    if (isnan(gsl_sf_gamma((3 + beta * lambda) / 2)) || isinf(gsl_sf_gamma((3 + beta * lambda) / 2)) || gsl_sf_gamma((3 + beta * lambda) / 2) > limit){
        if (isnan(gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi2_div_s / 2)) || isinf(gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi2_div_s / 2)) 
            || gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi2_div_s / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        }else if(isnan(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi2_div_s / 2)) || isinf(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi2_div_s / 2))
                 || gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi2_div_s / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        } else if(isnan(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2)) || isinf(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2))
                 || gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        }else if(isnan(gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi2_div_s / 2)) || isinf(gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi2_div_s / 2)) || 
                 gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi2_div_s / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        }else if(isnan(gsl_sf_hyperg_1F1(-(1 + beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2)) || isinf(gsl_sf_hyperg_1F1(-(1 + beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2))
                 || gsl_sf_hyperg_1F1(-(1 + beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        }else if(isnan(gsl_sf_hyperg_1F1(-beta * lambda / 2, 1.5, -beta * hi2_div_s / 2)) || isinf(gsl_sf_hyperg_1F1(-beta * lambda / 2, 1.5, -beta * hi2_div_s / 2)) || 
                 gsl_sf_hyperg_1F1(-beta * lambda / 2, 1.5, -beta * hi2_div_s / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        }else {
            return 2; // only gamma diverges
        }
    }else if (isnan(gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi2_div_s / 2)) || isinf(gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi2_div_s / 2)) 
            || gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi2_div_s / 2) > limit){
        return 1; // gamma and hypergeometric diverge
    }else if(isnan(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi2_div_s / 2)) || isinf(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi2_div_s / 2))
                 || gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi2_div_s / 2) > limit){
        return 1; // gamma and hypergeometric diverge
    }else if(isnan(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2)) || isinf(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2))
                 || gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2) > limit){
        return 1; // gamma and hypergeometric diverge
    }else if(isnan(gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi2_div_s / 2)) || isinf(gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi2_div_s / 2)) || 
                 gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi2_div_s / 2) > limit){
        return 1; // gamma and hypergeometric diverge
    }else if(isnan(gsl_sf_hyperg_1F1(-(1 + beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2)) || isinf(gsl_sf_hyperg_1F1(-(1 + beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2))
                 || gsl_sf_hyperg_1F1(-(1 + beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2) > limit){
        return 1; // gamma and hypergeometric diverge
    }else if(isnan(gsl_sf_hyperg_1F1(-beta * lambda / 2, 1.5, -beta * hi2_div_s / 2)) || isinf(gsl_sf_hyperg_1F1(-beta * lambda / 2, 1.5, -beta * hi2_div_s / 2)) || 
                 gsl_sf_hyperg_1F1(-beta * lambda / 2, 1.5, -beta * hi2_div_s / 2) > limit){
        return 1; // gamma and hypergeometric diverge
    }else{
        return 0; // cannot identify divergence
    }
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


double new_averages(long N, double beta, double lambda, Tnode *nodes, double tol, double normfactor = 1e-14){
    double delta = 0, delta_av, delta_q_sqr, s, hi_div_s, hi2_div_s, den, av_new, q_sqr_new;
    int identify_divergence = 0;

    for (long i = 0; i < N; i++){
        if (nodes[i].var > 0){
            nodes[i].var_positive = true;   
            s = sqrt(nodes[i].var);
            hi_div_s = nodes[i].av * (1.0 / s - s) + s * nodes[i].field;
            hi2_div_s = hi_div_s * hi_div_s;
            den = denominator(beta, lambda, hi_div_s, hi2_div_s, normfactor);
            av_new = s * numerator_av(beta, lambda, hi_div_s, hi2_div_s) / den; 
            q_sqr_new = nodes[i].var * numerator_q_sqr(beta, lambda, hi_div_s, hi2_div_s) / den;
            if (isnan(q_sqr_new) || isinf(q_sqr_new) || isnan(av_new) || isinf(av_new)){
                identify_divergence = check_wich_diverges(beta, lambda, hi2_div_s);
                if (identify_divergence == 1){
                    if (hi_div_s < 0){
                        av_new = 0;
                        q_sqr_new = 0;
                    }else{
                        av_new = s * hi_div_s;
                        q_sqr_new = nodes[i].var * hi2_div_s;
                    }
                }else if (identify_divergence == 2){
                    den = denominator_asymp(beta, lambda, hi_div_s, hi2_div_s, normfactor);
                    av_new = s * numerator_av_asymp(beta, lambda, hi_div_s, hi2_div_s) / den; 
                    q_sqr_new = nodes[i].var * numerator_q_sqr_asymp(beta, lambda, hi_div_s, hi2_div_s) / den;
                }else{
                    cout << "Cannot identify divergence for node " << i << endl;
                    exit(1);
                }
            }
        }else{
            nodes[i].var_positive = false;
            hi_div_s = nodes[i].field;
            hi2_div_s = hi_div_s * hi_div_s;
            den = denominator(beta, lambda, hi_div_s, hi2_div_s, normfactor);
            av_new = numerator_av(beta, lambda, hi_div_s, hi2_div_s) / den;
            if (isnan(av_new) || isinf(av_new)){
                identify_divergence = check_wich_diverges(beta, lambda, hi2_div_s);
                if (identify_divergence == 1){
                    if (hi_div_s < 0){
                        av_new = 0;
                    }else{
                        av_new = hi_div_s;
                    }
                }else if (identify_divergence == 2){
                    den = denominator_asymp(beta, lambda, hi_div_s, hi2_div_s, normfactor);
                    av_new = numerator_av_asymp(beta, lambda, hi_div_s, hi2_div_s) / den;
                }else{
                    cout << "Cannot identify divergence for node " << i << endl;
                    exit(1);
                }
            }
            q_sqr_new = av_new * av_new;
        }
        
        delta_av = fabs(av_new - nodes[i].av);
        delta_q_sqr = fabs(q_sqr_new - nodes[i].q_sqr);
        if (delta_av > delta){
            delta = delta_av;
        }
        if (delta_q_sqr > delta){
            delta = delta_q_sqr;
        }
        nodes[i].av = av_new;
        nodes[i].q_sqr = q_sqr_new;

        if (delta_av < tol && delta_q_sqr < tol){
            nodes[i].converged = true;
        }else{
            nodes[i].converged = false;
        }

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

int convergence(long N, double beta, double lambda, Tnode *nodes, double tol, int max_iter, bool &divergence){
    double delta = tol + 1;
    int iter = 0;

    comp_fields(N, nodes);
    comp_vars(N, nodes, beta);

    while (delta > tol && iter < max_iter){
        delta = new_averages(N, beta, lambda, nodes, tol);
        iter++;
        comp_fields(N, nodes);
        comp_vars(N, nodes, beta);
        if (isinf(delta)){
            divergence = true;
            return iter;
        }
        cout << iter << "\t" << delta << endl;
    }
    divergence = false;
    return iter;
}


void print_results(int iter, Tnode *nodes, long N, long seed, int max_iter, bool divergence){
    long counter_diverged = 0;
    for (long i = 0; i < N; i++){
        if (!nodes[i].converged){
            counter_diverged++;
        }
    }

    long counter_varneg = 0;
    for (long i = 0; i < N; i++){
        if (!nodes[i].var_positive){
            counter_varneg++;
        }
    }

    double av_field = average_field(N, nodes);
    double av_field_sqr = average_field_sqr(N, nodes);
    double av_var = average_var(N, nodes);
    double av_var_sqr = average_var_sqr(N, nodes);
    if (divergence){
        cout << iter << "\t" << "diverges" << "\t" << 
                av_field << "\t" << sqrt((av_field_sqr - av_field * av_field) / N) << "\t" << 
                av_var << "\t" << sqrt((av_var_sqr - av_var * av_var) / N) << "\t" << 
                counter_diverged << "\t" << counter_varneg << "\t" <<
                seed << endl;
    }else{
        bool conv = iter < max_iter;
        cout << iter << "\t" << conv << "\t" << 
                av_field << "\t" << sqrt((av_field_sqr - av_field * av_field) / N) << "\t" << 
                av_var << "\t" << sqrt((av_var_sqr - av_var * av_var) / N) << "\t" << 
                counter_diverged << "\t" << counter_varneg << "\t" <<
                seed << endl;
    }

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
    bool gr_inside = atoi(argv[10]);

    gsl_set_error_handler_off();

    Tnode *nodes;
    double beta = 1.0 / T;
    long N;

    if (gr_inside){
        if (argc > 13){
            if (atoi(argv[13]) == 1){
                N = atol(argv[11]);
                int c = atoi(argv[12]);
                gsl_rng * r;

                init_ran(r, seed);

                init_graph_inside_RRG(nodes, N, c, eps, mu, sigma, r);
            }else if (atoi(argv[13]) == 2)
            {
                N = atol(argv[11]);
                double c = atof(argv[12]);
                gsl_rng * r;
                init_ran(r, seed);
                init_graph_inside_RGER_full_asym(nodes, N, c, mu, sigma, r);
            }else{
                cout << "Wrong value for the 14th argument. It must be 1 or 2." << endl;
                exit(1);
            }
            
        }else{
            N = atol(argv[11]);
            int c = atoi(argv[12]);
            gsl_rng * r;

            init_ran(r, seed);

            init_graph_inside_RRG(nodes, N, c, eps, mu, sigma, r);
        }
    }else{
        init_graph_from_input(nodes, N);
    }

    init_avgs(N, nodes, avn_0);

    bool divergence;

    int iter = convergence(N, beta, lambda, nodes, tol, max_iter, divergence);

    print_results(iter, nodes, N, seed, max_iter, divergence);
    
    return 0;
}