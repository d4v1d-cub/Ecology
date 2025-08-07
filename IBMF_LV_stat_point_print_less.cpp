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
    double field; // average value of n in that node
    bool converged; // whether the node converged or not
    double av;
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
    }
}


double field_in(long i, Tnode *nodes){
    double field = 0;
    for (long j = 0; j < nodes[i].neighs.size(); j++){
        field += nodes[i].links_in[j] * nodes[nodes[i].neighs[j]].av;
    }
    return 1 - field;
}


bool comp_coefficients(double beta, double lambda, double **&coefficients, double maximum=1e10){
    bool gamma_diverges = false;
    if (isnan(gsl_sf_gamma((1 + beta * lambda) / 2)) || isinf(gsl_sf_gamma((1 + beta * lambda) / 2)) || 
        gsl_sf_gamma((1 + beta * lambda) / 2) > maximum){
        gamma_diverges = true;
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


double find_divergence(double beta, double alpha, double hmax=100, double precision=1e-4, double maximum=1e10){
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


double new_averages(long N, double beta, double lambda, Tnode *nodes, double tol, 
                    double hmax, double **coefficients, int iter, double normfactor = 1e-14){
    double var = 0, var_i;
    double av_new;
    for (long i = 0; i < N; i++){
        if (nodes[i].field > hmax){
            av_new = nodes[i].field * (1 - 1.0 / beta / nodes[i].field / nodes[i].field + 
                                       lambda / nodes[i].field / nodes[i].field);                      
        }else if (nodes[i].field < 0)
        {
            av_new = 0;
        }
        else {
            av_new = numerator_av(beta, lambda, nodes[i].field, coefficients[1]) /
                     denominator(beta, lambda, nodes[i].field, coefficients[0], normfactor);
        }

        if (isnan(av_new) || isinf(av_new)){
            cerr << "Error: av_new is nan or inf at site i=" << i << "   iter=" << iter << endl;
            return sqrt(-1);
        }
        
        var_i = fabs(av_new - nodes[i].av);
        if (var_i > var){
            var = var_i;
        }
        if (var_i < tol){
            nodes[i].converged = true;
        }
        else{
            nodes[i].converged = false;
        }

        nodes[i].av = av_new;
    }
    return var;
}

void comp_fields(long N, Tnode *nodes){
    for (long i = 0; i < N; i++){
        nodes[i].field = field_in(i, nodes);
    }
}

double average(long N, Tnode *nodes){
    double av = 0;
    for (long i = 0; i < N; i++){
        av += nodes[i].field;
    }
    return av / N;
}

double average_sqr(long N, Tnode *nodes){
    double av_sqr = 0;
    for (long i = 0; i < N; i++){
        av_sqr += nodes[i].field * nodes[i].field;
    }
    return av_sqr / N;
}


int convergence(long N, double beta, double lambda, Tnode *nodes, double tol, 
                 int max_iter, bool &divergence, double hmax, double **coefficients, 
                 double maximum=1e10){
    double var = tol + 1;
    int iter = 0;

    comp_fields(N, nodes);
    while (var > tol && iter < max_iter){
        var = new_averages(N, beta, lambda, nodes, tol, hmax, coefficients, iter);
        iter++;
        comp_fields(N, nodes);
        if (isinf(var) || isnan(var) || var > maximum){
            divergence = true;
            return iter;
        }
    }

    divergence = false;
    return iter;
}



void print_results_short(int iter, Tnode *nodes, long N, long seed, int max_iter, 
                         bool divergence){
    long counter = 0;
    for (long i = 0; i < N; i++){
        if (!nodes[i].converged){
            counter++;
        }
    }
    double av = average(N, nodes);
    double av_sqr = average_sqr(N, nodes);
    if (divergence){
        cout << iter << "\t" << "diverges" << "\t" << av << "\t" << sqrt((av_sqr - av * av) / N) << "\t" << counter << "\t" << seed << endl;
    }else{
        bool conv = iter < max_iter;
        cout << iter << "\t" << conv << "\t" << av << "\t" << sqrt((av_sqr - av * av) / N) << "\t" << counter << "\t" << seed << endl;
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
    char gr_str[20];

    if (gr_inside){
        N = atol(argv[11]);
        int c = atoi(argv[12]);
        gsl_rng * r;
        init_ran(r, seed);
        if (argc > 13){
            if (atoi(argv[13]) == 1){                
                init_graph_inside_RRG(nodes, N, c, eps, mu, sigma, r);
            }else if (atoi(argv[13]) == 2)
            {
                init_graph_inside_RGER_full_asym(nodes, N, c, mu, sigma, r);
            }else{
                cout << "Wrong value for the 14th argument. It must be 1 or 2." << endl;
                exit(1);
            }
            
        }else{
            init_graph_inside_RRG(nodes, N, c, eps, mu, sigma, r);
        }
    }else{
        init_graph_from_input(nodes, N);
    }

    double hmax = find_divergence(beta, (1 + beta * lambda) / 2);
    double **coefficients;
    comp_coefficients(beta, lambda, coefficients);

    init_avgs(N, nodes, avn_0);

    bool divergence = false;

    int iter = convergence(N, beta, lambda, nodes, tol, max_iter, divergence, 
                           hmax, coefficients);

    print_results_short(iter, nodes, N, seed, max_iter, divergence);
    
    return 0;
}