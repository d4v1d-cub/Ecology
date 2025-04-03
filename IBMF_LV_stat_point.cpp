#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include "math.h"

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


void init_graph_inside_RRG(Tnode *&nodes, long N, int c, double eps,
                           double mu, double sigma, gsl_rng * r){
    // eps is the degree of symmetry of the graph
    if (N * c % 2 != 0){
        cout << "N*c must be even to create a random regular graph" << endl;
        exit(1);
    }else{
        nodes = new Tnode [N];
        long M = N * c / 2;
        long pos_i, pos_j, i, j;
        double aij, aji;
        vector < long > copies = vector < long > (c * N);
        for (long i = 0; i < N; i++){
            for (int k = 0; k < c; k++){
                copies[i * c + k] = i;
            }
        }

        for (long e = 0; e < M; e++){
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
    }
}


void init_avgs(long N, double *&avgs, double avn_0){
    avgs = new double[N];
    for (long i = 0; i < N; i++){
        avgs[i] = avn_0;
    }
}


double field_in(long i, double *avgs, vector <long> neighs, vector <double> links_in){
    double field = 0;
    for (long j = 0; j < neighs.size(); j++){
        field += links_in[j] * avgs[neighs[j]];
    }
    return 1 - field;
}


double numerator(double beta, double lambda, double hi){
        return gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2) +
     sqrt(2 * beta) * hi * gsl_sf_gamma(1 + beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2);
}


double denominator(double beta, double lambda, double hi){
    return sqrt(beta / 2) * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2) + 
    beta * hi * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2);
}


double new_averages(long N, double *avgs, double *avgs_new, double beta, double lambda, Tnode *nodes){
    double hi, var = 0, var_i;
    for (long i = 0; i < N; i++){
        hi = field_in(i, avgs, nodes[i].neighs, nodes[i].links_in);
        avgs_new[i] = numerator(beta, lambda, hi) / denominator(beta, lambda, hi);
        var_i = fabs(avgs_new[i] - avgs[i]);
        if (var_i > var){
            var = var_i;
        }
    }
    return var;
}

double average(long N, double *avgs){
    double av = 0;
    for (long i = 0; i < N; i++){
        av += avgs[i];
    }
    return av / N;
}

double average_sqr(long N, double *avgs){
    double av_sqr = 0;
    for (long i = 0; i < N; i++){
        av_sqr += avgs[i] * avgs[i];
    }
    return av_sqr / N;
}

int convergence(long N, double *avgs, double beta, double lambda, Tnode *nodes, double tol, 
                 int max_iter, char *filehist){
    double *avgs_new;
    avgs_new = new double[N];
    double var = tol + 1;
    int iter = 0;

    ofstream fh(filehist);

    fh << "# iter\tmax(dn)\tav(n)" << endl;
    while (var > tol && iter < max_iter){
        var = new_averages(N, avgs, avgs_new, beta, lambda, nodes);
        for (long i = 0; i < N; i++){
            avgs[i] = avgs_new[i];
        }
        iter++;
        fh << iter << "\t" << var << "\t" << average(N, avgs) << endl;
    }

    fh.close();

    return iter;
}


int convergence(long N, double *avgs, double beta, double lambda, Tnode *nodes, double tol, 
                 int max_iter){
    double *avgs_new;
    avgs_new = new double[N];
    double var = tol + 1;
    int iter = 0;

    while (var > tol && iter < max_iter){
        var = new_averages(N, avgs, avgs_new, beta, lambda, nodes);
        for (long i = 0; i < N; i++){
            avgs[i] = avgs_new[i];
        }
        iter++;
    }

    return iter;
}


void print_results(int iter, double *avgs, long N, long seed, int max_iter, char *fileout){
    double av = average(N, avgs);
    double av_sqr = average_sqr(N, avgs);
    ofstream fo(fileout, ios::app);
    fo << "#Niter=" << "\t" << iter << endl;
    if (iter < max_iter){
        fo << "#convergence=" << "\t" << "true" << endl;
    }else{
        fo << "#convergence=" << "\t" << "false" << endl;
    }
    fo << "#av(n)=" << "\t" << av << endl;
    fo << "#std(n)=" << "\t" << sqrt((av_sqr - av * av) / N) << endl;
    fo << "#seed=" << "\t" << seed << endl;
    for (long i = 0; i < N; i++){
        fo << i << "\t" << avgs[i] << endl;
    }
    fo.close();
}


void print_results_short(int iter, double *avgs, long N, long seed, int max_iter){
    double av = average(N, avgs);
    double av_sqr = average_sqr(N, avgs);
    bool conv = iter < max_iter;
    cout << iter << "\t" << conv << "\t" << av << "\t" << sqrt((av_sqr - av * av) / N) << "\t" << seed << endl;
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


    Tnode *nodes;
    double *avgs;
    double beta = 1.0 / T;
    long N;
    char gr_str[20];

    if (gr_inside){
        sprintf(gr_str, "gr_inside_RRG");
        N = atol(argv[11]);
        int c = atoi(argv[12]);
        gsl_rng * r;

        init_ran(r, seed);

        init_graph_inside_RRG(nodes, N, c, eps, mu, sigma, r);
    }else{
        sprintf(gr_str, "gr_from_input");
        init_graph_from_input(nodes, N);
    }


    // char filehist[200];
    // sprintf(filehist, "IBMF_Lotka_Volterra_steady_state_convergence_%s_T_%.3lf_lambda_%.3lf_av0_%.3lf_tol_%.1e_maxiter_%d_eps_%.3lf_mu_%.3lf_sigma_%.3lf.txt", 
    //                   gr_str, T, lambda, avn_0, tol, max_iter, eps, mu, sigma);

    char fileout[200];
    sprintf(fileout, "IBMF_Lotka_Volterra_steady_state_avn_%s_T_%.3lf_lambda_%.3lf_av0_%.3lf_tol_%.1e_maxiter_%d_eps_%.3lf_mu_%.3lf_sigma_%.3lf.txt", 
                      gr_str, T, lambda, avn_0, tol, max_iter, eps, mu, sigma);


    init_avgs(N, avgs, avn_0);

    int iter = convergence(N, avgs, beta, lambda, nodes, tol, max_iter);

    print_results(iter, avgs, N, seed, max_iter, fileout);
    
    return 0;
}