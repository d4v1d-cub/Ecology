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
            sqrt(beta / 2) * hi * beta * lambda * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2);
}


double denominator(double beta, double lambda, double hi, double normfactor = 1e-14){
    return sqrt(beta / 2) * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2) + 
            beta * hi * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2)
            + normfactor;
}


double numerator_asymp(double beta, double lambda, double hi){
    return sqrt(lambda) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2) +
           beta * lambda * hi * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2);
}


double denominator_asymp(double beta, double lambda, double hi, double normfactor = 1e-14){
    return gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2) + 
           beta * sqrt(lambda) * hi * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2) + 
           normfactor;
}


int check_wich_diverges(double beta, double lambda, double hi, double limit = 1e+10){
    if (isnan(gsl_sf_gamma(1 + beta * lambda / 2)) || isinf(gsl_sf_gamma(1 + beta * lambda / 2)) || gsl_sf_gamma(1 + beta * lambda / 2) > limit){
        if (isnan(gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2)) 
            || gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        }else if(isnan(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2))
                 || gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        }else if(isnan(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2))
                 || gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        }else if(isnan(gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2)) || 
                 gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        }else{
            return 2; // only gamma diverges
        }
    }else if (isnan(gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2)) 
            || gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2) > limit){
        return 1; // gamma and hypergeometric diverge
    }else if(isnan(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2))
                 || gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2) > limit){
        return 1; // gamma and hypergeometric diverge
    }else if(isnan(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2))
                 || gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2) > limit){
        return 1; // gamma and hypergeometric diverge
    }else if(isnan(gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2)) || 
                 gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2) > limit){
        return 1; // hypergeometric diverges
    }else{
        return 0; // cannot identify divergence
    }
}


double new_averages(long N, double *avgs, double *avgs_new, double beta, double lambda, Tnode *nodes, double tol, 
                    double normfactor = 1e-10){
    double var = 0, var_i;
    int identify_divergence = 0;
    for (long i = 0; i < N; i++){
        avgs_new[i] = numerator(beta, lambda, nodes[i].field) / denominator(beta, lambda, nodes[i].field, normfactor);
        if (isnan(avgs_new[i]) || isinf(avgs_new[i])){
            identify_divergence = check_wich_diverges(beta, lambda, nodes[i].field);
            if (identify_divergence == 1){
                if (nodes[i].field < 0){
                    avgs_new[i] = 0;
                }else{
                    avgs_new[i] = nodes[i].field;
                }
            }else if (identify_divergence == 2){
                avgs_new[i] = numerator_asymp(beta, lambda, nodes[i].field) / denominator_asymp(beta, lambda, nodes[i].field, normfactor);
            }else{
                cout << "Cannot identify divergence for node " << i << endl;
                exit(1);
            }
        }

        var_i = fabs(avgs_new[i] - avgs[i]);
        if (var_i > var){
            var = var_i;
        }
        if (var_i < tol){
            nodes[i].converged = true;
        }
        else{
            nodes[i].converged = false;
        }

    }
    return var;
}


void comp_fields(long N, double *avgs, Tnode *nodes){
    for (long i = 0; i < N; i++){
        nodes[i].field = field_in(i, avgs, nodes[i].neighs, nodes[i].links_in);
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

int convergence(long N, double *avgs, double beta, double lambda, Tnode *nodes, double tol, 
                 int max_iter, char *filehist, char *filefield_hist, int print_every, 
                 bool &divergence){
    double *avgs_new;
    avgs_new = new double[N];
    double var = tol + 1;
    int iter = 0;

    ofstream fh(filehist);
    ofstream ffieldh(filefield_hist);

    fh << "# iter\tmax(dn)\tav(n)" << endl;
    ffieldh << "# iter\tav(n)..." << endl;

    comp_fields(N, avgs, nodes);

    while (var > tol && iter < max_iter){
        var = new_averages(N, avgs, avgs_new, beta, lambda, nodes, tol);
        for (long i = 0; i < N; i++){
            avgs[i] = avgs_new[i];
        }
        iter++;
        comp_fields(N, avgs, nodes);
        if (isinf(var)){
            divergence = true;
            return iter;
        }
        if (iter % print_every == 0){
            fh << iter << "\t" << var << "\t" << average(N, nodes) << endl;
            ffieldh << iter;
            for (long i  = 0; i < N; i++){
                ffieldh << "\t" << nodes[i].field;
            }
            ffieldh << endl;
        }
    }

    fh.close();
    ffieldh.close();

    divergence = false;
    return iter;
}


void print_results(int iter, Tnode *nodes, long N, long seed, int max_iter, char *filefield, 
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
        cout << iter << "\t" << "diverges" << "\t" << av << "\t" << sqrt((av_sqr - av * av) / N) << "\t" << counter << "\t" << seed << endl;;
    }else{
        bool conv = iter < max_iter;
        cout << iter << "\t" << conv << "\t" << av << "\t" << sqrt((av_sqr - av * av) / N) << "\t" << counter << "\t" << seed << endl;

        ofstream ffield(filefield);
        for (long i = 0; i < N; i++){
            ffield << i << "\t" << nodes[i].field << endl;
        }
        ffield.close();
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
    int print_every = atoi(argv[10]);
    bool gr_inside = atoi(argv[11]);

    gsl_set_error_handler_off();

    Tnode *nodes;
    double *avgs;
    double beta = 1.0 / T;
    long N;
    char gr_str[100];

    if (gr_inside){
        if (argc > 14){
            if (atoi(argv[14]) == 1){
                N = atol(argv[12]);
                int c = atoi(argv[13]);
                sprintf(gr_str, "gr_inside_RRG_N_%li_c_%d", N, c);
                gsl_rng * r;

                init_ran(r, seed);

                init_graph_inside_RRG(nodes, N, c, eps, mu, sigma, r);
            }else if (atoi(argv[14]) == 2){
                N = atol(argv[12]);
                double c = atof(argv[13]);
                sprintf(gr_str, "gr_inside_ER_fully_asym_N_%li_c_%.3lf", N, c);
                gsl_rng * r;
                init_ran(r, seed);
                init_graph_inside_RGER_full_asym(nodes, N, c, mu, sigma, r);
            }else{
                cout << "Wrong value for the 14th argument. It must be 1 or 2." << endl;
                exit(1);
            }
            
        }else{
            N = atol(argv[12]);
            int c = atoi(argv[13]);
            sprintf(gr_str, "gr_inside_RRG_N_%li_c_%d", N, c);
            gsl_rng * r;

            init_ran(r, seed);

            init_graph_inside_RRG(nodes, N, c, eps, mu, sigma, r);
        }
    }else{
        sprintf(gr_str, "gr_from_input");
        init_graph_from_input(nodes, N);
    }


    char filehist[400];
    sprintf(filehist, "IBMF_Lotka_Volterra_steady_state_convergence_%s_T_%.3lf_lambda_%.3lf_av0_%.3lf_tol_%.1e_maxiter_%d_eps_%.3lf_mu_%.3lf_sigma_%.3lf_print_every_%d_seed_%li.txt", 
                      gr_str, T, lambda, avn_0, tol, max_iter, eps, mu, sigma, print_every, seed);


    char filefield_hist[400];
    sprintf(filefield_hist, "IBMF_Lotka_Volterra_avn_hist_%s_T_%.3lf_lambda_%.3lf_av0_%.3lf_tol_%.1e_maxiter_%d_eps_%.3lf_mu_%.3lf_sigma_%.3lf_print_every_%d_seed_%li.txt", 
                          gr_str, T, lambda, avn_0, tol, max_iter, eps, mu, sigma, print_every, seed);

    char filefield[400];
    sprintf(filefield, "IBMF_Lotka_Volterra_steady_state_avn_%s_T_%.3lf_lambda_%.3lf_av0_%.3lf_tol_%.1e_maxiter_%d_eps_%.3lf_mu_%.3lf_sigma_%.3lf_seed_%li.txt", 
                      gr_str, T, lambda, avn_0, tol, max_iter, eps, mu, sigma, seed);


    init_avgs(N, avgs, avn_0);

    bool divergence;

    int iter = convergence(N, avgs, beta, lambda, nodes, tol, max_iter, filehist, filefield_hist, print_every, divergence);

    print_results(iter, nodes, N, seed, max_iter, filefield, divergence);
    
    return 0;
}