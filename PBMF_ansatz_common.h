#ifndef __PBMF_ANSATZ_COMMON_H_INCLUDED__
#define __PBMF_ANSATZ_COMMON_H_INCLUDED__

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
    vector <long> edges_in; // edges that contain the node
    vector <int> pos_there; // position occupied by the node in those edges
    double av_abundance; // average value of n in that node
    double var_abundance; // variance of n in that node
    double response_around_zero; // average response of the variable in that node to a small perturbation in one of its neighbors when the latter is at zero
    double av_prev_fixed_point; // previous value of av in the fixed point
    double var_prev_fixed_point; // previous value of var in the fixed point
    vector <double> Psingle; // Psingle[n_index] is the probability density of n = n_grid[n_index] in that node
    vector <double> Psingle_gaussian_part; // Psingle_gaussian_part[n_index] is the result of dividing the probability density 
    // of n = n_grid[n_index] in that node by n^(beta * lambda - 1) and renormalizing it.
    double normalization_Psingle; // normalization for Psingle, computed in the function compute_averages() and used 
    // in compute_distributions() if the parameter print_distributions is set to true.
    double normalization_Psingle_gaussian_part; // normalization for Psingle_gaussian_part, computed in the function compute_averages() and used
    // in compute_distributions() if the parameter print_distributions is set to true.
}Tnode;




typedef struct{
    long nodes_in[2]; // nodes inside the edge. nodes_in[i], with i={0, 1}.
    double links[2]; // links[i], with i={0, 1}. links[i] is the one pointing to the variable in nodes_in[i]
    vector < vector <double> > cond_av; // cond_av[i], with i = {0, 1}. cond_av[i] is the average of the variable in nodes_in[i]
    // conditioned to the variable in nodes_in[1 - i] being fixed. cond_av[i] is computed as an integral over the conditional distribution.
    vector < vector <double> > integrated_cond_av; // integrated_cond_av[i][n_index], with i = {0, 1}, is the integral of cond_av[i] 
    // from 0 to n=n_grid[n_index]
    vector < vector <long> > edges_except; // edges that contain the node in nodes_in[i] excepting this edge
    vector < vector <int> > pos_there; // position occupied by the node in nodes_in[i] in those edges
    int edge_index[2]; // position of the edge in the list of edges that contain the node
    bool converged[2]; // converged[i], with i = {0, 1}. converged[i] is true if the conditional average cond_av[i] arrived to convergence.
    double response_around_average[2]; // response_around_average[i], with i = {0, 1}. response_around_average[i] is the response of the 
    // variable in nodes_in[i] to a small perturbation around the average value of the variable in nodes_in[1 - i].
    double response_around_zero[2]; // response_around_average[i], with i = {0, 1}. response_around_average[i] is the response of the 
    // variable in nodes_in[i] to a small perturbation on the variable in nodes_in[1 - i] when the latter is at zero.
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


void init_graph_from_input_inverse(Tnode *&nodes, Tedge *&edges, long &N, long &M){
    scanf("%ld %ld", &N, &M);
    nodes = new Tnode[N];
    edges = new Tedge[M];
    long i, j;
    double aij, aji; // aij is the coupling that node j sees from node i
    for (long e = 0; e < M; e++){
        scanf("%ld %ld %lf %lf", &i, &j, &aij, &aji);
        edges[e].nodes_in[0] = i;
        edges[e].nodes_in[1] = j;
        edges[e].links[0] = aij;
        edges[e].links[1] = aji;

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


void init_avgs(long N, Tnode *nodes, double avn_0, bool random_init, double std_n0, unsigned long id_0){
    if (random_init){
        gsl_rng * r;
        init_ran(r, id_0);
        double n_i;
        for (long i = 0; i < N; i++){
            n_i = avn_0 - std_n0 + 2 * std_n0 * gsl_rng_uniform(r);
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


vector <double> init_grid(double n1, double dn, double nmax){
    vector <double> n_grid;
    n_grid.push_back(0);
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
    return simpson_weights;
}


vector <double> compute_fixed_integrand(double beta, double lambda, int p, vector <double> n_grid){
    vector <double> integrand = vector <double> (n_grid.size() - 1, 0);
    for (int i = 1; i < n_grid.size(); i++){
        integrand[i - 1] = pow(n_grid[i], beta * lambda - 1 + p) * exp(-beta * pow(n_grid[i] - 1, 2) / 2);
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
                            vector <double> &fixed_integrand_den, double beta, double lambda, double n1, double dn,
                            double nmax){
    n_grid = init_grid(n1, dn, nmax);
    simpson_weights = compute_simpson_weights(n_grid.size() - 1, dn);
    fixed_integrand_num = compute_fixed_integrand(beta, lambda, 1, n_grid);
    fixed_integrand_den = compute_fixed_integrand(beta, lambda, 0, n_grid);
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


void print_results(double av_n_graph, double error_n_graph, double av_var_n_graph, double error_var_n_graph, int iter, 
                   Tnode *nodes, Tedge *edges, long N, long M, long seed_graph,
                   long seed_seq, long seed_initcond, int max_iter, bool divergence, bool same_fixed_point, 
                   size_t elapsed, double lambda, double normfactor = 1e-14, double factor_for_deaths=1){
    long counter_diverged = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            if (!edges[e].converged[k]){
                counter_diverged++;
            }
        }
    }

    long count_dead = 0;
    for (long i = 0; i < N; i++){
        if (nodes[i].av_abundance <= lambda * factor_for_deaths){
            count_dead++;
        }
    }

    if (iter >= max_iter || divergence){
        same_fixed_point = false;
    }

    
    if (divergence){
        cout << iter << "\t" << "diverges" << "\t" << 
                av_n_graph << "\t" << error_n_graph << "\t" << 
                av_var_n_graph << "\t" << error_var_n_graph << "\t" << 
                counter_diverged << "\t" << count_dead << "\t" << seed_graph  << "\t"  << seed_seq  << "\t"  << 
                seed_initcond << "\t" << same_fixed_point << "\t" << double(elapsed) / 1000 << endl;
    }else{
        bool conv = iter < max_iter;
        cout << iter << "\t" << conv << "\t" << 
                av_n_graph << "\t" << error_n_graph << "\t" << 
                av_var_n_graph << "\t" << error_var_n_graph << "\t" << 
                counter_diverged << "\t" << count_dead << "\t" << seed_graph  << "\t"  << seed_seq  << "\t"  << 
                seed_initcond << "\t" << same_fixed_point << "\t" << double(elapsed) / 1000 << endl;
    }

}



void set_av_prev(Tnode *nodes, long N) {
    for (long i = 0; i < N; i++) {
        nodes[i].av_prev_fixed_point = nodes[i].av_abundance;
        nodes[i].var_prev_fixed_point = nodes[i].var_abundance;
    }
}


bool compare_fixed_points(Tnode *nodes, long N, double tol_fixed_point){
    double max_diff = 0.0;
    double diff;
    for (long i = 0; i < N; i++) {
        diff = fabs(nodes[i].av_abundance - nodes[i].av_prev_fixed_point);
        if (diff > max_diff) {
            max_diff = diff;
        }
        diff = fabs(nodes[i].var_abundance - nodes[i].var_prev_fixed_point);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    cerr << "Max difference in fixed points: " << max_diff << endl;
    return max_diff < tol_fixed_point;
}


void print_node_avgs_to_file(Tnode *nodes, long N, char *fileavgs){
    ofstream fav(fileavgs);
    fav << "#node\tav(n)\tvar(n)\tresponse(0)" << endl;
    for (long i = 0; i < N; i++){
        fav << i << "\t" << nodes[i].av_abundance << "\t" << nodes[i].var_abundance << "\t" << nodes[i].response_around_zero << endl;
    }
    fav.close();
    
}


void print_responses_to_file(Tedge *edges, long M, char *fileresponses){
    ofstream fresp(fileresponses);
    fresp << "#edge\tfrom_node\tto_node\tresponse(av)\tresponse(0)" << endl;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            fresp << e << "\t" << edges[e].nodes_in[1 - k] << "\t" << edges[e].nodes_in[k] << "\t" << edges[e].response_around_average[k] << "\t" << edges[e].response_around_zero[k] << endl;
        }
    }
    fresp.close();
}


void print_distributions_to_file(Tnode *nodes, vector <double> n_grid, long N, char *filedistr){
    ofstream fdist(filedistr);
    fdist << "#n\tP_1(n)...\tP_N(n)\tPgauss_1(n)...\tPgauss_N(n)" << endl;

    for (int n_index = 1; n_index < n_grid.size(); n_index++){
        fdist << n_grid[n_index];
        for (long i = 0; i < N; i++){
            fdist << "\t" << nodes[i].Psingle[n_index - 1];
        }
        for (long i = 0; i < N; i++){
            fdist << "\t" << nodes[i].Psingle_gaussian_part[n_index - 1];
        }
        fdist << endl;
    }
    fdist.close();
}



void create_graph(bool gr_inside, unsigned long seed_graph, long &N, long &M, Tnode *&nodes, Tedge *&edges, 
                  double eps, double mu, double sigma, char * gr_str, char * graph_type, 
                  double c_arg, bool alpha_inverse){
    if (gr_inside){
        gsl_rng * r;
        init_ran(r, seed_graph);
        if (graph_type == string("RRG")) {
            int c = (int) round(c_arg);
            M = init_graph_inside_RRG(nodes, edges, N, c, eps, mu, sigma, r);
            sprintf(gr_str, "gr_inside_RRG_eps_%.3lf_mu_%.3lf_sigma_%.3lf_N_%li_c_%d_seedgraph_%li", eps, mu, sigma, N, c, seed_graph);
            gsl_rng_free(r);
        }else{
            cerr << "graph_type must be RRG" << endl;
            gsl_rng_free(r);
            exit(1);
        }
    }else{
        if (alpha_inverse){
            init_graph_from_input_inverse(nodes, edges, N, M);
        }else{
            init_graph_from_input(nodes, edges, N, M);
        }
    }
}


void parse_arguments(int argc, char *argv[], double &avn_0, bool &random_init, double &std_n0, 
                     unsigned long &id_0, int &num_init_conds, double &T, double &lambda, double &tol, 
                     int &max_iter, unsigned long &seed_seq, unsigned long &num_seq,
                     double &tol_fixed_point, double &damping, bool &print_avgs, bool &print_responses,
                     bool &print_distributions, bool &print_only_last, bool &gr_inside, double &eps, double &mu,
                     double &sigma, unsigned long &seed_graph, long &N, char * graph_type,
                     double &c, char *input_graph_name, bool &print_params, bool &alpha_inverse, 
                     double &n1, double &dn, double &nmax){
    int arg_index = 1;
    while (arg_index < argc){
        if (string(argv[arg_index]) == "-h" || string(argv[arg_index]) == "--help"){
            cerr << "Usage: " << argv[0] << endl;
            cerr << "The following list describes the command line arguments" << endl;
            cerr << "the structure is --arg_name  [data_type: default]  ::  description" << endl;
            cerr << "--avn_0  [double: 0.08]  ::  the initial average abundance" << endl;
            cerr << "--random_init  [double: 0]  [unsigned long: 1]  [int: 1]  ::  it expects a double (std_n0), an unsigned long (id_0), and an int (num_init_conds). The abundances are initialized in the interval [n0-std_n0, n0+std_n0], where n0 is the average value specified with --avn_0. If --random_init is not included, the initial condition is n0 for all nodes" << endl;
            cerr << "-T  or --temp   [double: 0.01]  ::  temperature" << endl;
            cerr << "--lambda  [double: 1e-6]   ::   immigration rate (default is zero)" << endl;
            cerr << "--tol  [double: 1e-6]   ::   tolerance for the convergence of the individual abundances" << endl;
            cerr << "--max_iter   [int: 10000]   ::   maximum number of iterations" << endl;
            cerr << "--seed_seq   [unsigned long: 1]   ::   initial seed to generate the update sequence" << endl;
            cerr << "--num_seq   [int: 1]   ::   number of different sequences to try" << endl;
            cerr << "--tol_fp   [double: 1e-2]   ::   maximum allowed difference between individual abundances to determine that two fixed points are equal" << endl;
            cerr << "--damping   [double: 1.0]   :: damping for the convergence process. Setting it to 1 means no damping" << endl;
            cerr << "--print_avgs   ::   if this flag is added to the arguments, the program will print the individual average abundances" << endl;
            cerr << "--print_resp   ::   if this flag is added to the arguments, the program will print the individual responses" << endl;
            cerr << "--print_distr   ::   if this flag is added to the arguments, the program will print the abundances distribution for all nodes" << endl;
            cerr << "--print_only_last  ::  if this flag is added to the arguments, the program prints only the information obtained by running the convergence process with the last sequence (with seed 'seed_seq+num_seq-1')" << endl;
            cerr << "--gr_inside  ::  it this flag is added to the arguments, the program will generate the interaction graph. If not, it will expect the graph from standard input" << endl;
            cerr << "--eps   [double: 1.0]  ::   level of asymmetry in the graph (only needed if --gr_inside is set)" << endl;
            cerr << "--mu  [double: 0.2]   ::   average strength of the interactions (only needed if --gr_inside is set)" << endl;
            cerr << "--sigma  [double: 0.0]  ::  standard deviation of the interactions (only needed if --gr_inside is set)" << endl;
            cerr << "--seed_graph  [unsigned long: 1]  ::  seed for the generation of the graph  (only needed if --gr_inside is set)" << endl;
            cerr << "-N or --size  [long: 1024]  ::  number of species in the system  (only needed if --gr_inside is set)" << endl;
            cerr << "-c or --connect  [int or double, depending on graph type: 3]  ::  average connectivity of the interaction graph  (only needed if --gr_inside is set)" << endl;
            cerr << "--graph_type  [string: RRG]  ::  if graph_type=RRG, the program generates a random regular graph (only needed if --gr_inside is set)" << endl;
            cerr << "--input_graph_name  [string]  ::  name of the input graph to insert in the output files (only needed if --gr_inside is not set and --print_avgs is set)" << endl;
            cerr << "--print_params  ::  if this flag is added to the arguments, the program will print the parameters used for the run" << endl;
            cerr << "--alpha_inverse  ::  the program will read the input graph assuming that the interactions are given in the inverse order (is makes sense only if --gr_inside is not set)." << endl;
            cerr << "--n1 [double: 1e-4] :: second point of the grid for the distributions. The first point is always 0, and the second" << endl;
            cerr << "--dn [double: 5e-3] :: step of the grid for the distributions. The points are generated as (0, n1, n1+dn, n1+2*dn, ...)" << endl;
            cerr << "--nmax [double: 2.0] :: maximum value of the grid for the distributions" << endl;
            exit(0);
        }
        if (string(argv[arg_index]) == "--avn_0"){
            arg_index++;
            avn_0 = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--random_init"){
            random_init = true;
            arg_index++;
            std_n0 = atof(argv[arg_index]);
            arg_index++;
            id_0 = atol(argv[arg_index]);
            arg_index++;
            num_init_conds = atoi(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "-T" || string(argv[arg_index]) == "--temp"){
            arg_index++;
            T = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--lambda"){
            arg_index++;
            lambda = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--tol"){
            arg_index++;
            tol = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--max_iter"){
            arg_index++;
            max_iter = atoi(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--seed_seq"){
            arg_index++;
            seed_seq = atol(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--num_seq"){
            arg_index++;
            num_seq = atol(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--tol_fp"){
            arg_index++;
            tol_fixed_point = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--damping"){
            arg_index++;
            damping = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--print_avgs"){
            print_avgs = true;
            arg_index++;
        }else if (string(argv[arg_index]) == "--print_resp"){
            print_responses = true;
            arg_index++;
        }else if (string(argv[arg_index]) == "--print_distr"){
            print_distributions = true;
            arg_index++;
        }else if (string(argv[arg_index]) == "--print_only_last"){
            print_only_last = true;
            arg_index++;
        }else if (string(argv[arg_index]) == "--gr_inside"){
            gr_inside = true;
            arg_index++;
        }else if (string(argv[arg_index]) == "--eps"){
            arg_index++;
            eps = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--mu"){
            arg_index++;
            mu = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--sigma"){
            arg_index++;
            sigma = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--seed_graph"){
            arg_index++;
            seed_graph = atol(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "-N" || string(argv[arg_index]) == "--size"){
            arg_index++;
            N = atol(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "-c" || string(argv[arg_index]) == "--connect"){
            arg_index++;
            c = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--graph_type"){
            arg_index++;
            sprintf(graph_type, "%s", argv[arg_index]);
            if (string(graph_type) != "RRG"){
                cerr << "graph_type must be RRG" << endl;
                exit(1);
            }
            arg_index++;
        }else if (string(argv[arg_index]) == "--input_graph_name"){
            arg_index++;
            sprintf(input_graph_name, "%s", argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--print_params"){
            print_params = true;
            arg_index++;
        }else if (string(argv[arg_index]) == "--alpha_inverse"){
            alpha_inverse = true;
            arg_index++;
        }else if (string(argv[arg_index]) == "--n1"){
            arg_index++;
            n1 = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--dn"){
            arg_index++;
            dn = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--nmax"){
            arg_index++;
            nmax = atof(argv[arg_index]);
            arg_index++;
        }else{
            cerr << "Unknown argument: " << argv[arg_index] << endl;
            exit(1);
        }
    }
}


void print_params_run(double avn_0, bool random_init, double std_n0,
                     unsigned long id_0, int num_init_conds, double T, double lambda, double tol, 
                     int max_iter, unsigned long seed_seq, unsigned long num_seq,
                     double tol_fixed_point, double damping, bool print_avgs, bool print_responses, 
                     bool print_distributions, bool print_only_last, bool gr_inside, double eps, double mu,
                     double sigma, unsigned long seed_graph, long N, char * graph_type,
                     double c, char *input_graph_name, bool alpha_inverse, double n1, double dn, double nmax){
    cerr << "Initial average abundance: " << avn_0 << endl;
    cerr << "Random initial condition std_n0=" << std_n0 << "   extracted  " << num_init_conds << " times, with initial seed " << id_0 << endl;
    cerr << "Temperature: " << T << endl;
    cerr << "lambda: " << lambda << endl;
    cerr << "Tolerance for convergence: " << tol << endl;
    cerr << "Maximum number of iterations: " << max_iter << endl;
    cerr << "Initial seed for the update sequence: " << seed_seq << endl;
    cerr << "Number of different sequences to try: " << num_seq << endl;
    cerr << "Tolerance to determine if two fixed points are equal: " << tol_fixed_point << endl;
    cerr << "Damping for the convergence process: " << damping << endl;
    cerr << "Parameters to build the grid: n1=" << n1 << "   dn=" << dn << "   nmax=" << nmax << endl;
    cerr << "The grid is generated as (0, n1, n1+dn, n1+2*dn, ...), and the distributions are computed in those points" << endl;
    if (print_avgs){
        cerr << "The program will print the individual average abundances" << endl;
    }else{
        cerr << "The program will not print the individual average abundances" << endl;
    }
    if (print_responses){
        cerr << "The program will print the individual responses" << endl;
    }else{
        cerr << "The program will not print the individual responses" << endl;
    }
    if (print_distributions){
        cerr << "The program will print the abundances distribution for all nodes" << endl;
    }else{
        cerr << "The program will not print the abundances distribution for all nodes" << endl;
    }
    if (print_only_last){
        cerr << "The program will print only the information obtained by running the convergence process with the last sequence" << endl;
    }else{
        cerr << "The program will print the information obtained by running the convergence process with all sequences and initial conditions" << endl;
    }
    if (gr_inside){
        cerr << "The program will generate the interaction graph" << endl;
        cerr << "Level of asymmetry in the graph: " << eps << endl;
        cerr << "Average strength of the interactions: " << mu << endl;
        cerr << "Standard deviation of the interactions: " << sigma << endl;
        cerr << "Seed for the generation of the graph: " << seed_graph << endl;
        cerr << "Number of species in the system: " << N << endl;
        cerr << "Average connectivity of the interaction graph: " << c << endl;
        cerr << "Type of graph to generate: " << graph_type << endl;
    }else{
        cerr << "The program will read the interaction graph from standard input" << endl;      
        cerr << "Name of the input graph to insert in the output files: " << input_graph_name << endl;
        if (alpha_inverse){
            cerr << "The program will read the input graph assuming that the interactions are given in the inverse order" << endl;
        }
    }
}


#endif