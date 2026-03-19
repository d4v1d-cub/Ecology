#ifndef __GMF_COMMON_H_INCLUDED__
#define __GMF_COMMON_H_INCLUDED__

/**
 * @file GMF_common.h
 * @brief Common utilities and data structures for Gaussian Mean Field (GMF) analysis
 * of Generalized Lotka-Volterra dynamics on sparse graphs.
 * 
 * This file implements the core functionality for analyzing species interactions
 * in ecological networks using the GMF approach. The implementation focuses on
 * finding stationary solutions to the local Fokker-Planck equation that describes
 * the species abundance dynamics.
 */

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <gsl/gsl_randist.h>    // For random number generation
#include <gsl/gsl_sf_hyperg.h>  // For hypergeometric functions
#include <gsl/gsl_sf_gamma.h>   // For gamma functions
#include <cmath>

using namespace std;

/**
 * @brief Initialize the GSL random number generator
 * @param r Reference to the random number generator
 * @param s Seed for the random number generator
 */
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


void init_avgs(long M, Tedge *edges, double avn_0, double chi_0, bool random_init, double dn, double dchi, 
               unsigned long id_0){
    if (random_init){
        gsl_rng * r;
        init_ran(r, id_0);
        double cav_i, chi_i;
        for (long e = 0; e < M; e++){
            for (int k = 0; k < 2; k++){
                cav_i = avn_0 - dn + 2 * dn * gsl_rng_uniform(r);
                chi_i = chi_0 - dchi + 2 * dchi * gsl_rng_uniform(r);
                if (cav_i < 0){
                    cav_i = 0;
                }
                if (chi_i < 0){
                    chi_i = 0;
                }
                edges[e].cond_av[k] = cav_i;
                edges[e].chi_cav[k] = chi_i;
            }
        }
        gsl_rng_free(r);
    }else{
        for (long e = 0; e < M; e++){
            for (int k = 0; k < 2; k++){
                edges[e].cond_av[k] = avn_0;
                edges[e].chi_cav[k] = chi_0;
            }
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



void print_results(double av, int iter, Tnode *nodes, Tedge *edges, long N, long M, long seed_graph,
                   long seed_seq, long seed_initcond, int max_iter, bool divergence, bool same_fixed_point, 
                   size_t elapsed, double normfactor = 1e-14){
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
                count_dead << "\t" << seed_graph  << "\t"  << seed_seq  << "\t"  << 
                seed_initcond << "\t" << same_fixed_point << "\t" << double(elapsed) / 1000 << endl;
    }else{
        bool conv = iter < max_iter;
        cout << iter << "\t" << conv << "\t" << 
                av_cav << "\t" << sqrt(fabs(av_cav_sqr - av_cav * av_cav) / 2 / M) << "\t" << 
                av_chi_cav << "\t" << sqrt(fabs(av_chi_cav_sqr - av_chi_cav * av_chi_cav) / 2 / M) << "\t" << 
                av << "\t" << sqrt(fabs(av_sqr - av * av) / N) << "\t" << 
                av_chi << "\t" << sqrt(fabs(av_chi_sqr - av_chi * av_chi) / N) << "\t" << 
                counter_diverged << "\t" << counter_varneg << "\t" << counter_chi_cav_diverged << "\t" << 
                count_dead << "\t" << seed_graph  << "\t"  << seed_seq  << "\t"  << 
                seed_initcond << "\t" << same_fixed_point << "\t" << double(elapsed) / 1000 << endl;
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


void print_avgs_to_file(Tnode *nodes, long N, char *fileavgs){
    ofstream fav(fileavgs);
    for (long i = 0; i < N; i++){
        fav << i << "\t" << nodes[i].av << "\t" << nodes[i].chi << endl;
    }
    fav.close();
    
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
            cerr << "graph_type must be RRG or ER" << endl;
            gsl_rng_free(r);
            exit(1);
        }
    }else{
        long M;
        if (alpha_inverse){
            init_graph_from_input_inverse(nodes, edges, N, M);
        }else{
            init_graph_from_input(nodes, edges, N, M);
        }
    }
}


void parse_arguments(int argc, char *argv[], double &avn_0, double &chi_0, bool &random_init, double &dn, double &dchi, 
                     unsigned long &id_0, int &num_init_conds, double &T, double &lambda, double &tol, 
                     int &max_iter, unsigned long &seed_seq, unsigned long &num_seq,
                     double &tol_fixed_point, double &damping, bool &print_avgs,
                     bool &print_only_last, bool &gr_inside, double &eps, double &mu,
                     double &sigma, unsigned long &seed_graph, long &N, char * graph_type,
                     double &c, char *input_graph_name, bool &print_params, bool &alpha_inverse){
    int arg_index = 1;
    while (arg_index < argc){
        if (string(argv[arg_index]) == "-h" || string(argv[arg_index]) == "--help"){
            cerr << "Usage: " << argv[0] << endl;
            cerr << "The following list describes the command line arguments" << endl;
            cerr << "the structure is --arg_name  [data_type: default]  ::  description" << endl;
            cerr << "--avn_0  [double: 0.08]  ::  the initial average abundance" << endl;
            cerr << "--chi_0  [double: 0.08]  ::  the initial average response" << endl;
            cerr << "--random_init  [double: 0]  [double: 0]  [unsigned long: 1]  [int: 1]  ::  it expects a double (dn), a double (dchi), an unsigned long (id_0), and an int (num_init_conds). The abundances are initialized in the interval [n0-dn, n0+dn], where n0 is the average value specified with --avn_0. The responses are initialized in the interval [chi0-dchi, chi0+dchi], where chi0 is the average value specified with --chi_0. The initial conditions are drawn for 'num_init_conds' different seeds of the random number generator, starting at 'id_0'. If --random_init is not included, the initial condition is n0 for all nodes" << endl;
            cerr << "-T  or --temp   [double: 0.01]  ::  temperature" << endl;
            cerr << "--lambda  [double: 1e-6]   ::   immigration rate (default is zero)" << endl;
            cerr << "--tol  [double: 1e-6]   ::   tolerance for the convergence of the individual abundances" << endl;
            cerr << "--max_iter   [int: 10000]   ::   maximum number of iterations" << endl;
            cerr << "--seed_seq   [unsigned long: 1]   ::   initial seed to generate the update sequence" << endl;
            cerr << "--num_seq   [int: 1]   ::   number of different sequences to try" << endl;
            cerr << "--tol_fp   [double: 1e-2]   ::   maximum allowed difference between individual abundances to determine that two fixed points are equal" << endl;
            cerr << "--damping   [double: 1.0]   :: damping for the convergence process. Setting it to 1 means no damping" << endl;
            cerr << "--print_avgs   ::   if this flag is added to the arguments, the program will print individual average abundances" << endl;
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
            exit(0);
        }
        if (string(argv[arg_index]) == "--avn_0"){
            arg_index++;
            avn_0 = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--chi_0"){
            arg_index++;
            chi_0 = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--random_init"){
            random_init = true;
            arg_index++;
            dn = atof(argv[arg_index]);
            arg_index++;
            dchi = atof(argv[arg_index]);
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
        }else{
            cerr << "Unknown argument: " << argv[arg_index] << endl;
            exit(1);
        }
    }
}


void print_params_run(double avn_0, double chi_0, bool random_init, double dn, double dchi,
                     unsigned long id_0, int num_init_conds, double T, double lambda, double tol, 
                     int max_iter, unsigned long seed_seq, unsigned long num_seq,
                     double tol_fixed_point, double damping, bool print_avgs,
                     bool print_only_last, bool gr_inside, double eps, double mu,
                     double sigma, unsigned long seed_graph, long N, char * graph_type,
                     double c, char *input_graph_name, bool alpha_inverse){
    cerr << "Initial average abundance: " << avn_0 << endl;
    cerr << "Initial average response: " << chi_0 << endl;
    cerr << "Random initial condition dn=" << dn << " dchi=" << dchi << "   extracted  " << num_init_conds << " times, with initial seed " << id_0 << endl;
    cerr << "Temperature: " << T << endl;
    cerr << "lambda: " << lambda << endl;
    cerr << "Tolerance for convergence: " << tol << endl;
    cerr << "Maximum number of iterations: " << max_iter << endl;
    cerr << "Initial seed for the update sequence: " << seed_seq << endl;
    cerr << "Number of different sequences to try: " << num_seq << endl;
    cerr << "Tolerance to determine if two fixed points are equal: " << tol_fixed_point << endl;
    cerr << "Damping for the convergence process: " << damping << endl;
    if (print_avgs){
        cerr << "The program will print individual average abundances" << endl;
    }else{
        cerr << "The program will not print individual average abundances" << endl;
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