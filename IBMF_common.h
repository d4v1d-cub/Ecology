#ifndef __IBMF_COMMON_H_INCLUDED__
#define __IBMF_COMMON_H_INCLUDED__

/**
 * @file IBMF_common.h
 * @brief Common utilities and data structures for Individual Based Mean Field (IBMF) analysis
 * of Generalized Lotka-Volterra dynamics on sparse graphs.
 * 
 * This file implements the core functionality for analyzing species interactions
 * in ecological networks using the IBMF approach. The implementation focuses on
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
#include "math.h"
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

/**
 * @brief Node structure representing a species in the ecosystem
 * 
 * This structure contains all information needed to represent a species
 * in the IBMF approximation of the Lotka-Volterra dynamics:
 * - Network topology (neighbors and interaction strengths)
 * - Current state (field and abundance)
 * - Convergence information
 */
typedef struct{
    vector <long> neighs;         ///< Indices of neighboring species
    vector <double> links_in;     ///< Interaction strengths of incoming links
    double field;               ///< Local field (average abundance) at this node
    bool converged;             ///< Whether node has reached convergence
    double av;                  ///< Current average abundance
    double av_prev_fixed_point; ///< Previous fixed point for comparison
}Tnode;


/**
 * @brief Initialize interaction network from standard input
 * @param nodes Reference to array of nodes representing species
 * @param N Number of species (nodes) in the network
 * 
 * Expected input format:
 * N M          // Number of nodes and edges
 * i j aij aji  // Edge data: nodes i,j and their interaction strengths
 * ...          // (M lines of edge data)
 */
void init_graph_from_input(Tnode *&nodes, long &N){
    long M;  // Number of edges
    scanf("%ld %ld", &N, &M);
    nodes = new Tnode[N];
    long i, j;
    double aij, aji;  // Interaction strengths
    for (long e = 0; e < M; e++){
        scanf("%ld %ld %lf %lf", &i, &j, &aij, &aji);
        nodes[i].neighs.push_back(j);
        nodes[j].neighs.push_back(i);
        nodes[i].links_in.push_back(aji);  // Note: aji is incoming to i
        nodes[j].links_in.push_back(aij);  // aij is incoming to j
    }
}

/**
 * @brief Initialize interaction network from input with inverse interaction order
 * @param nodes Reference to array of nodes representing species
 * @param N Number of species (nodes) in the network
 * 
 * Similar to init_graph_from_input but swaps the interpretation of aij and aji
 * in the input format to support different conventions in input data.
 */
void init_graph_from_input_inverse(Tnode *&nodes, long &N){
    long M;
    scanf("%ld %ld", &N, &M);
    nodes = new Tnode[N];
    long i, j;
    double aij, aji;
    for (long e = 0; e < M; e++){
        scanf("%ld %ld %lf %lf", &i, &j, &aij, &aji);
        nodes[i].neighs.push_back(j);
        nodes[j].neighs.push_back(i);
        nodes[i].links_in.push_back(aij);  // Note: aij is incoming to i
        nodes[j].links_in.push_back(aji);  // aji is incoming to j
    }
}

/**
 * @brief Initialize empty node vectors for a network
 * @param nodes Array of nodes to initialize
 * @param N Number of nodes to initialize
 * 
 * Creates empty neighbor and interaction strength vectors for each node.
 * Called before building network structure to ensure clean initialization.
 */
void init_nodes(Tnode *nodes, long N){
    for (long i = 0; i < N; i++){
        nodes[i].links_in = vector <double> ();
        nodes[i].neighs = vector <long> ();
    }
}

/**
 * @brief Generate a Random Regular Graph (RRG) with specified parameters
 * @param nodes Reference to array of nodes to store the generated network
 * @param N Number of species (nodes)
 * @param c Degree (number of interactions) per node
 * @param eps Symmetry parameter (0-1) - probability of symmetric interactions
 * @param mu Mean interaction strength
 * @param sigma Standard deviation of interaction strengths
 * @param r Random number generator
 * 
 * Creates a random c-regular graph where each species interacts with exactly c others.
 * Interaction strengths are drawn from N(mu,sigma²). When eps=1, interactions are
 * symmetric (aij=aji). When eps=0, they are fully independent.
 * 
 * The algorithm uses the configuration model:
 * 1. Create c "stubs" for each node
 * 2. Randomly pair stubs to form edges
 * 3. Generate interaction strengths for each edge
 */
void init_graph_inside_RRG(Tnode *&nodes, long N, int c, double eps,
                           double mu, double sigma, gsl_rng * r){
    // Ensure valid parameters for RRG construction
    if (N * c % 2 != 0){
        cerr << "N*c must be even to create a random regular graph" << endl;
        exit(1);
    }else{
        bool success = false;
        long M = N * c / 2;  // Total number of edges
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
            nodes[i].av = n_i;
        }
        gsl_rng_free(r);
    }else{
        for (long i = 0; i < N; i++){
            nodes[i].av = avn_0;
        }
    }
}


/**
 * @brief Calculate the local field at a node
 * @param i Index of the node
 * @param nodes Array of all nodes
 * @return Value of the local field h_i = 1 - sum_j a_ij n_j
 * 
 * The local field determines the dynamics of species i through the
 * Lotka-Volterra equations. At steady state, positive fields indicate
 * survival while negative fields lead to extinction.
 */
double field_in(long i, Tnode *nodes){
    double field = 0;
    for (long j = 0; j < nodes[i].neighs.size(); j++){
        field += nodes[i].links_in[j] * nodes[nodes[i].neighs[j]].av;
    }
    return 1 - field;  // 1 is the carrying capacity
}

/**
 * @brief Calculate mean abundance across all species
 * @param N Number of species
 * @param nodes Array of nodes
 * @return Average abundance <n>
 * 
 * The mean abundance is a key observable that characterizes
 * the overall state of the ecosystem.
 */
double average(long N, Tnode *nodes){
    double av = 0;
    for (long i = 0; i < N; i++){
        av += nodes[i].av;
    }
    return av / N;
}

/**
 * @brief Calculate mean squared abundance
 * @param N Number of species
 * @param nodes Array of nodes
 * @return Average squared abundance <n²>
 * 
 * Used together with average() to compute abundance fluctuations
 * and characterize the distribution of species abundances.
 */
double average_sqr(long N, Tnode *nodes){
    double av_sqr = 0;
    for (long i = 0; i < N; i++){
        av_sqr += nodes[i].av * nodes[i].av;
    }
    return av_sqr / N;
}



void print_results_short(int iter, Tnode *nodes, long N, unsigned long seed_graph, 
                         unsigned long seed_seq, unsigned long seed_initcond,
                         int max_iter, bool divergence, bool same_fixed_point, size_t elapsed){
    long counter = 0;
    for (long i = 0; i < N; i++){
        if (!nodes[i].converged){
            counter++;
        }
    }

    long counter_dead = 0;
    for (long i = 0; i < N; i++){
        if (nodes[i].av <= 0){
            counter_dead++;
        }
    }

    if (iter >= max_iter || divergence){
        same_fixed_point = false;
    }

    double av = average(N, nodes);
    double av_sqr = average_sqr(N, nodes);
    if (divergence){
        cout << iter << "\t" << "diverges" << "\t" << av << "\t" << sqrt(fabs(av_sqr - av * av) / N) << "\t" << 
                counter << "\t" << counter_dead << "\t" << seed_graph  << "\t"  << seed_seq  << "\t"  << 
                seed_initcond << "\t" << same_fixed_point << "\t" << double(elapsed) / 1000 << endl;
    }else{
        bool conv = iter < max_iter;
        cout << iter << "\t" << conv << "\t" << av << "\t" << sqrt(fabs(av_sqr - av * av) / N) << "\t" << 
                counter << "\t" << counter_dead << "\t" << seed_graph  << "\t"  << seed_seq  << "\t"  << 
                seed_initcond << "\t" << same_fixed_point << "\t" << double(elapsed) / 1000 << endl;
    }
}


void produce_random_seq(unsigned long seed_seq, long N, long sequence[]){
    gsl_rng * r;
    init_ran(r, seed_seq);
    vector <long> elements(N);
    for (long i = 0; i < N; i++){
        elements[i] = i;
    }
    long pos;
    for (long i = 0; i < N; i++){
        pos = gsl_rng_uniform_int(r, N - i);
        sequence[i] = elements[pos];
        elements.erase(elements.begin() + pos);
    }
    gsl_rng_free(r);
}


void set_av_prev(Tnode *nodes, long N) {
    for (long i = 0; i < N; i++) {
        nodes[i].av_prev_fixed_point = nodes[i].av;
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
    }
    cerr << "Max difference in fixed points: " << max_diff << endl;
    return max_diff < tol_fixed_point;
}


void print_avgs_to_file(Tnode *nodes, long N, char *fileavgs){
    ofstream fav(fileavgs);
    for (long i = 0; i < N; i++){
        fav << i << "\t" << nodes[i].av << endl;
    }
    fav.close();
    
}


void create_graph(bool gr_inside, unsigned long seed_graph, long N, Tnode *&nodes, 
                  double eps, double mu, double sigma, char * gr_str, char * graph_type, 
                  double c_arg, bool alpha_inverse){
    if (gr_inside){
        gsl_rng * r;
        init_ran(r, seed_graph);
        if (graph_type == string("RRG")) {
            int c = (int) round(c_arg);
            init_graph_inside_RRG(nodes, N, c, eps, mu, sigma, r);
            sprintf(gr_str, "gr_inside_RRG_eps_%.3lf_mu_%.3lf_sigma_%.3lf_N_%li_c_%d_seedgraph_%li", eps, mu, sigma, N, c, seed_graph);
            gsl_rng_free(r);
        }else if (graph_type == string("ER")){
            double c = c_arg;
            init_graph_inside_RGER_full_asym(nodes, N, c, mu, sigma, r);
            sprintf(gr_str, "gr_inside_RRG_eps_%.3lf_mu_%.3lf_sigma_%.3lf_N_%li_c_%.3lf_seedgraph_%li", eps, mu, sigma, N, c, seed_graph);
            gsl_rng_free(r);
        }else{
            cerr << "graph_type must be RRG or ER" << endl;
            gsl_rng_free(r);
            exit(1);
        }

    }else{
        if (alpha_inverse){
            init_graph_from_input_inverse(nodes, N);
        }else{
            init_graph_from_input(nodes, N);
        }
    }
}


void parse_arguments(int argc, char *argv[], double &avn_0, bool &random_init, double &dn, 
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
            cerr << "--random_init  [double: 0]  [unsigned long: 1]  [int: 1]  ::  it expects a double (dn), an unsigned long (id_0), and an int (num_init_conds). The abundances are initialized in the interval [n0-dn, n0+dn], where n0 is the average value specified with --avn_0. The initial conditions are drawn for 'num_init_conds' different seeds of the random number generator, starting at 'id_0'. If --random_init is not included, the initial condition is n0 for all nodes" << endl;
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
            cerr << "--graph_type  [string: RRG]  ::  if graph_type=RRG, the program generates a random regular graph. If graph_type=ER, it generates an Erdos-Renyi graph (only needed if --gr_inside is set)" << endl;
            cerr << "--input_graph_name  [string]  ::  name of the input graph to insert in the output files (only needed if --gr_inside is not set and --print_avgs is set)" << endl;
            cerr << "--print_params  ::  if this flag is added to the arguments, the program will print the parameters used for the run" << endl;
            cerr << "--alpha_inverse  ::  the program will read the input graph assuming that the interactions are given in the inverse order (is makes sense only if --gr_inside is not set)." << endl;
            exit(0);
        }
        if (string(argv[arg_index]) == "--avn_0"){
            arg_index++;
            avn_0 = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--random_init"){
            random_init = true;
            arg_index++;
            dn = atof(argv[arg_index]);
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
            if (string(graph_type) != "RRG" && string(graph_type) != "ER"){
                cerr << "graph_type must be RRG or ER" << endl;
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


void print_params_run(double avn_0, bool random_init, double dn, 
                     unsigned long id_0, int num_init_conds, double T, double lambda, double tol, 
                     int max_iter, unsigned long seed_seq, unsigned long num_seq,
                     double tol_fixed_point, double damping, bool print_avgs,
                     bool print_only_last, bool gr_inside, double eps, double mu,
                     double sigma, unsigned long seed_graph, long N, char * graph_type,
                     double c, char *input_graph_name, bool alpha_inverse){
    cerr << "Initial average abundance: " << avn_0 << endl;
    cerr << "Random initial condition dn=" << dn << "   extrated  " << num_init_conds << " times, with initial seed " << id_0 << endl;
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