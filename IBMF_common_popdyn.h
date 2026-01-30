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
    double field;               ///< Local field (average abundance) at this node
    double av;                  ///< Current average abundance
}Tnode;



void init_avgs(long S, Tnode *nodes, double avn_0, bool random_init, double dn, unsigned long id_0){
    if (random_init){
        gsl_rng * r;
        init_ran(r, id_0);
        double n_i;
        for (long i = 0; i < S; i++){
            n_i = avn_0 - dn + 2 * dn * gsl_rng_uniform(r);
            if (n_i < 0){
                n_i = 0;
            }
            nodes[i].av = n_i;
        }
        gsl_rng_free(r);
    }else{
        for (long i = 0; i < S; i++){
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
double field_in_pop(long S, Tnode *nodes, int c, double mu, double sigma, gsl_rng * r){
    double field = 0, aij;
    long pos;
    for (int j = 0; j < c; j++){
        aij = mu + gsl_ran_gaussian(r, sigma);
        pos = gsl_rng_uniform_int(r, S);
        field += aij * nodes[pos].av;
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
double average(long S, Tnode *nodes){
    double av = 0;
    for (long i = 0; i < S; i++){
        av += nodes[i].av;
    }
    return av / S;
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
double average_sqr(long S, Tnode *nodes){
    double av_sqr = 0;
    for (long i = 0; i < S; i++){
        av_sqr += nodes[i].av * nodes[i].av;
    }
    return av_sqr / S;
}



void print_results_short(int iter, Tnode *nodes, long S, unsigned long seed_graph, 
                         unsigned long seed_seq, unsigned long seed_initcond,
                         int max_iter, bool divergence, size_t elapsed, 
                         double av_counter_dead, double av, double av_sqr){

    long counter_dead = 0;
    for (long i = 0; i < S; i++){
        if (nodes[i].field <= 0){
            counter_dead++;
        }
    }


    double av = average(S, nodes);
    double av_sqr = average_sqr(S, nodes);
    if (divergence){
        cout << iter << "\t" << "diverges" << "\t" << av << "\t" << sqrt(fabs(av_sqr - av * av) / S) << "\t" << 
                counter_dead << "\t" << seed_graph  << "\t"  << seed_seq  << "\t"  << 
                seed_initcond << "\t" << double(elapsed) / 1000 << endl;
    }else{
        bool conv = iter < max_iter;
        cout << iter << "\t" << conv << "\t" << av << "\t" << sqrt(fabs(av_sqr - av * av) / S) << "\t" << 
                counter_dead << "\t" << seed_graph  << "\t"  << seed_seq  << "\t"  << 
                seed_initcond << "\t" << double(elapsed) / 1000 << endl;
    }
}





void print_avgs_to_file(Tnode *nodes, long N, char *fileavgs){
    ofstream fav(fileavgs);
    for (long i = 0; i < N; i++){
        fav << i << "\t" << nodes[i].av << endl;
    }
    fav.close();
    
}



void parse_arguments(int argc, char *argv[], double &avn_0, bool &random_init, double &dn, 
                     unsigned long &id_0, int &num_init_conds, double &T, double &lambda, double &tol, 
                     int &max_iter, double &damping, bool &print_avgs,
                     bool &print_only_last, double &mu,
                     double &sigma, long &S, double &c, char * graph_type, bool &print_params){
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
            cerr << "--damping   [double: 1.0]   :: damping for the convergence process. Setting it to 1 means no damping" << endl;
            cerr << "--print_avgs   ::   if this flag is added to the arguments, the program will print individual average abundances" << endl;
            cerr << "--print_only_last  ::  if this flag is added to the arguments, the program prints only the information obtained by running the convergence process with the last sequence (with seed 'seed_seq+num_seq-1')" << endl;
            cerr << "--mu  [double: 0.2]   ::   average strength of the interactions" << endl;
            cerr << "--sigma  [double: 0.0]  ::  standard deviation of the interactions" << endl;
            cerr << "-S or --size  [long: 1024]  ::  number of species in the population" << endl;
            cerr << "-c or --connect  [int or double, depending on graph type: 3]  ::  average connectivity of the interaction graph " << endl;
            cerr << "--graph_type  [string: RRG]  ::  if graph_type=RRG, the program generates a random regular graph. If graph_type=ER, it generates an Erdos-Renyi graph" << endl;
            cerr << "--print_params  ::  if this flag is added to the arguments, the program will print the parameters used for the run" << endl;
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
        }else if (string(argv[arg_index]) == "--mu"){
            arg_index++;
            mu = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "--sigma"){
            arg_index++;
            sigma = atof(argv[arg_index]);
            arg_index++;
        }else if (string(argv[arg_index]) == "-S" || string(argv[arg_index]) == "--size"){
            arg_index++;
            S = atol(argv[arg_index]);
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
        }else if (string(argv[arg_index]) == "--print_params"){
            print_params = true;
            arg_index++;
        }else{
            cerr << "Unknown argument: " << argv[arg_index] << endl;
            exit(1);
        }
    }
}


void print_params_run(double avn_0, bool random_init, double dn, 
                     unsigned long id_0, int num_init_conds, double T, double lambda, double tol, 
                     int max_iter, double damping, bool print_avgs,
                     bool print_only_last, double eps, double mu,
                     double sigma, long S, double c, char * graph_type){
    cerr << "Initial average abundance: " << avn_0 << endl;
    cerr << "Random initial condition dn=" << dn << "   extrated  " << num_init_conds << " times, with initial seed " << id_0 << endl;
    cerr << "Temperature: " << T << endl;
    cerr << "lambda: " << lambda << endl;
    cerr << "Tolerance for convergence: " << tol << endl;
    cerr << "Maximum number of iterations: " << max_iter << endl;
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
    cerr << "Level of asymmetry in the graph: " << eps << endl;
    cerr << "Average strength of the interactions: " << mu << endl;
    cerr << "Standard deviation of the interactions: " << sigma << endl;
    cerr << "Number of species in the population: " << S << endl;
    cerr << "Average connectivity of the interaction graph: " << c << endl;
    cerr << "Type of graph to generate: " << graph_type << endl;
    
}


#endif