#include "IBMF_common_popdyn.h"
#include "IBMF_convergence_finite_T_popdyn.h"
#include "IBMF_convergence_T0_popdyn.h"

/**
 * @file IBMF_LV_popdyn.cpp
 * @brief Population dynamics implementation of Individual Based Mean Field for the generalized Lotka-Volterra model
 * 
 * This program implements the IBMF approach for analyzing the stationary states
 * of generalized Lotka-Volterra dynamics on sparse interaction networks using population dynamics. 
 * The method can handle both zero and finite temperature cases, with optional immigration.
 * 
 * Key features:
 * - Supports both random regular graphs (RRG) and Erdős-Rényi (ER) networks
 * - Handles symmetric and asymmetric interactions
 * - Implements both T=0 and T>0 solutions
 * - Includes damping for improved convergence
 * - Multiple initial conditions and update sequences for robustness
 */

using namespace std;


int main(int argc, char *argv[]) {
    double avn_0 = 0.08;
    bool random_init = false;
    double dn = 0;
    unsigned long id_0 = 1;
    double T = 0.01;
    double lambda = 1e-6;
    double tol = 1e-6;
    int max_iter = 10000;
    double damping = 1.0;
    bool print_avgs = false;
    int print_every = 0;
    double mu = 0.2;
    double sigma = 0.0;
    long S = 1024;
    double c_arg = 3.0;
    unsigned long seed_graph = 1;
    char graph_type[10];
    sprintf(graph_type, "RRG");
    char gr_str[100];
    sprintf(gr_str, "gr_inside_RRG_mu_%.3lf_sigma_%.3lf_S_%li_c_%d_seedgraph_%li", mu, sigma, S, 
                    int(round(c_arg)), seed_graph);
    int N_measurements = 1;
    unsigned long seed_choose = 1;
    bool print_params = false;
    bool alpha_inverse = false;

    cout << fixed;

    parse_arguments(argc, argv, avn_0, random_init, dn, id_0, T, lambda, tol, max_iter, 
                    damping, print_avgs, print_every, mu, sigma, S, c_arg, graph_type, 
                    N_measurements, seed_choose, print_params);
    if (print_params) {
        print_params_run(avn_0, random_init, dn, id_0, T, lambda, tol, max_iter, damping, 
                         print_avgs, print_every, mu, sigma, S, c_arg, 
                         graph_type, N_measurements, seed_choose);
    }

    gsl_set_error_handler_off();

    Tnode *nodes;
    nodes = new Tnode[S];

    char fileout_base[300];

    if (T == 0) {
        sprintf(fileout_base, "IBMF_T0_popdyn_%s_Lotka_Volterra_final_av0_%.3lf_dn_%.3lf_tol_%.1e_maxiter_%d_damping_%.2lf", 
                              gr_str, avn_0, dn, tol, max_iter, damping);
        run_IBMF_T0(seed_choose, S, nodes, tol, max_iter, avn_0, damping, 
                    print_every, print_avgs, fileout_base, random_init, dn, id_0, 
                    c_arg, mu, sigma, graph_type, N_measurements);
    } else {
        sprintf(fileout_base, "IBMF_popdyn_%s_Lotka_Volterra_final_av0_%.3lf_dn_%.3lf_T_%.3lf_lambda_%.1e_tol_%.1e_maxiter_%d_damping_%.2lf.txt", 
                          gr_str, avn_0, dn, T, lambda, tol, max_iter, damping);
        run_IBMF_finite_T(seed_choose, S, nodes, T, lambda, tol, max_iter, avn_0, damping, 
                          print_every, print_avgs, fileout_base, random_init, dn, id_0, 
                          c_arg, mu, sigma, graph_type, N_measurements);
    }
    
    
    
    return 0;
}