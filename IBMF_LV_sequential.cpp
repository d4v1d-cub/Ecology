#include "IBMF_common.h"
#include "IBMF_convergence_finite_T_seq.h"
#include "IBMF_convergence_T0_seq.h"

/**
 * @file IBMF_LV_sequential.cpp
 * @brief Main program for Individual Based Mean Field analysis of Lotka-Volterra dynamics
 * 
 * This program implements the IBMF approach for analyzing the stationary states
 * of generalized Lotka-Volterra dynamics on sparse interaction networks. The method
 * can handle both zero and finite temperature cases, with optional immigration.
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
    int num_init_conds = 1;
    double T = 0.01;
    double lambda = 1e-6;
    double tol = 1e-6;
    int max_iter = 10000;
    unsigned long seed_seq = 1;
    unsigned long num_seq = 1;
    double tol_fixed_point = 1e-2;
    double damping = 1.0;
    bool print_avgs = false;
    bool print_only_last = false;
    bool gr_inside = false;
    double eps = 1.0;
    double mu = 0.2;
    double sigma = 0.0;
    long N = 1024;
    double c_arg = 3.0;
    unsigned long seed_graph = 1;
    char graph_type[10];
    sprintf(graph_type, "RRG");
    char gr_str[100];
    sprintf(gr_str, "gr_inside_RRG_eps_%.3lf_mu_%.3lf_sigma_%.3lf_N_%li_c_%d_seedgraph_%li", eps, mu, sigma, N, int(round(c_arg)), seed_graph);

    bool print_params = false;
    bool alpha_inverse = false;

    cout << fixed;

    parse_arguments(argc, argv, avn_0, random_init, dn, id_0, num_init_conds, T, lambda, tol, max_iter,
                    seed_seq, num_seq, tol_fixed_point, damping,
                    print_avgs, print_only_last, gr_inside, eps, mu,
                    sigma, seed_graph, N, graph_type, c_arg, gr_str, print_params, alpha_inverse);
    if (print_params) {
        print_params_run(avn_0, random_init, dn, id_0, num_init_conds, T, lambda, tol, max_iter,
                         seed_seq, num_seq, tol_fixed_point, damping,
                         print_avgs, print_only_last, gr_inside, eps, mu,
                         sigma, seed_graph, N, graph_type, c_arg, gr_str, alpha_inverse);
    }

    gsl_set_error_handler_off();

    Tnode *nodes;

    create_graph(gr_inside, seed_graph, N, nodes, eps, mu, sigma, gr_str, graph_type, c_arg, alpha_inverse);

    char fileout_base[300];

    if (T == 0) {
        sprintf(fileout_base, "IBMF_T0_seq_%s_Lotka_Volterra_final_av0_%.3lf_dn_%.3lf_tol_%.1e_maxiter_%d_damping_%.2lf", 
                              gr_str, avn_0, dn, tol, max_iter, damping);
        several_seq_IBMF_T0(seed_graph, seed_seq, N, nodes, tol,
                            max_iter, num_seq, tol_fixed_point,
                            avn_0, damping, print_only_last, print_avgs,
                            fileout_base, random_init, dn, id_0, num_init_conds);
    } else {
        sprintf(fileout_base, "IBMF_seq_%s_Lotka_Volterra_final_av0_%.3lf_dn_%.3lf_T_%.3lf_lambda_%.1e_tol_%.1e_maxiter_%d_damping_%.2lf.txt", 
                          gr_str, avn_0, dn, T, lambda, tol, max_iter, damping);
        several_seq_IBMF(seed_graph, seed_seq, N, nodes, T, lambda, tol,
                         max_iter, num_seq, tol_fixed_point,
                         avn_0, damping, print_only_last, print_avgs,
                         fileout_base, random_init, dn, id_0, num_init_conds);
    }
    
    
    
    return 0;
}