#include "PBMF_ansatz_convergence.h"
#include <iomanip>


using namespace std;


int main(int argc, char *argv[]) {
    double avn_0 = 0.08;
    bool random_init = false;
    double std_n0 = 0;
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
    bool print_responses = false;
    bool print_distributions = false;
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
    double n1 = 1e-4;
    double dn = 5e-3;
    double nmax = 2.0;
    bool adaptive_nmax = false;
    double tail_tol = 1e-2;
    double nmax_growth = 1.5;
    double nmax_limit = 15.0;

    cout << fixed;

    cerr << setprecision(17);

    parse_arguments(argc, argv, avn_0, random_init, std_n0, id_0, num_init_conds, T, lambda, tol, max_iter,
                seed_seq, num_seq, tol_fixed_point, damping, print_avgs, print_responses, 
                print_distributions, print_only_last, gr_inside, eps, mu,
                sigma, seed_graph, N, graph_type, c_arg, gr_str, print_params, alpha_inverse, n1, dn, nmax,
                adaptive_nmax, tail_tol, nmax_growth, nmax_limit);
    if (print_params) {
        print_params_run(avn_0, std_n0, id_0, num_init_conds, T, lambda, tol, max_iter,
                 seed_seq, num_seq, tol_fixed_point, damping,
                 print_avgs, print_responses, print_distributions, print_only_last, gr_inside, eps, mu,
                 sigma, seed_graph, N, graph_type, c_arg, gr_str, alpha_inverse, n1, dn, nmax,
                 adaptive_nmax, tail_tol, nmax_growth, nmax_limit);
    }

    gsl_set_error_handler_off();

    Tnode *nodes;
    Tedge *edges;
    long M;

    create_graph(gr_inside, seed_graph, N, M, nodes, edges, eps, mu, sigma, gr_str, graph_type, c_arg, alpha_inverse);

    char fileout_base[300];

    if (T == 0) {
        cout << "The program is not prepared to run at T=0, since the convergence process is not well defined in that case. Please set a small but nonzero value of T" << endl;
        exit(1);
    } else {
        sprintf(fileout_base, "PBMF_seq_%s_Lotka_Volterra_final_av0_%.3lf_dn_%.3lf_T_%.3lf_lambda_%.1e_tol_%.1e_maxiter_%d_damping_%.2lf_n1_%.1e_dn_%.1e", 
        gr_str, avn_0, std_n0, T, lambda, tol, max_iter, damping, n1, dn);
        several_seq_PBMF(seed_graph, seed_seq, N, M, nodes, edges, T, lambda, tol, max_iter, num_seq, tol_fixed_point,
                 avn_0, damping, print_only_last, print_avgs, print_responses, print_distributions,
                 fileout_base, random_init, std_n0, id_0, num_init_conds, n1, dn, nmax,
                 adaptive_nmax, tail_tol, nmax_growth, nmax_limit);
    }
    
    
    
    return 0;
}