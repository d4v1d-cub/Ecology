/**
 * @file LV_ODE_countFP.cpp
 * @brief Fixed points reached by the Lotka-Volterra dynamics, integrated numerically, compatible with
 *        IBMF_LV_sequential_countFP
 *
 * Integrates   dn_i/dt = n_i (1 - n_i - sum_{j in di} a_ij n_j) + lambda
 * from many initial conditions on one graph and counts the distinct fixed points reached, exactly as
 * IBMF_LV_sequential_countFP does for the IBMF iteration (T = 0). Everything that is not the dynamics is
 * shared with that program, so that the two can be compared run by run:
 *   - the graph is built by create_graph (same seed, same a_ij);
 *   - the initial condition of run k is init_avgs with seed k (same vectors n^(0));
 *   - distinct fixed points are identified with find_fixed_point_index (max_i |m_i - m'_i| < tol_fp);
 *   - the output files (summary, intermediate checkpoints, binary file of the fixed points) have the same
 *     format, with "LV_ODE" instead of "IBMF_T0_seq" at the start of the name.
 * With --method ibmf the program runs the IBMF iteration instead (IBMF_single_try), with the same output,
 * so that both methods also write the per-run log described below.
 *
 * Integration: adaptive Runge-Kutta of orders 5(4) in the variables x_i = ln n_i, Cash-Karp (as in the
 * integrator of Langevin/ecosystem_numerical_integration_toolbox.cpp) or Dormand-Prince (--integrator), with the
 * largest error on the x_i per step divided by tol_rk as error measure. The immigration rate lambda keeps the
 * abundances of extinct species at n_i ~ lambda / |f_i|, with f_i = 1 - sum_j a_ij n_j the field.
 *
 * Convergence: every dt_check time units the largest relative rate max_i |d ln n_i / dt| is computed; the run
 * has converged when it stays below tol_conv during t_persist time units. A species that could invade has
 * d ln n_i/dt ~ f_i > 0, so the test also requires non-invadability. A run that has not converged at t_max does
 * not converge; a run whose abundances exceed 1e6 diverges.
 *
 * Extinct species: at a fixed point with lambda > 0, n_i^2 - f_i n_i - lambda = 0, so that n_i ~ f_i if f_i > 0
 * and n_i ~ lambda/|f_i| < sqrt(lambda) if f_i < 0. A species is extinct if its field is negative, with no
 * threshold on the abundances. The fixed point stored has the extinct species at exactly 0 and the other species
 * at their abundances.
 *
 * Per-run log "<base>_runs.txt": seed_init converged diverged t_end(or iterations) steps max_rate n_surviving
 * fp_index cpu_run wall_run cpu_fp, where max_rate is the final value of max_i |d ln n_i/dt| and fp_index = -1 if
 * the run did not converge (indices start at 0, in the order of the binary file). For --method ibmf the fourth and
 * fifth columns are the number of iterations (sweeps) and 0. cpu_run and wall_run are the CPU time
 * (CLOCK_PROCESS_CPUTIME_ID, user + system) and the wall-clock time of the dynamics of the run, in seconds; cpu_fp
 * is the CPU time spent comparing the result with the fixed points already stored. The header also gives the host
 * and the CPU model. The first eight columns are unchanged with respect to earlier versions.
 *
 * --time_budget <s>: no new run is started if the wall-clock time since the start of the program plus the longest
 * run so far exceeds s seconds. The outputs are then written for the runs done (the file names keep the requested
 * number of initial conditions) and the log ends with a comment line "# stopped by the time budget after k runs".
 */
#include "IBMF_common_countFP.h"
#include "IBMF_convergence_T0_seq_countFP.h"
#include <chrono>
#include <string>
#include <time.h>
#include <unistd.h>

using namespace std;

// CPU time of the process (user + system), in seconds
static double cpu_seconds() {
    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static double wall_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static string cpu_model() {
    ifstream f("/proc/cpuinfo");
    string line;
    while (getline(f, line)) {
        if (line.rfind("model name", 0) == 0) {
            size_t p = line.find(':');
            if (p != string::npos) return line.substr(line.find_first_not_of(" \t", p + 1));
        }
    }
    return "unknown";
}

struct Graph_flat {
    long N;
    vector<long> start;      // neighbours of i: neigh[start[i]] ... neigh[start[i+1]-1]
    vector<long> neigh;
    vector<double> a;        // a[k] = a_{i j} with j = neigh[k] (effect of j on i)
};

Graph_flat flatten_graph(Tnode *nodes, long N) {
    Graph_flat g;
    g.N = N;
    g.start.resize(N + 1);
    g.start[0] = 0;
    for (long i = 0; i < N; i++) {
        g.start[i + 1] = g.start[i] + nodes[i].neighs.size();
        for (size_t k = 0; k < nodes[i].neighs.size(); k++) {
            g.neigh.push_back(nodes[i].neighs[k]);
            g.a.push_back(nodes[i].links_in[k]);    // field_in: 1 - sum_j links_in[j] * av[neighs[j]]
        }
    }
    return g;
}

// field_i = 1 - sum_j a_ij n_j
inline double field(const Graph_flat &g, const double *n, long i) {
    double f = 1.0;
    for (long k = g.start[i]; k < g.start[i + 1]; k++) f -= g.a[k] * n[g.neigh[k]];
    return f;
}

// dx_i/dt = field_i - n_i + lambda / n_i, with n_i = exp(x_i)
void rhs(const Graph_flat &g, const vector<double> &x, vector<double> &n, vector<double> &dx, double lambda) {
    for (long i = 0; i < g.N; i++) n[i] = exp(x[i]);
    for (long i = 0; i < g.N; i++) dx[i] = field(g, n.data(), i) - n[i] + lambda / n[i];
}

// fixed point stored: extinct species (negative field) at 0, the others at their abundances
void project_by_field(const Graph_flat &g, const vector<double> &x, vector<double> &n, vector<double> &m, long &n_surv) {
    for (long i = 0; i < g.N; i++) n[i] = exp(x[i]);
    n_surv = 0;
    for (long i = 0; i < g.N; i++) {
        bool surv = field(g, n.data(), i) > 0.0;
        m[i] = surv ? n[i] : 0.0;
        n_surv += surv;
    }
}

// largest relative rate max_i |d ln n_i/dt|
inline double max_rate(const vector<double> &dx) {
    double r = 0.0;
    for (double v : dx) r = max(r, fabs(v));
    return r;
}

struct ODE_params {
    double lambda = 1e-6, tol_rk = 1e-8, t_max = 200000.0, tol_conv = 1e-6, t_persist = 20.0, dt_check = 1.0;
    double h_first = 1e-2, h_min = 1e-10, n_div = 1e6;
    string integrator = "ck";
};

/**
 * Integrates from the abundances in nodes[i].av; on return nodes[i].av holds the projected state m.
 * Returns true if converged.
 */
bool integrate_LV(const Graph_flat &g, Tnode *nodes, const ODE_params &p, bool &diverged, double &t_end,
                  long &steps, double &rate_end, long &n_surv) {
    // Dormand-Prince 5(4)
    static const double a21 = 1. / 5;
    static const double a31 = 3. / 40, a32 = 9. / 40;
    static const double a41 = 44. / 45, a42 = -56. / 15, a43 = 32. / 9;
    static const double a51 = 19372. / 6561, a52 = -25360. / 2187, a53 = 64448. / 6561, a54 = -212. / 729;
    static const double a61 = 9017. / 3168, a62 = -355. / 33, a63 = 46732. / 5247, a64 = 49. / 176,
                        a65 = -5103. / 18656;
    static const double b1 = 35. / 384, b3 = 500. / 1113, b4 = 125. / 192, b5 = -2187. / 6784, b6 = 11. / 84;
    static const double e1 = 71. / 57600, e3 = -71. / 16695, e4 = 71. / 1920, e5 = -17253. / 339200,
                        e6 = 22. / 525, e7 = -1. / 40;
    // Cash-Karp 5(4) (same coefficients as rkck_algorithm in the Langevin toolbox)
    static const double c10 = 0.2, c20 = 3.0 / 40.0, c21 = 9.0 / 40.0, c30 = 0.3, c31 = -0.9, c32 = 1.2,
                        c40 = -11.0 / 54.0, c41 = 2.5, c42 = -70.0 / 27.0, c43 = 35.0 / 27.0,
                        c50 = 1631.0 / 55296.0, c51 = 175.0 / 512.0, c52 = 575.0 / 13824.0, c53 = 44275.0 / 110592.0,
                        c54 = 253.0 / 4096.0;
    static const double d0 = 37.0 / 378.0, d2 = 250.0 / 621.0, d3 = 125.0 / 594.0, d5 = 512.0 / 1771.0;
    static const double f0 = 37.0 / 378.0 - 2825.0 / 27648.0, f2 = 250.0 / 621.0 - 18575.0 / 48384.0,
                        f3 = 125.0 / 594.0 - 13525.0 / 55296.0, f4 = -277.0 / 14336.0, f5 = 512.0 / 1771.0 - 0.25;
    const bool dp = p.integrator == "dp";
    long N = g.N;
    vector<double> x(N), xt(N), xn(N), n(N), m(N), k1(N), k2(N), k3(N), k4(N), k5(N), k6(N), k7(N);
    double n_floor = p.lambda > 0 ? p.lambda : 1e-12;
    for (long i = 0; i < N; i++) x[i] = log(max(nodes[i].av, n_floor));
    double t = 0.0, h = p.h_first, t_next_check = p.dt_check, t_ok = -1.0;
    double x_div = log(p.n_div);
    bool converged = false;
    diverged = false;
    steps = 0;
    rhs(g, x, n, k1, p.lambda);
    rate_end = max_rate(k1);
    while (t < p.t_max) {
        if (t + h > p.t_max) h = p.t_max - t;
        double err = 0.0;
        if (dp) {
            for (long i = 0; i < N; i++) xt[i] = x[i] + h * a21 * k1[i];
            rhs(g, xt, n, k2, p.lambda);
            for (long i = 0; i < N; i++) xt[i] = x[i] + h * (a31 * k1[i] + a32 * k2[i]);
            rhs(g, xt, n, k3, p.lambda);
            for (long i = 0; i < N; i++) xt[i] = x[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
            rhs(g, xt, n, k4, p.lambda);
            for (long i = 0; i < N; i++) xt[i] = x[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
            rhs(g, xt, n, k5, p.lambda);
            for (long i = 0; i < N; i++)
                xt[i] = x[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
            rhs(g, xt, n, k6, p.lambda);
            for (long i = 0; i < N; i++) xn[i] = x[i] + h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i]);
            rhs(g, xn, n, k7, p.lambda);
            for (long i = 0; i < N; i++)
                err = max(err, fabs(h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i])));
        } else {
            for (long i = 0; i < N; i++) xt[i] = x[i] + h * c10 * k1[i];
            rhs(g, xt, n, k2, p.lambda);
            for (long i = 0; i < N; i++) xt[i] = x[i] + h * (c20 * k1[i] + c21 * k2[i]);
            rhs(g, xt, n, k3, p.lambda);
            for (long i = 0; i < N; i++) xt[i] = x[i] + h * (c30 * k1[i] + c31 * k2[i] + c32 * k3[i]);
            rhs(g, xt, n, k4, p.lambda);
            for (long i = 0; i < N; i++) xt[i] = x[i] + h * (c40 * k1[i] + c41 * k2[i] + c42 * k3[i] + c43 * k4[i]);
            rhs(g, xt, n, k5, p.lambda);
            for (long i = 0; i < N; i++)
                xt[i] = x[i] + h * (c50 * k1[i] + c51 * k2[i] + c52 * k3[i] + c53 * k4[i] + c54 * k5[i]);
            rhs(g, xt, n, k6, p.lambda);
            for (long i = 0; i < N; i++) {
                xn[i] = x[i] + h * (d0 * k1[i] + d2 * k3[i] + d3 * k4[i] + d5 * k6[i]);
                err = max(err, fabs(h * (f0 * k1[i] + f2 * k3[i] + f3 * k4[i] + f4 * k5[i] + f5 * k6[i])));
            }
        }
        err /= p.tol_rk;
        if (err <= 1.0 || h <= p.h_min) {             // accept the step
            t += h;
            x.swap(xn);
            if (dp) k1.swap(k7);                       // FSAL: derivative at the new point
            else rhs(g, x, n, k1, p.lambda);           // Cash-Karp: new derivative evaluation
            steps++;
            for (long i = 0; i < N; i++) {
                if (x[i] > x_div || !std::isfinite(x[i])) { diverged = true; break; }
            }
            if (diverged) break;
            if (t >= t_next_check) {
                t_next_check = t + p.dt_check;
                rate_end = max_rate(k1);
                if (rate_end < p.tol_conv) {
                    if (t_ok < 0) t_ok = t;
                    if (t - t_ok >= p.t_persist) { converged = true; break; }
                } else {
                    t_ok = -1.0;
                }
            }
        }
        double fac = err > 0 ? 0.9 * pow(err, -0.2) : 5.0;
        h *= min(5.0, max(0.2, fac));
        if (h < p.h_min) h = p.h_min;
    }
    t_end = t;
    rate_end = max_rate(k1);
    project_by_field(g, x, n, m, n_surv);
    for (long i = 0; i < N; i++) nodes[i].av = m[i];
    return converged && !diverged;
}


void print_usage(const char *prog) {
    cerr << "Usage: " << prog << " [options]\n"
         << "Same options as IBMF_LV_sequential_countFP for the graph, the initial conditions and the output:\n"
         << "  --gr_inside --graph_type RRG -N <N> -c <c> --eps <eps> --mu <mu> --sigma <sigma> --seed_graph <s>\n"
         << "  --avn_0 <n0> --random_init <dn> <id_0> <num_init_conds> --tol_fp <tol> --print_fp_bin --no_checkpoints\n"
         << "Dynamics:\n"
         << "  --method <ode|ibmf>   [ode]    numerical integration of the LV equations, or IBMF iteration\n"
         << "  --integrator <ck|dp>  [ck]     Cash-Karp or Dormand-Prince adaptive Runge-Kutta 5(4)\n"
         << "  --lambda <l>          [1e-6]   immigration rate\n"
         << "  --tol_rk <t>          [1e-8]   tolerance of the adaptive Runge-Kutta step (on ln n)\n"
         << "  --t_max <t>           [2e5]    maximum integration time\n"
         << "  --tol_conv <t>        [1e-6]   tolerance on max_i |d ln n_i/dt|\n"
         << "  --t_persist <t>       [20]     time during which max_i |d ln n_i/dt| must stay below tol_conv\n"
         << "  --dt_check <t>        [1]      interval between convergence checks\n"
         << "  --time_budget <s>     [0]      wall-clock budget in seconds (0: none); see the header of the source\n"
         << "IBMF (--method ibmf): --tol <1e-6> --max_iter <10000> --seed_seq <1> --damping <1.0>\n";
}


int main(int argc, char *argv[]) {
    double avn_0 = 0.5, dn = 0.5, tol_fixed_point = 1e-2, eps = 0.0, mu = 0.2, sigma = 0.0, c_arg = 3.0;
    double tol_ibmf = 1e-6, damping = 1.0;
    int max_iter = 10000;
    bool random_init = false, gr_inside = false, print_fp_bin = false, no_checkpoints = false;
    unsigned long id_0 = 1, seed_graph = 1, seed_seq = 1;
    int num_init_conds = 1;
    long N = 1024;
    char graph_type[10];
    sprintf(graph_type, "RRG");
    string method = "ode";
    ODE_params p;
    double time_budget = 0.0;
    const double wall_start = wall_seconds();

    for (int k = 1; k < argc; k++) {
        string a = argv[k];
        auto next = [&]() { if (k + 1 >= argc) { cerr << "Missing value for " << a << endl; exit(1); } return argv[++k]; };
        if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else if (a == "--avn_0") avn_0 = atof(next());
        else if (a == "--random_init") { random_init = true; dn = atof(next()); id_0 = atol(next()); num_init_conds = atoi(next()); }
        else if (a == "--tol_fp") tol_fixed_point = atof(next());
        else if (a == "--gr_inside") gr_inside = true;
        else if (a == "--eps") eps = atof(next());
        else if (a == "--mu") mu = atof(next());
        else if (a == "--sigma") sigma = atof(next());
        else if (a == "--seed_graph") seed_graph = atol(next());
        else if (a == "-N" || a == "--size") N = atol(next());
        else if (a == "-c" || a == "--connect") c_arg = atof(next());
        else if (a == "--graph_type") sprintf(graph_type, "%s", next());
        else if (a == "--print_fp_bin") print_fp_bin = true;
        else if (a == "--no_checkpoints") no_checkpoints = true;
        else if (a == "--method") method = next();
        else if (a == "--lambda") p.lambda = atof(next());
        else if (a == "--tol_rk") p.tol_rk = atof(next());
        else if (a == "--t_max") p.t_max = atof(next());
        else if (a == "--tol_conv") p.tol_conv = atof(next());
        else if (a == "--t_persist") p.t_persist = atof(next());
        else if (a == "--dt_check") p.dt_check = atof(next());
        else if (a == "--integrator") p.integrator = next();
        else if (a == "--time_budget") time_budget = atof(next());
        else if (a == "--tol") tol_ibmf = atof(next());
        else if (a == "--max_iter") max_iter = atoi(next());
        else if (a == "--seed_seq") seed_seq = atol(next());
        else if (a == "--damping") damping = atof(next());
        else { cerr << "Unknown argument: " << a << endl; print_usage(argv[0]); return 1; }
    }
    if (!gr_inside) { cerr << "Only --gr_inside is supported" << endl; return 1; }
    if (method != "ode" && method != "ibmf") { cerr << "--method must be ode or ibmf" << endl; return 1; }
    if (p.integrator != "ck" && p.integrator != "dp") { cerr << "--integrator must be ck or dp" << endl; return 1; }

    gsl_set_error_handler_off();
    Tnode *nodes;
    char gr_str[100];
    create_graph(gr_inside, seed_graph, N, nodes, eps, mu, sigma, gr_str, graph_type, c_arg, false);
    Graph_flat g = flatten_graph(nodes, N);

    char fileout_base[400];
    if (method == "ode") {
        snprintf(fileout_base, sizeof(fileout_base),
                 "LV_ODE_%s_Lotka_Volterra_final_av0_%.3lf_dn_%.3lf_ninitcond_%d_lambda_%.1e_%s_tolrk_%.1e_tolconv_%.1e_tmax_%g",
                 gr_str, avn_0, dn, num_init_conds, p.lambda, p.integrator.c_str(), p.tol_rk, p.tol_conv, p.t_max);
    } else {
        snprintf(fileout_base, sizeof(fileout_base),
                 "LV_IBMF_%s_Lotka_Volterra_final_av0_%.3lf_dn_%.3lf_ninitcond_%d_tol_%.1e_maxiter_%d_damping_%.2lf",
                 gr_str, avn_0, dn, num_init_conds, tol_ibmf, max_iter, damping);
    }
    char filelog[460];
    snprintf(filelog, sizeof(filelog), "%s_runs.txt", fileout_base);
    FILE *flog = fopen(filelog, "w");
    fprintf(flog, "# method %s  N %ld  mu %g  sigma %g  eps %g  seed_graph %lu\n", method.c_str(), N, mu, sigma, eps, seed_graph);
    char host[256] = "unknown";
    gethostname(host, sizeof(host) - 1);
    fprintf(flog, "# host %s  cpu %s  time_budget %g\n", host, cpu_model().c_str(), time_budget);
    if (method == "ode")
        fprintf(flog, "# seed_init converged diverged t_end steps max_rate n_surviving fp_index cpu_run wall_run cpu_fp\n");
    else
        fprintf(flog, "# seed_init converged diverged iterations 0 0 n_surviving fp_index cpu_run wall_run cpu_fp\n");
    fflush(flog);

    gsl_rng *r_condinit, *r_seq;
    init_ran(r_condinit, id_0);
    init_ran(r_seq, seed_seq);
    long *sequence = new long[N];
    vector<int> fixed_point_counts;
    init_fixed_points_storage(nodes, N, fixed_point_counts);
    int attempts = 0;
    double longest_run = 0.0;          // longest wall-clock time of a run so far (time budget)
    bool stopped = false;

    for (unsigned long seed_init = id_0; seed_init < id_0 + num_init_conds; seed_init++) {
        if (time_budget > 0 && attempts > 0 && wall_seconds() - wall_start + longest_run > time_budget) {
            stopped = true;
            break;
        }
        bool converged, diverged = false;
        double t_end = 0, R_end = 0;   // R_end: final max_i |d ln n_i/dt| (ode)
        long steps = 0, n_surv = 0;
        const double w0 = wall_seconds(), c0 = cpu_seconds();
        if (method == "ode") {
            init_avgs(N, nodes, avn_0, random_init, dn, seed_init, r_condinit);
            converged = integrate_LV(g, nodes, p, diverged, t_end, steps, R_end, n_surv);
        } else {
            int iter;
            IBMF_single_try(seed_seq, N, nodes, tol_ibmf, max_iter, avn_0, damping, random_init, dn, seed_init,
                            sequence, diverged, iter, r_seq, r_condinit);
            converged = !diverged && iter < max_iter;
            t_end = iter;
            for (long i = 0; i < N; i++) if (nodes[i].av > 0) n_surv++;
        }
        const double c1 = cpu_seconds(), w1 = wall_seconds();
        attempts++;
        int fp_index = -1;
        if (converged) {
            fp_index = find_fixed_point_index(nodes, N, tol_fixed_point);
            if (fp_index >= 0) {
                update_fixed_point_counts(fixed_point_counts, fp_index);
            } else {
                add_new_fixed_point(nodes, N, 1, fixed_point_counts);
                fp_index = (int) fixed_point_counts.size() - 1;
            }
        }
        const double c2 = cpu_seconds();
        longest_run = max(longest_run, wall_seconds() - w0);
        fprintf(flog, "%lu %d %d %.6g %ld %.3e %ld %d %.4e %.4e %.4e\n", seed_init, (int) converged, (int) diverged,
                t_end, steps, R_end, n_surv, fp_index, c1 - c0, w1 - w0, c2 - c1);
        fflush(flog);

        long ninitcond_done = seed_init - id_0 + 1;
        if (!no_checkpoints && is_power_of_two(ninitcond_done) && ninitcond_done != num_init_conds && ninitcond_done >= 256) {
            char fileout_checkpoint[460];
            build_checkpoint_fileout_base(fileout_base, num_init_conds, ninitcond_done, fileout_checkpoint,
                                          sizeof(fileout_checkpoint));
            print_fixed_points_summary(nodes, N, fileout_checkpoint, seed_graph, attempts, false, fixed_point_counts);
        }
    }
    print_fixed_points_summary(nodes, N, fileout_base, seed_graph, attempts, false, fixed_point_counts);
    if (print_fp_bin) print_fixed_points_binary(nodes, N, fileout_base, attempts, fixed_point_counts);
    if (stopped) fprintf(flog, "# stopped by the time budget after %d runs\n", attempts);
    fclose(flog);
    delete[] sequence;
    gsl_rng_free(r_condinit);
    gsl_rng_free(r_seq);
    return 0;
}
