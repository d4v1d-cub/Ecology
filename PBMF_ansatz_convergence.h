#ifndef __PBMF_ANSATZ_CONVERGENCE_H_INCLUDED__
#define __PBMF_ANSATZ_CONVERGENCE_H_INCLUDED__

#include "PBMF_ansatz_common.h"
#include <chrono>
#include <limits>
#include <algorithm>

/**
 * @brief Compute coefficients for the finite temperature IBMF solution
 * @param beta Inverse temperature (1/T)
 * @param lambda Immigration rate
 * @param coefficients Output matrix of coefficients for the hypergeometric functions
 * @param gamma_vals Output array of gamma function values
 * @param maximum Maximum allowed value before switching to asymptotic form
 * @return True if gamma functions diverge and asymptotic form is used
 * 
 * The stationary solution at finite T involves ratios of confluent hypergeometric
 * functions with coefficients determined by beta and lambda. This function computes
 * these coefficients, handling both the regular case and the asymptotic approximation
 * when the gamma functions become too large to evaluate directly.
 */
bool comp_coefficients(double beta, double lambda, double **&coefficients, double *&gamma_vals, 
                       double maximum=1e10){
    bool gamma_diverges = false;
    gamma_vals = new double[2];
    // Check if gamma functions can be evaluated directly
    if (std::isnan(gsl_sf_gamma((1 + beta * lambda) / 2)) || std::isinf(gsl_sf_gamma((1 + beta * lambda) / 2)) || 
        gsl_sf_gamma((1 + beta * lambda) / 2) > maximum){
        gamma_diverges = true;
        gamma_vals[0] = sqrt(2 * M_PI / beta / lambda) * pow(beta * lambda / 2 / M_E, beta * lambda / 2);
        gamma_vals[1] = sqrt(4 * M_PI / (1 + beta * lambda)) * pow((1 + beta * lambda) / 2 / M_E, (1 + beta * lambda) / 2);
    }else{
        gamma_vals[0] = gsl_sf_gamma(beta * lambda / 2);
        gamma_vals[1] = gsl_sf_gamma((1 + beta * lambda) / 2);
    }

    coefficients = new double *[2];
    for (int i = 0; i < 2; i++){
        coefficients[i] = new double[2];
    }

    if (gamma_diverges){
        coefficients[0][0] = 1;
        coefficients[0][1] = beta * sqrt(lambda) * (1 - 1.0 / 4 / beta / lambda);

        coefficients[1][0] = sqrt(lambda) * (1 - 1.0 / 4 / beta / lambda);
        coefficients[1][1] = lambda * beta;
    }else{
        double gammabl2 = gsl_sf_gamma(beta * lambda / 2);
        double gammabl12 = gsl_sf_gamma((1 + beta * lambda) / 2);
        
        coefficients[0][0] = sqrt(beta / 2) * gammabl2;
        coefficients[0][1] = beta * gammabl12;

        coefficients[1][0] = gammabl12;
        coefficients[1][1] = sqrt(beta / 2) * beta * lambda * gammabl2;
    }

    return gamma_diverges;
}


double find_divergence_max(double beta, double alpha, double hmax=100, double precision=1e-4, double maximum=1e10){
    double val1, val2;
    val1 = gsl_sf_hyperg_1F1(alpha, 0.5, beta * hmax * hmax / 2);
    val2 = gsl_sf_hyperg_1F1(alpha + 0.5, 1.5, beta * hmax * hmax / 2);
    while (!(std::isnan(val1) || std::isinf(val1) || std::isnan(val2) || std::isinf(val2) || 
             val1 > maximum || val2 > maximum)){
        hmax *= 2;
        val1 = gsl_sf_hyperg_1F1(alpha, 0.5, beta * hmax * hmax / 2);
        val2 = gsl_sf_hyperg_1F1(alpha + 0.5, 1.5, beta * hmax * hmax / 2);   
    }

    double hmin = 0;
    double h = (hmax + hmin) / 2;
    while (hmax - hmin > precision){
        val1 = gsl_sf_hyperg_1F1(alpha, 0.5, beta * h * h / 2);
        val2 = gsl_sf_hyperg_1F1(alpha + 0.5, 1.5, beta * h * h / 2);
        if (std::isnan(val1) || std::isinf(val1) || std::isnan(val2) || std::isinf(val2) || 
            val1 > maximum || val2 > maximum){
            hmax = h;
        }else{
            hmin = h;
        }
        h = (hmax + hmin) / 2;
    }

    cerr << "Divergence found at h = " << hmax << endl;
    cerr << "Last value to converge: " << hmin << endl;
    return hmin;
}


double numerator_av(double beta, double lambda, double hi, double *coefficients){
    return coefficients[0] * gsl_sf_hyperg_1F1((1 + beta * lambda) / 2, 0.5, beta * hi * hi / 2) +
           coefficients[1] * hi * gsl_sf_hyperg_1F1(1 + beta * lambda / 2, 1.5, beta * hi * hi / 2);
}


double denominator(double beta, double lambda, double hi, double *coefficients, double normfactor = 1e-14){
    return coefficients[0] * gsl_sf_hyperg_1F1(beta * lambda / 2, 0.5, beta * hi * hi / 2) + 
           coefficients[1] * hi * gsl_sf_hyperg_1F1((1 + beta * lambda) / 2, 1.5, beta * hi * hi / 2)
           + normfactor;
}


double find_divergence_min(double beta, double lambda, double **coefficients, double hmin=-100, double precision=1e-4, double maximum=1e10){
    double num, den;
    num = numerator_av(beta, lambda, hmin, coefficients[1]);
    den = denominator(beta, lambda, hmin, coefficients[0]);
    
    while (!(std::isnan(num) || std::isinf(num) || std::isnan(den) || std::isinf(den) || 
             num > maximum || den > maximum || num < 0 || den < 0)){
        hmin *= 2;
        num = numerator_av(beta, lambda, hmin, coefficients[1]);
        den = denominator(beta, lambda, hmin, coefficients[0]);   
    }

    double hmax = 0;
    double h = (hmax + hmin) / 2;
    while (hmax - hmin > precision){
        num = numerator_av(beta, lambda, h, coefficients[1]);
        den = denominator(beta, lambda, h, coefficients[0]); 
        if (std::isnan(num) || std::isinf(num) || std::isnan(den) || std::isinf(den) || 
             num > maximum || den > maximum || num < 0 || den < 0){
            hmin = h;
        }else{
            hmax = h;
        }
        h = (hmax + hmin) / 2;
    }

    cerr << "Divergence found at h = " << hmin << endl;
    cerr << "Last value to converge: " << hmax << endl;
    return hmin;
}






double field_cav_in_from_single_avgs(long e, int k, Tnode *nodes, Tedge *edges, double n_neigh){
    double sum = 0;
    long edge_neigh;
    int pos_there;
    for (long j = 0; j < edges[e].edges_except[k].size(); j++){
        edge_neigh = edges[e].edges_except[k][j];
        pos_there = edges[e].pos_there[k][j];
        sum += edges[edge_neigh].links[pos_there] * nodes[edges[edge_neigh].nodes_in[1 - pos_there]].av_abundance;
    }
    sum += edges[e].links[k] * n_neigh;
    return 1 - sum;
}


void cond_av_from_single_avgs(double beta, double lambda, Tnode *nodes, Tedge *edges, long edge_index, int k,
                              const vector <double> &n_grid, double hmin, double hmax, double **coefficients, double *gamma_vals,
                              double normfactor = 1e-14){
    double field_cav, cond_av, den;
    for (long n_index = 0; n_index < n_grid.size(); n_index++){
        field_cav = field_cav_in_from_single_avgs(edge_index, k, nodes, edges, n_grid[n_index]);
        if (field_cav > hmax){
            edges[edge_index].cond_av[k][n_index] = (1 - 1.0 / beta / field_cav + lambda / field_cav);
        }else if (field_cav < hmin){
            edges[edge_index].cond_av[k][n_index] = lambda / fabs(field_cav);
        }else if(field_cav == 0){
            edges[edge_index].cond_av[k][n_index] = sqrt(2.0 / beta) * gamma_vals[1] / gamma_vals[0];
        }else{
            den = denominator(beta, lambda, field_cav, coefficients[0], normfactor);
            edges[edge_index].cond_av[k][n_index] = numerator_av(beta, lambda, field_cav, coefficients[1]) / den;           
        }
    }
}

void update_integrated_cond_av(const vector<double> &cond_av, vector <double> &integrated_cond_av,
                               const vector<double> &n_grid){
    for (int index_n = 1; index_n < n_grid.size(); index_n++){
        integrated_cond_av[index_n] = integrated_cond_av[index_n - 1] + (cond_av[index_n] + cond_av[index_n - 1]) * (n_grid[index_n] - n_grid[index_n - 1]) / 2.0;
    }
}


void init_cond_av_single_avgs(long M, double beta, double lambda, Tnode *nodes, Tedge *edges,
                   const vector <double> &n_grid, double hmin, double hmax, double **coefficients, double *gamma_vals){
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            cond_av_from_single_avgs(beta, lambda, nodes, edges, e, k, n_grid, hmin, hmax,
                                     coefficients, gamma_vals);
            update_integrated_cond_av(edges[e].cond_av[k], edges[e].integrated_cond_av[k], n_grid);
        }
    }
}



double sum_over_neighs_except(long e, int k, Tedge *edges, double n_neigh, int n_index_node){
    double sum = 0;
    long edge_neigh;
    int pos_there;
    for (long j = 0; j < edges[e].edges_except[k].size(); j++){
        edge_neigh = edges[e].edges_except[k][j];
        pos_there = edges[e].pos_there[k][j];
        sum += edges[edge_neigh].links[pos_there] * edges[edge_neigh].integrated_cond_av[1 - pos_there][n_index_node];
    }
    return sum;
}

///// added by TT

double log_add_exp(double log_a, double log_b) {
    const double minus_infinity =
        -std::numeric_limits<double>::infinity();

    const double plus_infinity =
        std::numeric_limits<double>::infinity();

    if (std::isnan(log_a) || std::isnan(log_b)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (log_a == minus_infinity) {
        return log_b;
    }

    if (log_b == minus_infinity) {
        return log_a;
    }

    if (log_a == plus_infinity || log_b == plus_infinity) {
        return plus_infinity;
    }

    const double maximum = std::max(log_a, log_b);

    return maximum
         + std::log1p(
               std::exp(-std::fabs(log_a - log_b))
           );
}


bool invalid_log_value(double log_value) {
    return std::isnan(log_value)
        || log_value == std::numeric_limits<double>::infinity();
}


bool exp_from_log(double log_value, double &value) {
    static const double log_double_max =
        std::log(std::numeric_limits<double>::max());

    static const double log_double_min =
        std::log(std::numeric_limits<double>::denorm_min());

    if (invalid_log_value(log_value)) {
        return false;
    }

    if (
        log_value == -std::numeric_limits<double>::infinity()
        || log_value < log_double_min
    ) {
        value = 0.0;
        return true;
    }

    if (log_value > log_double_max) {
        return false;
    }

    value = std::exp(log_value);

    return std::isfinite(value);
}


bool nonnegative_difference( double a, double b, double &difference) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return false;
    }

    difference = a - b;

    const double scale =
        std::max( 1.0, std::max(std::fabs(a), std::fabs(b)) );

    const double roundoff_tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;

    if ( difference < 0.0 && difference >= -roundoff_tolerance ) {
        difference = 0.0;
    }

    return std::isfinite(difference)
        && difference >= 0.0;
}

/////

double integrate_direct(const vector<double> &fixed_integrand, double beta, double lambda, int p, const vector<double> &n_grid, 
                 Tedge *edges, double n_neigh, long edge_index, int k, const vector<double> &simpson_weights){
    double sum_cav = sum_over_neighs_except(edge_index, k, edges, n_neigh, 1) + edges[edge_index].links[k] * n_neigh * n_grid[1];
    double integral = fixed_integrand[0] * n_grid[1] / (beta * lambda + p) * exp(-beta * sum_cav);
    for (int n_index = 1; n_index < n_grid.size(); n_index++){
        sum_cav = sum_over_neighs_except(edge_index, k, edges, n_neigh, n_index) + edges[edge_index].links[k] * n_neigh * n_grid[n_index];
        integral += simpson_weights[n_index - 1] * fixed_integrand[n_index - 1] * exp(-beta * sum_cav);
    }
    return integral;
}


double integrate_log( double beta, double lambda, int p, const vector<double> &n_grid, Tedge *edges, double n_neigh, long edge_index, int k, const vector<double> &simpson_weights ) {
    const double minus_infinity =
        -std::numeric_limits<double>::infinity();

    double log_integral = minus_infinity;

    double n = n_grid[1];

    double sum_cav = sum_over_neighs_except( edge_index, k, edges, n_neigh, 1 ) + edges[edge_index].links[k] * n_neigh * n;

    double denominator_zero_part = beta * lambda + p;

    if (denominator_zero_part <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double log_first_term = (beta * lambda + p) * std::log(n) - std::log(denominator_zero_part) - beta * ( 0.5 * (n - 1.0) * (n - 1.0) + sum_cav );

    log_integral = log_add_exp(log_integral, log_first_term);

    for (int n_index = 1; n_index < static_cast<int>(n_grid.size()); n_index++) {

        n = n_grid[n_index];

        sum_cav = sum_over_neighs_except( edge_index, k, edges, n_neigh, n_index ) + edges[edge_index].links[k] * n_neigh * n;

        double weight = simpson_weights[n_index - 1];

        if (n <= 0.0 || weight <= 0.0) {
            continue;
        }

        double log_term = std::log(weight) + (beta * lambda - 1.0 + p) * std::log(n) - beta * ( 0.5 * (n - 1.0) * (n - 1.0) + sum_cav );

        if (!std::isfinite(log_term)) {
            cerr << "Bad logarithmic integrand"
                 << " edge=" << edge_index
                 << " k=" << k
                 << " p=" << p
                 << " n_neigh=" << n_neigh
                 << " n_index=" << n_index
                 << " n=" << n
                 << " sum_cav=" << sum_cav
                 << " log_term=" << log_term
                 << endl;

            return std::numeric_limits<double>::quiet_NaN();
        }

        log_integral = log_add_exp(log_integral, log_term);
    }

    return log_integral;
}


bool compute_ratio_logarithmic( double beta, double lambda, const vector<double> &n_grid, Tedge *edges, double n_neigh, long edge_index, int k, const vector<double> &simpson_weights, double &ratio ) {
    const double log_num = integrate_log( beta, lambda, 1, n_grid, edges, n_neigh, edge_index, k, simpson_weights );

    const double log_den = integrate_log( beta, lambda, 0, n_grid, edges, n_neigh, edge_index, k, simpson_weights );

    if (
        !std::isfinite(log_num)
        || !std::isfinite(log_den)
    ) {
        cerr << "Bad logarithmic integral"
             << " edge=" << edge_index
             << " k=" << k
             << " n_neigh=" << n_neigh
             << " log_num=" << log_num
             << " log_den=" << log_den
             << endl;

        return false;
    }

    const double log_ratio = log_num - log_den;

    static const double log_double_max = std::log(std::numeric_limits<double>::max());

    static const double log_double_min = std::log(std::numeric_limits<double>::denorm_min());

    if (log_ratio > log_double_max) {
        cerr << "Logarithmic ratio overflows"
             << " edge=" << edge_index
             << " k=" << k
             << " n_neigh=" << n_neigh
             << " log_ratio=" << log_ratio
             << endl;

        return false;
    }

    if (log_ratio < log_double_min) {
        ratio = 0.0;
    } else {
        ratio = std::exp(log_ratio);
    }

    return std::isfinite(ratio);
}


double update_cond_avgs( Tedge *edges, const vector<double> &n_grid, const vector<double> &simpson_weights, const vector<double> &fixed_integrand_num, const vector<double> &fixed_integrand_den, double beta, double lambda, long M, double dn, double damping, double tol ) {
    double max_variation = 0.0;

    for (long e = 0; e < M; ++e) {
        for (int k = 0; k < 2; ++k) {
            double variation = 0.0;

            for ( int n_index = 0; n_index < static_cast<int>(n_grid.size()); ++n_index ) {
                double ratio = 0.0;

                if (!edges[e].use_log_integration[k]) {
                    const double num = integrate_direct( fixed_integrand_num, beta, lambda, 1, n_grid, edges, n_grid[n_index], e, k, simpson_weights );

                    const double den = integrate_direct( fixed_integrand_den, beta, lambda, 0, n_grid, edges, n_grid[n_index], e, k, simpson_weights );

                    const bool direct_integrals_valid = std::isfinite(num) && std::isfinite(den) && num > 0.0 && den > 0.0;

                    if (direct_integrals_valid) {
                        ratio = num / den;

                        if (!std::isfinite(ratio)) {
                            edges[e].use_log_integration[k] = true;
                        }
                    } else {
                        edges[e].use_log_integration[k] = true;
                    }

                    if (edges[e].use_log_integration[k]) {
                        cerr << "Switching directed edge to log integration"
                             << " edge=" << e
                             << " k=" << k
                             << " n_index=" << n_index
                             << " n=" << n_grid[n_index]
                             << " num=" << num
                             << " den=" << den
                             << endl;
                    }
                }

                if (edges[e].use_log_integration[k]) {
                    const bool log_ok =
                        compute_ratio_logarithmic( beta, lambda, n_grid, edges, n_grid[n_index], e, k, simpson_weights, ratio );

                    if (!log_ok) {
                        cerr << "Logarithmic integration failed"
                             << " edge=" << e
                             << " k=" << k
                             << " n_index=" << n_index
                             << " n=" << n_grid[n_index]
                             << endl;

                        return
                            std::numeric_limits<double>::quiet_NaN();
                    }
                }

                const double old_cond_av = edges[e].cond_av[k][n_index];

                const double cond_av_new = damping * ratio + (1.0 - damping) * old_cond_av;

                if (!std::isfinite(cond_av_new)) {
                    cerr << "Bad conditional-average update"
                         << " edge=" << e
                         << " k=" << k
                         << " n_index=" << n_index
                         << " ratio=" << ratio
                         << " old=" << old_cond_av
                         << " new=" << cond_av_new
                         << endl;

                    return
                        std::numeric_limits<double>::quiet_NaN();
                }

                const double diff = std::fabs(cond_av_new - old_cond_av);

                if (!std::isfinite(diff)) {
                    cerr << "Bad update difference"
                         << " edge=" << e
                         << " k=" << k
                         << " n_index=" << n_index
                         << " new=" << cond_av_new
                         << " old=" << old_cond_av
                         << endl;

                    return
                        std::numeric_limits<double>::quiet_NaN();
                }

                variation += diff;

                edges[e].cond_av[k][n_index] = cond_av_new;
            }

            update_integrated_cond_av( edges[e].cond_av[k], edges[e].integrated_cond_av[k], n_grid );

            variation *= dn;

            if (variation > max_variation) {
                max_variation = variation;
            }

            edges[e].converged[k] = variation < tol;
        }
    }

    return max_variation;
}


int convergence(Tedge *edges, const vector<double> &n_grid, const vector<double> &simpson_weights, const vector<double> &fixed_integrand_num,
                const vector<double> &fixed_integrand_den, double beta, double lambda, long M, double dn, double damping, double tol,
                int max_iter, bool &divergence, int min_consecutive=5, double maximum=1e10){
    double delta = tol + 1;
    int iter = 0;


    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            edges[e].converged[k] = false;
        }
    }

    int consecutive = 0;
    while (consecutive < min_consecutive && iter < max_iter){
        delta = update_cond_avgs(edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta,
                                 lambda, M, dn, damping, tol);
        iter++;
        cerr << "Iteration " << iter << ", delta = " << delta << endl;
        if (std::isinf(delta) || std::isnan(delta) || delta > maximum){
            divergence = true;
            return iter;
        }
        if (delta < tol){
            consecutive++;
        }else{
            consecutive = 0;
        }
    }
    divergence = false;
    return iter;
}


// Subset-aware analogue of update_cond_avgs: identical per-entry logic (full n_grid range,
// both k=0,1), but only for edges in active_edges. Edges outside active_edges are left
// completely untouched (cond_av, integrated_cond_av, converged[], use_log_integration[]).
double update_cond_avgs_subset( Tedge *edges, const vector<double> &n_grid, const vector<double> &simpson_weights, const vector<double> &fixed_integrand_num, const vector<double> &fixed_integrand_den, double beta, double lambda, const vector<long> &active_edges, double dn, double damping, double tol ) {
    double max_variation = 0.0;

    for (long e : active_edges) {
        for (int k = 0; k < 2; ++k) {
            double variation = 0.0;

            for ( int n_index = 0; n_index < static_cast<int>(n_grid.size()); ++n_index ) {
                double ratio = 0.0;

                if (!edges[e].use_log_integration[k]) {
                    const double num = integrate_direct( fixed_integrand_num, beta, lambda, 1, n_grid, edges, n_grid[n_index], e, k, simpson_weights );

                    const double den = integrate_direct( fixed_integrand_den, beta, lambda, 0, n_grid, edges, n_grid[n_index], e, k, simpson_weights );

                    const bool direct_integrals_valid = std::isfinite(num) && std::isfinite(den) && num > 0.0 && den > 0.0;

                    if (direct_integrals_valid) {
                        ratio = num / den;

                        if (!std::isfinite(ratio)) {
                            edges[e].use_log_integration[k] = true;
                        }
                    } else {
                        edges[e].use_log_integration[k] = true;
                    }

                    if (edges[e].use_log_integration[k]) {
                        cerr << "Switching directed edge to log integration"
                             << " edge=" << e
                             << " k=" << k
                             << " n_index=" << n_index
                             << " n=" << n_grid[n_index]
                             << " num=" << num
                             << " den=" << den
                             << endl;
                    }
                }

                if (edges[e].use_log_integration[k]) {
                    const bool log_ok =
                        compute_ratio_logarithmic( beta, lambda, n_grid, edges, n_grid[n_index], e, k, simpson_weights, ratio );

                    if (!log_ok) {
                        cerr << "Logarithmic integration failed"
                             << " edge=" << e
                             << " k=" << k
                             << " n_index=" << n_index
                             << " n=" << n_grid[n_index]
                             << endl;

                        return
                            std::numeric_limits<double>::quiet_NaN();
                    }
                }

                const double old_cond_av = edges[e].cond_av[k][n_index];

                const double cond_av_new = damping * ratio + (1.0 - damping) * old_cond_av;

                if (!std::isfinite(cond_av_new)) {
                    cerr << "Bad conditional-average update"
                         << " edge=" << e
                         << " k=" << k
                         << " n_index=" << n_index
                         << " ratio=" << ratio
                         << " old=" << old_cond_av
                         << " new=" << cond_av_new
                         << endl;

                    return
                        std::numeric_limits<double>::quiet_NaN();
                }

                const double diff = std::fabs(cond_av_new - old_cond_av);

                if (!std::isfinite(diff)) {
                    cerr << "Bad update difference"
                         << " edge=" << e
                         << " k=" << k
                         << " n_index=" << n_index
                         << " new=" << cond_av_new
                         << " old=" << old_cond_av
                         << endl;

                    return
                        std::numeric_limits<double>::quiet_NaN();
                }

                variation += diff;

                edges[e].cond_av[k][n_index] = cond_av_new;
            }

            update_integrated_cond_av( edges[e].cond_av[k], edges[e].integrated_cond_av[k], n_grid );

            variation *= dn;

            if (variation > max_variation) {
                max_variation = variation;
            }

            edges[e].converged[k] = variation < tol;
        }
    }

    return max_variation;
}


// Subset-aware analogue of convergence(): resets converged[] only for edges in active_edges,
// then iterates update_cond_avgs_subset to the same min_consecutive/tol/max_iter fixed-point
// criterion as convergence(). Edges outside active_edges keep whatever state (cond_av,
// converged[]) they carried in.
int convergence_subset(Tedge *edges, const vector<double> &n_grid, const vector<double> &simpson_weights, const vector<double> &fixed_integrand_num,
                const vector<double> &fixed_integrand_den, double beta, double lambda, const vector<long> &active_edges, double dn, double damping, double tol,
                int max_iter, bool &divergence, int min_consecutive=5, double maximum=1e10){
    double delta = tol + 1;
    int iter = 0;

    for (long e : active_edges){
        for (int k = 0; k < 2; k++){
            edges[e].converged[k] = false;
        }
    }

    int consecutive = 0;
    while (consecutive < min_consecutive && iter < max_iter){
        delta = update_cond_avgs_subset(edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta,
                                 lambda, active_edges, dn, damping, tol);
        iter++;
        cerr << "Iteration " << iter << ", delta = " << delta << " (subset, " << active_edges.size() << " active edges)" << endl;
        if (std::isinf(delta) || std::isnan(delta) || delta > maximum){
            divergence = true;
            return iter;
        }
        if (delta < tol){
            consecutive++;
        }else{
            consecutive = 0;
        }
    }
    divergence = false;
    return iter;
}



double sum_over_neighs(long node, Tnode *nodes, Tedge *edges, int n_index_node){
    double sum = 0;
    long edge_neigh;
    int pos_there;
    for (long j = 0; j < nodes[node].edges_in.size(); j++){
        edge_neigh = nodes[node].edges_in[j];
        pos_there = nodes[node].pos_there[j];
        sum += edges[edge_neigh].links[pos_there] * edges[edge_neigh].integrated_cond_av[1 - pos_there][n_index_node];
    }
    return sum;
}

double integrate_node_log( double beta, double lambda, int p, const vector<double> &n_grid, const vector<double> &simpson_weights, long node, Tnode *nodes, Tedge *edges ) {
    const double minus_infinity = -std::numeric_limits<double>::infinity();

    if ( n_grid.size() < 2 || simpson_weights.size() != n_grid.size() - 1 ) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double zero_interval_power = beta * lambda + p;

    const double n1 = n_grid[1];

    if (zero_interval_power <= 0.0 || n1 <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double sum_neighs_at_n1 = sum_over_neighs( node, nodes, edges, 1 );

    double log_integral = zero_interval_power * std::log(n1) - std::log(zero_interval_power) - beta * ( 0.5 * (n1 - 1.0) * (n1 - 1.0) + sum_neighs_at_n1 );

    if (invalid_log_value(log_integral)) { return std::numeric_limits<double>::quiet_NaN(); }

    for ( int n_index = 1; n_index < static_cast<int>(n_grid.size()); ++n_index ) {
        const double n = n_grid[n_index];
        const double weight = simpson_weights[n_index - 1];

        if (n <= 0.0 || weight <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double sum_neighs = sum_over_neighs( node, nodes, edges, n_index );

        const double log_term = std::log(weight) + (beta * lambda - 1.0 + p) * std::log(n) - beta * ( 0.5 * (n - 1.0) * (n - 1.0) + sum_neighs );

        if (invalid_log_value(log_term)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (log_term != minus_infinity) {
            log_integral = log_add_exp( log_integral, log_term );
        }
    }

    return log_integral;
}


double integrate_node_gaussian_part_log( double beta, const vector<double> &n_grid, const vector<double> &simpson_weights, long node, Tnode *nodes, Tedge *edges ) {
    if ( n_grid.size() < 2 || simpson_weights.size() != n_grid.size() - 1 ) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double first_width = n_grid[1] - n_grid[0];

    if (first_width <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double sum_zero = sum_over_neighs( node, nodes, edges, 0 );

    const double sum_one = sum_over_neighs( node, nodes, edges, 1 );

    const double log_g_zero = -beta * ( 0.5 * (n_grid[0] - 1.0) * (n_grid[0] - 1.0) + sum_zero );

    const double log_g_one = -beta * ( 0.5 * (n_grid[1] - 1.0) * (n_grid[1] - 1.0) + sum_one );

    if ( invalid_log_value(log_g_zero) || invalid_log_value(log_g_one) ) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double log_integral = std::log(0.5 * first_width) + log_add_exp( log_g_zero, log_g_one );

    for ( int n_index = 1; n_index < static_cast<int>(n_grid.size()); ++n_index ) {
        const double weight =
            simpson_weights[n_index - 1];

        if (weight <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double n = n_grid[n_index];

        const double sum_neighs = sum_over_neighs( node, nodes, edges, n_index );

        const double log_term = std::log(weight) - beta * ( 0.5 * (n - 1.0) * (n - 1.0) + sum_neighs );

        if (invalid_log_value(log_term)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        log_integral = log_add_exp( log_integral, log_term );
    }

    return log_integral;
}


bool compute_averages( Tnode *nodes, Tedge *edges, const vector<double> &n_grid, const vector<double> &simpson_weights, const vector<double> &fixed_integrand_num, const vector<double> &fixed_integrand_den, double beta, double lambda, long N, double &av_n_graph, double &av_var_n_graph, double &error_av_n_graph, double &error_var_n_graph ) {
    (void)fixed_integrand_num;
    (void)fixed_integrand_den;

    if ( N <= 0 || beta <= 0.0 || n_grid.size() < 2 ) {
        return false;
    }

    av_n_graph = 0.0;
    av_var_n_graph = 0.0;

    double av_sqr_av_n_graph = 0.0;
    double av_sqr_var_n_graph = 0.0;

    for (long i = 0; i < N; ++i) { const double log_i0 = integrate_node_log( beta, lambda, 0, n_grid, simpson_weights, i, nodes, edges );

        const double log_i1 = integrate_node_log( beta, lambda, 1, n_grid, simpson_weights, i, nodes, edges );

        const double log_i2 = integrate_node_log( beta, lambda, 2, n_grid, simpson_weights, i, nodes, edges );

        const double log_gaussian_normalization = integrate_node_gaussian_part_log( beta, n_grid, simpson_weights, i, nodes, edges );

        if ( !std::isfinite(log_i0) || !std::isfinite(log_i1) || !std::isfinite(log_i2) || !std::isfinite(log_gaussian_normalization) ) {
            cerr << "Bad node logarithmic integral"
                 << " node=" << i
                 << " log_i0=" << log_i0
                 << " log_i1=" << log_i1
                 << " log_i2=" << log_i2
                 << " log_gaussian_norm="
                 << log_gaussian_normalization
                 << endl;

            return false;
        }

        nodes[i].log_normalization_Psingle = log_i0;

        nodes[i].log_normalization_Psingle_gaussian_part = log_gaussian_normalization;

        double mean = 0.0;
        double second_moment = 0.0;

        if (
            !exp_from_log( log_i1 - log_i0, mean )
            || !exp_from_log( log_i2 - log_i0, second_moment )
        ) {
            cerr << "Bad node moment ratio"
                 << " node=" << i
                 << " log_mean="
                 << log_i1 - log_i0
                 << " log_second_moment="
                 << log_i2 - log_i0
                 << endl;

            return false;
        }

        double variance = 0.0;

        if (
            !nonnegative_difference( second_moment, mean * mean, variance )
        ) {
            cerr << "Negative or invalid node variance"
                 << " node=" << i
                 << " second_moment=" << second_moment
                 << " mean=" << mean
                 << endl;

            return false;
        }

        nodes[i].av_abundance = mean;
        nodes[i].var_abundance = variance;

        av_n_graph += mean;
        av_var_n_graph += variance;

        av_sqr_av_n_graph += mean * mean;
        av_sqr_var_n_graph += variance * variance;

        nodes[i].response_around_zero = 0.0;

        for ( long j = 0; j < static_cast<long>(nodes[i].edges_in.size()); ++j ) {
            const long edge_neigh = nodes[i].edges_in[j];

            const int pos_there = nodes[i].pos_there[j];

            edges[edge_neigh] .response_around_zero[pos_there] = -( edges[edge_neigh] .cond_av[pos_there][1] - edges[edge_neigh] .cond_av[pos_there][0] ) / beta / n_grid[1];

            nodes[i].response_around_zero += edges[edge_neigh] .response_around_zero[pos_there] / edges[edge_neigh] .links[pos_there];
        }

        nodes[i].response_around_zero /= nodes[i].edges_in.size();
    }

    av_n_graph /= N;
    av_var_n_graph /= N;

    av_sqr_av_n_graph /= N;
    av_sqr_var_n_graph /= N;

    double variance_of_node_means = 0.0;
    double variance_of_node_variances = 0.0;

    if (
        !nonnegative_difference( av_sqr_av_n_graph, av_n_graph * av_n_graph, variance_of_node_means )
        || !nonnegative_difference( av_sqr_var_n_graph, av_var_n_graph * av_var_n_graph, variance_of_node_variances )
    ) {
        cerr << "Bad graph-level variance used for error bars"
             << endl;

        return false;
    }

    error_av_n_graph = std::sqrt( variance_of_node_means / N );

    error_var_n_graph = std::sqrt( variance_of_node_variances / N );

    return std::isfinite(av_n_graph)
        && std::isfinite(av_var_n_graph)
        && std::isfinite(error_av_n_graph)
        && std::isfinite(error_var_n_graph);
}


void compute_responses(Tnode *nodes, Tedge *edges, const vector <double> &n_grid, double beta, long M){
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            for (int n_index = 1; n_index < n_grid.size(); n_index++){
                if(nodes[edges[e].nodes_in[k]].av_abundance >= n_grid[n_index - 1] && nodes[edges[e].nodes_in[k]].av_abundance < n_grid[n_index]){
                    edges[e].response_around_average[k] = -(edges[e].cond_av[k][n_index] - edges[e].cond_av[k][n_index - 1]) / beta / (n_grid[n_index] - n_grid[n_index - 1]);
                    break;
                }
            }
        }
    }
}


bool compute_distributions( Tnode *nodes, Tedge *edges, const vector<double> &n_grid, const vector<double> &fixed_integrand_den, double beta, double lambda, long N ) {
    (void)fixed_integrand_den;

    if (n_grid.size() < 2) {
        return false;
    }

    for (long i = 0; i < N; ++i) {
        if (
            !std::isfinite(
                nodes[i].log_normalization_Psingle
            )
            || !std::isfinite(
                nodes[i]
                    .log_normalization_Psingle_gaussian_part
            )
        ) {
            cerr << "Missing logarithmic normalization"
                 << " node=" << i
                 << endl;

            return false;
        }

        nodes[i].Psingle = vector<double>( n_grid.size() - 1, 0.0 );

        nodes[i].Psingle_gaussian_part = vector<double>( n_grid.size() - 1, 0.0 );

        for ( int n_index = 1; n_index < static_cast<int>(n_grid.size()); ++n_index ) {
            const double n = n_grid[n_index];

            if (n <= 0.0) {
                return false;
            }

            const double sum_neighs = sum_over_neighs( i, nodes, edges, n_index );

            const double log_gaussian_part = -beta * ( 0.5 * (n - 1.0) * (n - 1.0) + sum_neighs );

            const double log_psingle_unnormalized = (beta * lambda - 1.0) * std::log(n) + log_gaussian_part;

            const double log_psingle = log_psingle_unnormalized - nodes[i].log_normalization_Psingle;

            const double log_psingle_gaussian_part = log_gaussian_part - nodes[i] .log_normalization_Psingle_gaussian_part;

            double psingle = 0.0;
            double psingle_gaussian_part = 0.0;

            if (
                !exp_from_log( log_psingle, psingle )
                || !exp_from_log( log_psingle_gaussian_part, psingle_gaussian_part )
            ) {
                cerr << "Bad logarithmic distribution value"
                     << " node=" << i
                     << " n_index=" << n_index
                     << " n=" << n
                     << " log_psingle="
                     << log_psingle
                     << " log_gaussian_part="
                     << log_psingle_gaussian_part
                     << endl;

                return false;
            }

            nodes[i].Psingle[n_index - 1] = psingle;

            nodes[i] .Psingle_gaussian_part[n_index - 1] = psingle_gaussian_part;
        }
    }

    return true;
}


// boundary_ratios[i] = P_i(nmax) / max_n P_i(n), or +inf if that boundary value itself is
// non-finite. 0.0 for nodes with no distribution computed yet. Used both to derive the
// worst-case scalar check (max_boundary_ratio_Psingle) and to drive selective nmax growth
// (flag_bad_nodes), so the per-node detail isn't computed twice.
vector<double> boundary_ratio_per_node( const Tnode *nodes, long N ) {
    vector<double> ratios( N, 0.0 );

    for (long i = 0; i < N; ++i) {
        const vector<double> &P = nodes[i].Psingle;

        if (P.empty()) {
            continue;
        }

        double pmax = 0.0;

        for (double value : P) {
            if (std::isfinite(value) && value > pmax) {
                pmax = value;
            }
        }

        if (pmax > 0.0) {
            const double boundary_value = P.back();

            ratios[i] = std::isfinite(boundary_value)
                ? boundary_value / pmax
                : std::numeric_limits<double>::infinity();
        }
    }

    return ratios;
}


double max_boundary_ratio_Psingle( const Tnode *nodes, long N ) {
    const vector<double> ratios = boundary_ratio_per_node(nodes, N);

    double worst_ratio = 0.0;

    for (double ratio : ratios) {
        if (!std::isfinite(ratio)) {
            return std::numeric_limits<double>::infinity();
        }

        if (ratio > worst_ratio) {
            worst_ratio = ratio;
        }
    }

    return worst_ratio;
}


bool need_larger_nmax( const Tnode *nodes, long N, double tail_tol ) {
    const double worst_ratio =
        max_boundary_ratio_Psingle(nodes, N);

    cerr << "Adaptive nmax check: "
         << "max_i P_i(nmax)/max(P_i) = "
         << worst_ratio
         << endl;

    return !std::isfinite(worst_ratio)
        || worst_ratio > tail_tol;
}


// Nodes whose tail past the current nmax is not (yet) negligible: ratio > tail_tol, or
// non-finite (treated conservatively as bad). Drives selective nmax growth.
vector<bool> flag_bad_nodes( const vector<double> &boundary_ratios, double tail_tol ) {
    vector<bool> bad( boundary_ratios.size(), false );

    for (size_t i = 0; i < boundary_ratios.size(); ++i) {
        bad[i] = !std::isfinite( boundary_ratios[i] ) || boundary_ratios[i] > tail_tol;
    }

    return bad;
}


// Edges touching at least one bad node (either endpoint) -- the edges whose cond_av must be
// recomputed to extend the bad nodes' distributions; see the derivation in the plan for why
// this 1-hop rule is sufficient and self-correcting.
vector<long> build_active_edge_list( const Tedge *edges, long M, const vector<bool> &bad_node_mask ) {
    vector<long> active_edges;

    for (long e = 0; e < M; ++e) {
        const bool node0_bad = bad_node_mask[ edges[e].nodes_in[0] ];
        const bool node1_bad = bad_node_mask[ edges[e].nodes_in[1] ];

        if (node0_bad || node1_bad) {
            active_edges.push_back(e);
        }
    }

    return active_edges;
}


bool grid_is_strict_extension( const vector<double> &old_grid, const vector<double> &new_grid ) {
    if (new_grid.size() <= old_grid.size()) {
        return false;
    }

    for (size_t i = 0; i < old_grid.size(); ++i) {
        const double scale =
            std::max(
                1.0,
                std::max( std::fabs(old_grid[i]), std::fabs(new_grid[i]) )
            );

        const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;

        if ( std::fabs(old_grid[i] - new_grid[i]) > tolerance ) {
            return false;
        }
    }

    return true;
}


bool extend_conditional_messages_constant_tail( Tedge *edges, long M, const vector<double> &old_grid, const vector<double> &new_grid ) {
    if (!grid_is_strict_extension( old_grid, new_grid )) {
        return false;
    }

    const size_t old_size = old_grid.size();
    const size_t new_size = new_grid.size();

    for (long e = 0; e < M; ++e) {
        for (int k = 0; k < 2; ++k) {
            const vector<double> &cond_av =
                edges[e].cond_av[k];

            const vector<double> &integrated_cond_av =
                edges[e].integrated_cond_av[k];

            if (
                cond_av.size() != old_size
                || integrated_cond_av.size() != old_size
                || cond_av.empty()
                || !std::isfinite(cond_av.back())
            ) {
                cerr
                    << "Cannot extend conditional message"
                    << " edge=" << e
                    << " k=" << k
                    << " cond_size=" << cond_av.size()
                    << " integrated_size="
                    << integrated_cond_av.size()
                    << " expected_size=" << old_size
                    << endl;

                return false;
            }
        }
    }

    for (long e = 0; e < M; ++e) {
        for (int k = 0; k < 2; ++k) {
            vector<double> &cond_av = edges[e].cond_av[k];

            vector<double> &integrated_cond_av = edges[e].integrated_cond_av[k];

            const double boundary_value = cond_av.back();

            cond_av.resize( new_size, boundary_value );

            integrated_cond_av.resize(new_size);

            for ( size_t n_index = old_size; n_index < new_size; ++n_index ) {
                integrated_cond_av[n_index] = integrated_cond_av[n_index - 1] + 0.5 * ( cond_av[n_index - 1] + cond_av[n_index] ) * ( new_grid[n_index] - new_grid[n_index - 1] );
            }
        }
    }

    return true;
}


size_t PBMF_ansatz_single_try(Tnode *nodes, Tedge *edges, const vector <double> &n_grid, const vector <double> &simpson_weights, const vector <double> &fixed_integrand_num,
                              const vector <double> &fixed_integrand_den, double beta, double lambda, long N, long M, double dn, double damping,
                              double tol, int max_iter, long sequence[], unsigned long seed_seq, double avn_0, bool random_init, 
                              double std_n0, unsigned long seed_condinit, int &iter, double hmin, double hmax, double **coefficients,
                              double *gamma_vals, bool &divergence){
    produce_random_seq(seed_seq, M, sequence);
    init_avgs(N, nodes, avn_0, random_init, std_n0, seed_condinit);
    init_cond_av_single_avgs(M, beta, lambda, nodes, edges, n_grid, hmin, hmax, coefficients, gamma_vals);
    
    reset_log_integration_flags(edges, M);

    auto start = std::chrono::high_resolution_clock::now();
    iter = convergence(edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, 
                       M, dn, damping, tol, max_iter, divergence);
    auto end = std::chrono::high_resolution_clock::now();
    size_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    return elapsed;
}


size_t PBMF_ansatz_single_try_adaptive_nmax( Tnode *nodes, Tedge *edges, vector<double> &n_grid, vector<double> &simpson_weights, vector<double> &fixed_integrand_num, vector<double> &fixed_integrand_den, double beta, double lambda, long N, long M, double &nmax, double n1, double dn, double damping, double tol, int max_iter, long sequence[], unsigned long seed_seq, double avn_0, bool random_init, double std_n0, unsigned long seed_condinit, int &iter, double hmin, double hmax, double **coefficients, double *gamma_vals, bool &divergence, double tail_tol, double nmax_growth, double nmax_limit, double &av_n_graph, double &av_var_n_graph, double &error_av_n_graph, double &error_var_n_graph ) {
    size_t total_elapsed = 0;

    cerr
        << "Running PBMF with requested nmax = "
        << nmax
        << endl;

    init_auxiliary_vectors( n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, n1, dn, nmax );

    init_cond_av( M, edges, n_grid.size());

    total_elapsed += PBMF_ansatz_single_try( nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, N, M, dn, damping, tol, max_iter, sequence, seed_seq, avn_0, random_init, std_n0, seed_condinit, iter, hmin, hmax, coefficients, gamma_vals, divergence );

    // FULL: retry with the unrestricted, full-graph convergence() -- used when convergence
    // itself failed (no reliable localized signal for which node(s) are responsible).
    // SELECTIVE: retry restricted to edges touching the nodes whose tail actually violated
    // tail_tol -- used for the well-diagnosed tail-tolerance-triggered growth case.
    enum class RetryKind { SELECTIVE, FULL };

    while (true) {
        const bool averages_ok = compute_averages( nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, N, av_n_graph, av_var_n_graph, error_av_n_graph, error_var_n_graph );

        const bool distributions_ok = averages_ok && compute_distributions( nodes, edges, n_grid, fixed_integrand_den, beta, lambda, N );

        RetryKind retry;

        if (!averages_ok || !distributions_ok) {
            cerr
                << "Adaptive nmax stopped because averages or "
                << "distributions could not be computed."
                << endl;

            divergence = true;
            break;
        } else if (divergence || iter >= max_iter) {
            cerr
                << "Adaptive nmax did not converge; retrying "
                << "with a larger nmax using full-graph convergence."
                << endl;

            retry = RetryKind::FULL;
        } else if ( !need_larger_nmax( nodes, N, tail_tol ) ) {
            cerr
                << "Adaptive nmax accepted:"
                << " requested_nmax=" << nmax
                << " actual_grid_boundary="
                << n_grid.back()
                << endl;

            break;
        } else {
            retry = RetryKind::SELECTIVE;
        }

        const double proposed_nmax = nmax * nmax_growth;

        if (!std::isfinite(proposed_nmax)) {
            cerr
                << "Adaptive nmax stopped because the proposed "
                << "nmax is not finite."
                << endl;

            break;
        } else if (proposed_nmax > nmax_limit) {
            if (nmax < nmax_limit) {
                cerr
                    << "Increasing nmax from "
                    << nmax
                    << " to the limit "
                    << nmax_limit
                    << endl;

                nmax = nmax_limit;
            } else {
                cerr
                    << "Warning: "
                    << (retry == RetryKind::FULL
                        ? "the iteration still did not converge "
                        : "the tail criterion is still not satisfied ")
                    << "at nmax_limit = "
                    << nmax_limit
                    << endl;

                break;
            }
        } else {
            cerr
                << "Increasing nmax from "
                << nmax
                << " to "
                << proposed_nmax
                << endl;

            nmax = proposed_nmax;
        }

        cerr
            << "Running PBMF with requested nmax = "
            << nmax
            << endl;

        vector<double> new_n_grid;
        vector<double> new_simpson_weights;
        vector<double> new_fixed_integrand_num;
        vector<double> new_fixed_integrand_den;

        init_auxiliary_vectors( new_n_grid, new_simpson_weights, new_fixed_integrand_num, new_fixed_integrand_den, beta, lambda, n1, dn, nmax );

        if (!grid_is_strict_extension( n_grid, new_n_grid )) {
            cerr
                << "Adaptive nmax failed: the new grid is not "
                << "a strict extension of the old grid."
                << " old_size=" << n_grid.size()
                << " new_size=" << new_n_grid.size()
                << " old_boundary=" << n_grid.back()
                << " requested_nmax=" << nmax
                << endl;

            divergence = true;
            break;
        }

        if ( !extend_conditional_messages_constant_tail( edges, M, n_grid, new_n_grid ) ) {
            cerr
                << "Adaptive nmax failed while extending "
                << "conditional messages."
                << endl;

            divergence = true;
            break;
        }

        n_grid.swap(new_n_grid);

        simpson_weights.swap( new_simpson_weights );

        fixed_integrand_num.swap( new_fixed_integrand_num );

        fixed_integrand_den.swap( new_fixed_integrand_den );

        cerr
            << "Warm-starting on extended grid:"
            << " points=" << n_grid.size()
            << " actual_boundary=" << n_grid.back()
            << endl;

        const auto start = std::chrono::high_resolution_clock::now();

        if (retry == RetryKind::FULL) {
            cerr << "Warm-starting with full-graph convergence ("
                 << M << " edges)." << endl;

            iter = convergence( edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, M, dn, damping, tol, max_iter, divergence );
        } else {
            const vector<double> boundary_ratios = boundary_ratio_per_node( nodes, N );
            const vector<bool> bad_node_mask = flag_bad_nodes( boundary_ratios, tail_tol );
            const vector<long> active_edges = build_active_edge_list( edges, M, bad_node_mask );

            cerr << "Warm-starting with selective convergence ("
                 << active_edges.size() << " / " << M << " edges active)." << endl;

            iter = convergence_subset( edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, active_edges, dn, damping, tol, max_iter, divergence );
        }

        const auto end = std::chrono::high_resolution_clock::now();

        total_elapsed += std::chrono::duration_cast< std::chrono::milliseconds >(end - start).count();
    }

    return total_elapsed;
}


void several_seq_PBMF(unsigned long seed_graph, unsigned long seed_seq_init, 
                      long N, long M, Tnode *nodes, Tedge *edges, double T, double lambda, double tol,
                      int max_iter, unsigned long num_seq, double tol_fixed_point,
                      double avn_0, double damping, bool print_only_last, bool print_avgs, 
                      bool print_responses, bool print_distributions, char * fileout_base, 
                      bool random_init, double std_n0, unsigned long id_0, int num_init_conds,
                      double n1, double dn, double nmax, bool adaptive_nmax, double tail_tol, 
                      double nmax_growth, double nmax_limit){
    double beta = 1.0 / T;

    double hmax = find_divergence_max(beta, 1 + beta * lambda / 2);
    double **coefficients, *gamma_vals;
    comp_coefficients(beta, lambda, coefficients, gamma_vals);
    double hmin = find_divergence_min(beta, lambda, coefficients);

    long *sequence;
    sequence = new long[M];
    bool divergence;

    char fileavgs[400];
    char fileresponses[400];
    char filedistributions[400];
    
    
    divergence = false;
    unsigned long seed_seq, seed_condinit;
    bool make_other_tries;
    int iter;
    bool same_fixed_point = true;
    double av_n_graph, av_var_n_graph, error_av_n_graph, error_var_n_graph;
    
    vector <double> n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den;


    if (!adaptive_nmax) {
    init_auxiliary_vectors(n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, n1, dn, nmax);
    init_cond_av(M, edges, n_grid.size());
    }

    seed_seq = seed_seq_init;
    seed_condinit = id_0;

size_t elapsed;

if (adaptive_nmax) {
    elapsed = PBMF_ansatz_single_try_adaptive_nmax( nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, N, M, nmax, n1, dn, damping, tol, max_iter, sequence, seed_seq, avn_0, random_init, std_n0, seed_condinit, iter, hmin, hmax, coefficients, gamma_vals, divergence, tail_tol, nmax_growth, nmax_limit, av_n_graph, av_var_n_graph, error_av_n_graph, error_var_n_graph );
} else {
    elapsed = PBMF_ansatz_single_try( nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, N, M, dn, damping, tol, max_iter, sequence, seed_seq, avn_0, random_init, std_n0, seed_condinit, iter, hmin, hmax, coefficients, gamma_vals, divergence );

    compute_averages( nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, N, av_n_graph, av_var_n_graph, error_av_n_graph, error_var_n_graph );
}
        
    if (!print_only_last){
        print_results(av_n_graph, error_av_n_graph, av_var_n_graph, error_var_n_graph, iter, nodes, edges, N, M, seed_graph, seed_seq, seed_condinit, max_iter, divergence, true, elapsed, lambda);
        if (print_avgs){
            sprintf(fileavgs, "%s_nmaxlimit_%.3lf_av_abundances_seedseq_%li_seedinit_%li.txt", fileout_base, nmax_limit, seed_seq, seed_condinit);
            print_node_avgs_to_file(nodes, N, fileavgs);
        }
        if (print_responses){
            sprintf(fileresponses,"%s_nmaxlimit_%.3lf_responses_seedseq_%li_seedinit_%li.txt",fileout_base, nmax_limit, seed_seq, seed_condinit);
            compute_responses(nodes, edges, n_grid, beta, M);
            print_responses_to_file(edges, M, fileresponses);
        }
        if (print_distributions){
            sprintf(filedistributions,"%s_nmaxlimit_%.3lf_distributions_seedseq_%li_seedinit_%li.txt",fileout_base, nmax_limit, seed_seq, seed_condinit);
            compute_distributions(nodes, edges, n_grid, fixed_integrand_den, beta, lambda, N);
            print_distributions_to_file(nodes, n_grid, N, filedistributions);
        }
    }else if(divergence || iter >= max_iter){
        print_results(av_n_graph, error_av_n_graph, av_var_n_graph, error_var_n_graph, iter, nodes, edges, N, M, seed_graph, seed_seq, seed_condinit, max_iter, divergence, true, elapsed, lambda);
        if (print_avgs){
            sprintf(fileavgs, "%s_nmaxlimit_%.3lf_av_abundances_seedseq_%li_seedinit_%li.txt", fileout_base, nmax_limit, seed_seq, seed_condinit);
            print_node_avgs_to_file(nodes, N, fileavgs);
        }
        if (print_responses){
            sprintf(fileresponses,"%s_nmaxlimit_%.3lf_responses_seedseq_%li_seedinit_%li.txt",fileout_base, nmax_limit, seed_seq, seed_condinit);
            compute_responses(nodes, edges, n_grid, beta, M);
            print_responses_to_file(edges, M, fileresponses);
        }
        if (print_distributions){
            sprintf(filedistributions,"%s_nmaxlimit_%.3lf_distributions_seedseq_%li_seedinit_%li.txt",fileout_base, nmax_limit, seed_seq, seed_condinit);
            compute_distributions(nodes, edges, n_grid, fixed_integrand_den, beta, lambda, N);
            print_distributions_to_file(nodes, n_grid, N, filedistributions);
        }
    }

    make_other_tries = !print_only_last || (!divergence && iter < max_iter);
    
    if (make_other_tries){
        set_av_prev(nodes, N);
        bool cond = true;

        seed_seq = seed_seq_init + 1;
        while (seed_seq < seed_seq_init + num_seq && cond){

        if (adaptive_nmax) {
    elapsed = PBMF_ansatz_single_try_adaptive_nmax( nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, N, M, nmax, n1, dn, damping, tol, max_iter, sequence, seed_seq, avn_0, random_init, std_n0, seed_condinit, iter, hmin, hmax, coefficients, gamma_vals, divergence, tail_tol, nmax_growth, nmax_limit, av_n_graph, av_var_n_graph, error_av_n_graph, error_var_n_graph );
} else {
    elapsed = PBMF_ansatz_single_try( nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, N, M, dn, damping, tol, max_iter, sequence, seed_seq, avn_0, random_init, std_n0, seed_condinit, iter, hmin, hmax, coefficients, gamma_vals, divergence );

    compute_averages( nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, N, av_n_graph, av_var_n_graph, error_av_n_graph, error_var_n_graph );
}

            same_fixed_point = compare_fixed_points(nodes, N, tol_fixed_point);
            if (!print_only_last){
                print_results(av_n_graph, error_av_n_graph, av_var_n_graph, error_var_n_graph, iter, nodes, edges, N, M, seed_graph, seed_seq, seed_condinit, max_iter, divergence, same_fixed_point, elapsed, lambda);
                if (print_avgs){
                    sprintf(fileavgs, "%s_nmaxlimit_%.3lf_av_abundances_seedseq_%li_seedinit_%li.txt", fileout_base, nmax_limit, seed_seq, seed_condinit);
                    print_node_avgs_to_file(nodes, N, fileavgs);
                }
                if (print_responses){
                    sprintf(fileresponses,"%s_nmaxlimit_%.3lf_responses_seedseq_%li_seedinit_%li.txt",fileout_base, nmax_limit, seed_seq, seed_condinit);
                    compute_responses(nodes, edges, n_grid, beta, M);
                    print_responses_to_file(edges, M, fileresponses);
                }
                if (print_distributions){
                    sprintf(filedistributions,"%s_nmaxlimit_%.3lf_distributions_seedseq_%li_seedinit_%li.txt",fileout_base, nmax_limit, seed_seq, seed_condinit);
                    compute_distributions(nodes, edges, n_grid, fixed_integrand_den, beta, lambda, N);
                    print_distributions_to_file(nodes, n_grid, N, filedistributions);
                }
            }else{
                if (!same_fixed_point || divergence || iter >= max_iter){
                    cond = false;
                }
            }
            seed_seq++;
        }
        
        seed_condinit++;

        while (seed_condinit < id_0 + num_init_conds && cond){
            seed_seq = seed_seq_init;
            while (seed_seq < seed_seq_init + num_seq && cond){

            if (adaptive_nmax) {
    elapsed = PBMF_ansatz_single_try_adaptive_nmax( nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, N, M, nmax, n1, dn, damping, tol, max_iter, sequence, seed_seq, avn_0, random_init, std_n0, seed_condinit, iter, hmin, hmax, coefficients, gamma_vals, divergence, tail_tol, nmax_growth, nmax_limit, av_n_graph, av_var_n_graph, error_av_n_graph, error_var_n_graph );
} else {
    elapsed = PBMF_ansatz_single_try( nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, N, M, dn, damping, tol, max_iter, sequence, seed_seq, avn_0, random_init, std_n0, seed_condinit, iter, hmin, hmax, coefficients, gamma_vals, divergence );

    compute_averages( nodes, edges, n_grid, simpson_weights, fixed_integrand_num, fixed_integrand_den, beta, lambda, N, av_n_graph, av_var_n_graph, error_av_n_graph, error_var_n_graph );
}

                same_fixed_point = compare_fixed_points(nodes, N, tol_fixed_point);
                if (!print_only_last){
                    print_results(av_n_graph, error_av_n_graph, av_var_n_graph, error_var_n_graph, iter, nodes, edges, N, M, seed_graph, seed_seq, seed_condinit, max_iter, divergence, same_fixed_point, elapsed, lambda);
                    if (print_avgs){
                        sprintf(fileavgs, "%s_nmaxlimit_%.3lf_av_abundances_seedseq_%li_seedinit_%li.txt", fileout_base, nmax_limit, seed_seq, seed_condinit);
                        print_node_avgs_to_file(nodes, N, fileavgs);
                    }
                    if (print_responses){
                        sprintf(fileresponses,"%s_nmaxlimit_%.3lf_responses_seedseq_%li_seedinit_%li.txt",fileout_base, nmax_limit, seed_seq, seed_condinit);
                        compute_responses(nodes, edges, n_grid, beta, M);
                        print_responses_to_file(edges, M, fileresponses);
                    }
                    if (print_distributions){
                        sprintf(filedistributions,"%s_nmaxlimit_%.3lf_distributions_seedseq_%li_seedinit_%li.txt",fileout_base, nmax_limit, seed_seq, seed_condinit);
                        compute_distributions(nodes, edges, n_grid, fixed_integrand_den, beta, lambda, N);
                        print_distributions_to_file(nodes, n_grid, N, filedistributions);
                    }
                }else{
                    if (!same_fixed_point || divergence || iter >= max_iter){
                        cond = false;
                    }
                }
                seed_seq++;
            }
            seed_condinit++;
        }


        if (print_only_last){
            print_results(av_n_graph, error_av_n_graph, av_var_n_graph, error_var_n_graph, iter, nodes, edges, N, M, seed_graph, seed_seq-1, seed_condinit-1, max_iter, divergence, true, elapsed, lambda);
            if (print_avgs){
                sprintf(fileavgs,"%s_nmaxlimit_%.3lf_av_abundances_seedseq_%li_seedinit_%li.txt",fileout_base, nmax_limit, seed_seq - 1, seed_condinit - 1);
                print_node_avgs_to_file(nodes, N, fileavgs);
            }
            if (print_responses){
                sprintf(fileresponses,"%s_nmaxlimit_%.3lf_responses_seedseq_%li_seedinit_%li.txt",fileout_base, nmax_limit, seed_seq - 1, seed_condinit - 1);
                compute_responses(nodes, edges, n_grid, beta, M);
                print_responses_to_file(edges, M, fileresponses);
            }
            if (print_distributions){
                sprintf(filedistributions,"%s_nmaxlimit_%.3lf_distributions_seedseq_%li_seedinit_%li.txt",fileout_base, nmax_limit, seed_seq - 1, seed_condinit - 1);
                compute_distributions(nodes, edges, n_grid, fixed_integrand_den, beta, lambda, N);
                print_distributions_to_file(nodes, n_grid, N, filedistributions);
            }
        }
    }
    delete [] sequence;
}



#endif