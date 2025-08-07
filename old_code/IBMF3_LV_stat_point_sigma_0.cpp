#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_integration.h>
#include "math.h"
#include <cmath>

using namespace std;


typedef struct{
    double field; // local field in that node
    double var; // variance of the perturbed gaussian that depends on the neighbors
    double W; // third moment of the perturbed gaussian that depends on the neighbors
    double av; // average value of n in that node
    double q_sqr; // average value of n^2 in that node
    double w3; // average value of n^3 in that node
}Tnode;



double field_in(Tnode node, int c, double mu){
    return 1 - c * mu * node.av;
}


double var_in(Tnode node, double beta, int c, double mu){
    return 1.0 / (1 - beta * c * mu * mu * (node.q_sqr - node.av * node.av));
}


double W_in(Tnode node, double beta, int c, double mu){
    return beta * beta * mu * mu * mu * c * 
           (node.w3 - 3 * node.av * node.q_sqr + 2 * node.av * node.av * node.av);
}


double numerator_av_IBMF(double beta, double lambda, double hi_div_s, double hi2_div_s){
    return 2 * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi2_div_s / 2) + 
           sqrt(2 * beta) * hi_div_s * beta * lambda * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi2_div_s / 2);
}


double numerator_q_sqr_IBMF2(double beta, double lambda, double hi_div_s, double hi2_div_s){
    return 4 * hi_div_s * gsl_sf_gamma((3 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 1.5, -beta * hi2_div_s / 2) + 
           sqrt(2 * beta) * lambda * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1(-(1 + beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2);
}


double denominator_IBMF(double beta, double lambda, double hi_div_s, double hi2_div_s){
    return sqrt(2 * beta) * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2) + 
           2 * beta * hi_div_s * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi2_div_s / 2);
}


double integrand_av(double ni, void *params){
    double *params_d = (double *) params;
    double beta = params_d[0];
    double lambda = params_d[1];
    double W = params_d[2];
    double S2 = params_d[3];
    double M = params_d[4];
    double m = params_d[5];
    double integrand_av = pow(ni, beta * lambda) * 
                          exp(-beta * (W * ni * ni * ni / 6 + ni * ni / 2 * (1.0 / S2 - m * W) - 
                                       ni * (M + m * (1 - S2) - W * m * m / 2)));
    return integrand_av;
}

double integrand_q2(double ni, void *params){
    return integrand_av(ni, params) * ni;
}

double integrand_w3(double ni, void *params){
    return integrand_q2(ni, params) * ni;
}


double numerator_IBMF3(double beta, double lambda, double W, double S2, double M, double m, 
                       gsl_function function, double epsabs, double epsrel, long limit, 
                       gsl_integration_workspace *workspace, double *params){
    double result, error;
    params[0] = beta;
    params[1] = lambda;
    params[2] = W;
    params[3] = S2;
    params[4] = M;
    params[5] = m;
    gsl_integration_qagiu(&function, 0, epsabs, epsrel, limit, workspace, &result, &error);
    return result;
}

void comp_field(Tnode &node, int c, double mu){
    node.field = field_in(node, c, mu);
}

void comp_var(Tnode &node, double beta, int c, double mu){
    node.var = var_in(node, beta, c, mu);
}


void comp_W(Tnode &node, double beta, int c, double mu){
    node.W = W_in(node, beta, c, mu);
}

double new_averages(double beta, double lambda, Tnode &node, double tol_asymp, 
                    gsl_function integrand_av, gsl_function integrand_q_sqr, gsl_function integrand_w3, 
                    double epsabs, double epsrel, long limit, gsl_integration_workspace *workspace, double *params){
    double delta = 0, delta_av, delta_q_sqr, delta_w3, s, hi_div_s, hi2_div_s, den, av_new, q_sqr_new, w3_new;
    
    if (node.W > 0){
        av_new = numerator_IBMF3(beta, lambda, node.W, node.var, node.field, node.av, integrand_av, 
                                 epsabs, epsrel, limit, workspace, params);
        q_sqr_new = numerator_IBMF3(beta, lambda, node.W, node.var, node.field, node.av, integrand_q_sqr, 
                                    epsabs, epsrel, limit, workspace, params);
        w3_new = numerator_IBMF3(beta, lambda, node.W, node.var, node.field, node.av, integrand_w3, 
                                 epsabs, epsrel, limit, workspace, params);
        den = (node.W * w3_new / 2 + 
               q_sqr_new * (1.0 / node.var - node.av * node.W) - 
               av_new * (node.field + node.av * (1 - node.var) - node.W * node.av * node.av / 2)) / lambda;
        av_new /= den;
        q_sqr_new /= den;
        w3_new /= den;
    }else{
        if (node.var > 0){
            s = sqrt(node.var);
            hi_div_s = node.av * (1.0 / s - s) + s * node.field;
            hi2_div_s = hi_div_s * hi_div_s;
            if (exp(-beta * hi2_div_s / 2) / pow(hi2_div_s, beta * lambda / 2) < tol_asymp){
                hi_div_s = node.field;
                hi2_div_s = hi_div_s * hi_div_s;
                den = denominator_IBMF(beta, lambda, hi_div_s, hi2_div_s);
                av_new = numerator_av_IBMF(beta, lambda, hi_div_s, hi2_div_s) / den;
                q_sqr_new = av_new * av_new;
                w3_new = av_new * av_new * av_new;
            }else{
                den = denominator_IBMF(beta, lambda, hi_div_s, hi2_div_s);
                av_new = s * numerator_av_IBMF(beta, lambda, hi_div_s, hi2_div_s) / den; 
                q_sqr_new = node.var * numerator_q_sqr_IBMF2(beta, lambda, hi_div_s, hi2_div_s) / den;
                w3_new = av_new * av_new * av_new;
            }
        }else{
            hi_div_s = node.field;
            hi2_div_s = hi_div_s * hi_div_s;
            den = denominator_IBMF(beta, lambda, hi_div_s, hi2_div_s);
            av_new = numerator_av_IBMF(beta, lambda, hi_div_s, hi2_div_s) / den;
            q_sqr_new = av_new * av_new;
            w3_new = av_new * av_new * av_new;
        }
    }
    delta_av = fabs(av_new - node.av);
    delta_q_sqr = fabs(q_sqr_new - node.q_sqr);
    delta_w3 = fabs(w3_new - node.w3);
    if (delta_av > delta){
        delta = delta_av;
    }
    if (delta_q_sqr > delta){
        delta = delta_q_sqr;
    }
    if (delta_w3 > delta){
        delta = delta_w3;
    }
    node.av = av_new;
    node.q_sqr = q_sqr_new;
    node.w3 = w3_new;
    return delta;
}


int convergence(double beta, double lambda, int c, double mu, Tnode &node, double tol, double tol_asymp, 
                int max_iter, bool &divergence, gsl_function integrand_av, gsl_function integrand_q_sqr, 
                gsl_function integrand_w3, double epsabs, double epsrel, long limit, 
                gsl_integration_workspace *workspace, double *params){
    double delta = tol + 1;
    int iter = 0;

    comp_field(node, c, mu);
    comp_var(node, beta, c, mu);
    comp_W(node, beta, c, mu);

    while (delta > tol && iter < max_iter){
        delta = new_averages(beta, lambda, node, tol_asymp,
                             integrand_av, integrand_q_sqr, integrand_w3, 
                             epsabs, epsrel, limit, workspace, params);
        iter++;
        comp_field(node, c, mu);
        comp_var(node, beta, c, mu);
        comp_W(node, beta, c, mu);
        if (isinf(delta)){
            divergence = true;
            return iter;
        }
    }

    divergence = false;
    return iter;
}


int main(int argc, char *argv[]) {
    double avn_0 = atof(argv[1]);
    double T = atof(argv[2]);
    double lambda = atof(argv[3]);
    double tol = atof(argv[4]);
    double tol_asymp = atof(argv[5]);
    int max_iter = atoi(argv[6]);
    double mu0 = atof(argv[7]);
    double dmu = atof(argv[8]);
    double muf = atof(argv[9]);
    int c = atoi(argv[10]);
    double epsabs = atof(argv[11]);
    double epsrel = atof(argv[12]);
    long limit = atol(argv[13]);


    Tnode node;
    double beta = 1.0 / T;
    int iter;
    bool conv;
    bool divergence;

    double params[5]; // beta, lambda, W, S2, M
    gsl_function integrand_av_gsl, integrand_q_sqr_gsl, integrand_w3_gsl;
    integrand_av_gsl.function = &integrand_av;
    integrand_q_sqr_gsl.function = &integrand_q2;
    integrand_w3_gsl.function = &integrand_w3;
    integrand_av_gsl.params = params;
    integrand_q_sqr_gsl.params = params;
    integrand_w3_gsl.params = params;
    gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(limit);

    for (double mu = mu0; mu < muf + dmu / 2; mu += dmu) {
        node.av = avn_0;
        node.q_sqr = avn_0 * avn_0;
        node.w3 = avn_0 * avn_0 * avn_0;
        iter = convergence(beta, lambda, c, mu, node, tol, tol_asymp, max_iter, divergence,
                           integrand_av_gsl, integrand_q_sqr_gsl, integrand_w3_gsl,
                           epsabs, epsrel, limit, workspace, params);
        if (divergence){
            cout << mu << "\t" << iter << "\t" << "diverges" << "\t" << node.field << "\t" << node.var << "\t" << node.W << endl;
        }else{
            conv = iter < max_iter;
            cout << mu << "\t" << iter << "\t" << conv << "\t" << node.field << "\t" << node.var << "\t" << node.W << endl;
        }
    }
    
    return 0;
}