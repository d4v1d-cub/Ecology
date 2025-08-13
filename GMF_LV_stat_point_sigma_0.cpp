#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include "math.h"
#include <cmath>

using namespace std;


typedef struct{
    double field_cav; // local field in that node, given that one neighbor is zero
    double var_cav; // variance of the perturbed gaussian that depends on the neighbors, given that one neighbor is zero
    double av_cav; // average value of n in that node, given that one neighbor is zero
    double chi_cav; // beta * (q_sqr_cav - av_cav^2)
    bool chi_cav_converged; // whether the average value of n^2 has converged
    double av; // average value of n in that node
    double var; // variance of the perturbed gaussian that depends on the neighbors
    double field; // field in that node
    double chi; // beta * (q_sqr - av^2)
}Tnode;



double field_in(Tnode node, int c, double mu){
    return 1 - (c - 1) * mu * node.av_cav;
}


double var_in(Tnode node, int c, double mu){
    return 1.0 / (1 - (c - 1) * mu * mu * node.chi_cav);
}


bool comp_coefficients(double beta, double lambda, double **&coefficients){
    bool gamma_diverges = false;
    if (isnan(gsl_sf_gamma((1 + beta * lambda) / 2)) || isinf(gsl_sf_gamma((1 + beta * lambda) / 2))){
        gamma_diverges = true;
    }

    coefficients = new double *[3];
    for (int i = 0; i < 3; i++){
        coefficients[i] = new double[2];
    }

    if (gamma_diverges){
        coefficients[0][0] = 1;
        coefficients[0][1] = beta * sqrt(lambda) * (1 - 1.0 / 4 / beta / lambda);

        coefficients[1][0] = sqrt(lambda) * (1 - 1.0 / 4 / beta / lambda);
        coefficients[1][1] = lambda * beta;

        coefficients[2][0] = lambda;
        coefficients[2][1] = lambda * beta * sqrt(lambda) * (1 + 3.0 / 4 / beta / lambda);

    }else{
        double gammabl2 = gsl_sf_gamma(beta * lambda / 2);
        double gammabl12 = gsl_sf_gamma((1 + beta * lambda) / 2);
        
        coefficients[0][0] = sqrt(beta / 2) * gammabl2;
        coefficients[0][1] = beta * gammabl12;

        coefficients[1][0] = gammabl12;
        coefficients[1][1] = sqrt(beta / 2) * beta * lambda * gammabl2;

        coefficients[2][0] = sqrt(beta / 2) * lambda * gammabl2;
        coefficients[2][1] = (1 + beta * lambda) * gammabl12;
    }

    return gamma_diverges;
}


double find_divergence(double beta, double alpha, double hmax=100, double precision=1e-4, double maximum=1e10){
    double val1, val2;
    val1 = gsl_sf_hyperg_1F1(alpha, 0.5, beta * hmax * hmax / 2);
    val2 = gsl_sf_hyperg_1F1(alpha + 0.5, 1.5, beta * hmax * hmax / 2);
    while (!(isnan(val1) || isinf(val1) || isnan(val2) || isinf(val2) || 
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
        if (isnan(val1) || isinf(val1) || isnan(val2) || isinf(val2) || 
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

double numerator_av(double beta, double lambda, double hi_div_Q, double *coefficients){
    return coefficients[0] * gsl_sf_hyperg_1F1((1 + beta * lambda) / 2, 0.5, beta * hi_div_Q * hi_div_Q / 2) +
           coefficients[1] * hi_div_Q * gsl_sf_hyperg_1F1(1 + beta * lambda / 2, 1.5, beta * hi_div_Q * hi_div_Q / 2);
}


double numerator_q_sqr(double beta, double lambda, double hi_div_Q, double *coefficients){
    return coefficients[0] * gsl_sf_hyperg_1F1(1 + beta * lambda / 2, 0.5, beta * hi_div_Q * hi_div_Q / 2) + 
           coefficients[1] * hi_div_Q * gsl_sf_hyperg_1F1((3 + beta * lambda) / 2, 1.5, beta * hi_div_Q * hi_div_Q / 2);
}


double denominator(double beta, double lambda, double hi_div_Q, double *coefficients, double normfactor = 1e-14){
    return coefficients[0] * gsl_sf_hyperg_1F1(beta * lambda / 2, 0.5, beta * hi_div_Q * hi_div_Q / 2) + 
           coefficients[1] * hi_div_Q * gsl_sf_hyperg_1F1((1 + beta * lambda) / 2, 1.5, beta * hi_div_Q * hi_div_Q / 2)
           + normfactor;
}


void comp_field(Tnode &node, int c, double mu){
    node.field_cav = field_in(node, c, mu);
}

void comp_var(Tnode &node, int c, double mu){
    node.var_cav = var_in(node, c, mu);
}


double new_averages(double beta, double lambda, Tnode &node, 
                    double hmax, double **coefficients, int iter, double tol, 
                    double normfactor = 1e-14){
    double Q, hi, hi_div_Q, den, av_new, q_sqr_new, chi_cav_new;

    if (node.chi_cav_converged){
        chi_cav_new = node.chi_cav;
        if (node.var_cav > 0){
            Q = sqrt(node.var_cav);
            hi = node.field_cav * node.var_cav;
            hi_div_Q = hi / Q;
            if (hi_div_Q > hmax){
                av_new = hi * (1 - 1.0 / beta / hi_div_Q / hi_div_Q + lambda / hi_div_Q / hi_div_Q);
            }else if (hi_div_Q < 0){
                av_new = 0;
            }else{
                den = denominator(beta, lambda, hi_div_Q, coefficients[0], normfactor);
                av_new = Q * numerator_av(beta, lambda, hi_div_Q, coefficients[1]) / den;
            }
        }
    }else if (node.var_cav > 0){
        Q = sqrt(node.var_cav);
        hi = node.field_cav * node.var_cav;
        hi_div_Q = hi / Q;
        if (hi_div_Q > hmax){
            av_new = hi * (1 - 1.0 / beta / hi_div_Q / hi_div_Q + lambda / hi_div_Q / hi_div_Q);
            chi_cav_new = node.var_cav;
        }else if (hi_div_Q < 0){
            av_new = 0;
            chi_cav_new = node.var_cav;
        }else{
            den = denominator(beta, lambda, hi_div_Q, coefficients[0], normfactor);
            av_new = Q * numerator_av(beta, lambda, hi_div_Q, coefficients[1]) / den;
            q_sqr_new = node.var_cav * numerator_q_sqr(beta, lambda, hi_div_Q, coefficients[2]) / den;
            chi_cav_new = beta * (q_sqr_new - av_new * av_new);
        }
    }else{
        hi = node.field_cav;
        if (hi > hmax){
            av_new = hi * (1 - 1.0 / beta / hi / hi + lambda / hi / hi);
        }else if (hi < 0){
            av_new = 0;
        }else{
            den = denominator(beta, lambda, hi, coefficients[0], normfactor);
            av_new = numerator_av(beta, lambda, hi, coefficients[1]) / den;
        }
            
        chi_cav_new = 0;
    }
        

    if (isnan(av_new) || isinf(av_new) || isnan(chi_cav_new) || isinf(chi_cav_new)){
        cerr << "Error: averages are nan or inf at iter=" << iter << endl;
        return sqrt(-1);
    }

    double delta_av = fabs(av_new - node.av_cav);
    double delta_chi_cav;
    if (node.var_cav > 0){
        delta_chi_cav = fabs(chi_cav_new - node.chi_cav);
    }else{
        delta_chi_cav = 1; // If the variance is zero, we consider the chi_cav to be converged
    }


    if (!node.chi_cav_converged && delta_chi_cav < tol){
        node.chi_cav_converged = true;
    }

    node.av_cav = av_new;
    node.chi_cav = chi_cav_new;

    return max(delta_av, delta_chi_cav);
}


int convergence(double beta, double lambda, int c, double mu, Tnode &node, double tol, int max_iter, 
                bool &divergence, double hmax, double **coefficients, double maximum=1e10){
    double delta = tol + 1;
    int iter = 0;

    comp_field(node, c, mu);
    comp_var(node, c, mu);

    node.chi_cav_converged = false;

    while (delta > tol && iter < max_iter){
        delta = new_averages(beta, lambda, node, hmax, coefficients, iter, tol);
        iter++;
        comp_field(node, c, mu);
        comp_var(node, c, mu);
        if (isinf(delta) || isnan(delta) || delta > maximum){
            divergence = true;
            return iter;
        }
    }

    divergence = false;
    return iter;
}


void averages_node(Tnode &node, double beta, double lambda, int c, double mu, 
                     double hmax, double **coefficients, double normfactor = 1e-14){
    double av = 0;
    double h, h_div_Q, Q, den, q_sqr_new;
    node.field = 1 - c * mu * node.av_cav;
    node.var = 1.0 / (1 - c * mu * mu * node.chi_cav);
    if (node.var > 0){
        h = node.field * node.var;
        Q = sqrt(node.var);
        h_div_Q = h / Q;

        if (h_div_Q > hmax){
            node.av = h * (1 - 1.0 / beta / h_div_Q / h_div_Q + lambda / h_div_Q / h_div_Q);
            node.chi = node.var;
        } else if (h_div_Q < 0){
            node.av = 0;
            node.chi = node.var;
        } else{
            den = denominator(beta, lambda, h_div_Q, coefficients[0], normfactor);
            node.av = Q * numerator_av(beta, lambda, h_div_Q, coefficients[1]) / den;
            q_sqr_new = node.var * numerator_q_sqr(beta, lambda, h_div_Q, coefficients[2]) / den;
            node.chi = beta * (q_sqr_new - node.av * node.av);
        }
    }else{
        h = node.field;
        if (h > hmax){
            node.av = h * (1 - 1.0 / beta / h / h + lambda / h / h);
        }else if (h < 0){
            node.av = 0;
        }else{
            den = denominator(beta, lambda, h, coefficients[0], normfactor);
            node.av = numerator_av(beta, lambda, h, coefficients[1]) / den;
        }
        node.chi = node.var;
    }
}


int main(int argc, char *argv[]) {
    double avn_0 = atof(argv[1]);
    double T = atof(argv[2]);
    double lambda = atof(argv[3]);
    double tol = atof(argv[4]);
    int max_iter = atoi(argv[5]);
    double mu0 = atof(argv[6]);
    double dmu = atof(argv[7]);
    double muf = atof(argv[8]);
    int c = atoi(argv[9]);

    gsl_set_error_handler_off();

    Tnode node;
    double beta = 1.0 / T;
    int iter;
    bool conv;
    bool divergence;

    double hmax = find_divergence(beta, 1 + beta * lambda / 2);
    double **coefficients;
    comp_coefficients(beta, lambda, coefficients);

    node.av_cav = avn_0;
    node.chi_cav = 0;
    for (double mu = mu0; mu < muf + dmu / 2; mu += dmu) {
        // node.av_cav = avn_0;
        // node.chi_cav = 0;
        iter = convergence(beta, lambda, c, mu, node, tol, max_iter, divergence, 
                           hmax, coefficients);
        averages_node(node, beta, lambda, c, mu, hmax, coefficients);
        if (divergence){
            cout << mu << "\t" << iter << "\t" << "diverges" << "\t" << node.chi_cav_converged << "\t" << node.av_cav << "\t" 
                 << node.chi_cav << "\t" << node.av << "\t" << node.chi << endl;
        }else{
            conv = iter < max_iter;
            cout << mu << "\t" << iter << "\t" << conv << "\t" << node.chi_cav_converged << "\t" << node.av_cav << "\t" 
                 << node.chi_cav << "\t" << node.av << "\t" << node.chi << endl;
        }
    }
    
    return 0;
}