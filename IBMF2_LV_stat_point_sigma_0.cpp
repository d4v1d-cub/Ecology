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
    double field; // local field in that node
    double var; // variance of the perturbed gaussian that depends on the neighbors
    double av; // average value of n in that node
    double q_sqr; // average value of n^2 in that node
}Tnode;



double field_in(Tnode node, int c, double mu){
    return 1 - c * mu * node.av;
}


double var_in(Tnode node, double beta, int c, double mu){
    return 1.0 / (1 - beta * c * mu * mu * (node.q_sqr - node.av * node.av));
}

double numerator_av(double beta, double lambda, double hi_div_s, double hi2_div_s){
    return 2 * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi2_div_s / 2) + 
           sqrt(2 * beta) * hi_div_s * beta * lambda * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi2_div_s / 2);
}


double numerator_q_sqr(double beta, double lambda, double hi_div_s, double hi2_div_s){
    return 4 * hi_div_s * gsl_sf_gamma((3 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 1.5, -beta * hi2_div_s / 2) + 
           sqrt(2 * beta) * lambda * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1(-(1 + beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2);
}


double denominator(double beta, double lambda, double hi_div_s, double hi2_div_s){
    return sqrt(2 * beta) * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi2_div_s / 2) + 
           2 * beta * hi_div_s * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi2_div_s / 2);
}


void comp_field(Tnode &node, int c, double mu){
    node.field = field_in(node, c, mu);
}

void comp_var(Tnode &node, double beta, int c, double mu){
    node.var = var_in(node, beta, c, mu);
}


double new_averages(double beta, double lambda, Tnode &node, double tol_asymp){
    double delta = 0, delta_av, delta_q_sqr, s, hi_div_s, hi2_div_s, den, av_new, q_sqr_new;

    if (node.var > 0){
        s = sqrt(node.var);
        hi_div_s = node.av * (1.0 / s - s) + s * node.field;
        hi2_div_s = hi_div_s * hi_div_s;
        if (exp(-beta * hi2_div_s / 2) / pow(hi2_div_s, beta * lambda / 2) < tol_asymp){
            hi_div_s = node.field;
            hi2_div_s = hi_div_s * hi_div_s;
            den = denominator(beta, lambda, hi_div_s, hi2_div_s);
            av_new = numerator_av(beta, lambda, hi_div_s, hi2_div_s) / den;
            q_sqr_new = av_new * av_new;
        }else{
            den = denominator(beta, lambda, hi_div_s, hi2_div_s);
            av_new = s * numerator_av(beta, lambda, hi_div_s, hi2_div_s) / den; 
            q_sqr_new = node.var * numerator_q_sqr(beta, lambda, hi_div_s, hi2_div_s) / den;
        }
    }else{
        hi_div_s = node.field;
        hi2_div_s = hi_div_s * hi_div_s;
        den = denominator(beta, lambda, hi_div_s, hi2_div_s);
        av_new = numerator_av(beta, lambda, hi_div_s, hi2_div_s) / den;
        q_sqr_new = av_new * av_new;
    }
        
    delta_av = fabs(av_new - node.av);
    delta_q_sqr = fabs(q_sqr_new - node.q_sqr);
    if (delta_av > delta){
        delta = delta_av;
    }
    if (delta_q_sqr > delta){
        delta = delta_q_sqr;
    }
    node.av = av_new;
    node.q_sqr = q_sqr_new;

    return delta;
}


int convergence(double beta, double lambda, int c, double mu, Tnode &node, double tol, double tol_asymp, 
                int max_iter, bool &divergence){
    double delta = tol + 1;
    int iter = 0;

    comp_field(node, c, mu);
    comp_var(node, beta, c, mu);

    while (delta > tol && iter < max_iter){
        delta = new_averages(beta, lambda, node, tol_asymp);
        iter++;
        comp_field(node, c, mu);
        comp_var(node, beta, c, mu);
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


    Tnode node;
    double beta = 1.0 / T;
    int iter;
    bool conv;
    bool divergence;

    for (double mu = mu0; mu < muf + dmu / 2; mu += dmu) {
        node.av = avn_0;
        node.q_sqr = avn_0 * avn_0;
        iter = convergence(beta, lambda, c, mu, node, tol, tol_asymp, max_iter, divergence);
        if (divergence){
            cout << mu << "\t" << iter << "\t" << "diverges" << "\t" << node.field << "\t" << node.var << endl;
        }else{
            conv = iter < max_iter;
            cout << mu << "\t" << iter << "\t" << conv << "\t" << node.field << "\t" << node.var << endl;
        }
    }
    
    return 0;
}