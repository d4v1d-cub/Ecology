#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include "math.h"

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

double numerator_av(double beta, double lambda, double hi, double hi2){
    return 2 * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi2 / 2) + 
           sqrt(2 * beta) * hi * beta * lambda * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi2 / 2);
}


double numerator_q_sqr(double beta, double lambda, double hi, double hi2){
    return 4 * hi * gsl_sf_gamma((3 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 1.5, -beta * hi2 / 2) + 
           sqrt(2 * beta) * lambda * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1(-(1 + beta * lambda) / 2, 0.5, -beta * hi2 / 2);
}


double denominator(double beta, double lambda, double hi, double hi2){
    return sqrt(2 * beta) * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi2 / 2) + 
           2 * beta * hi * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi2 / 2);
}


void comp_field(Tnode &node, int c, double mu){
    node.field = field_in(node, c, mu);
}

void comp_var(Tnode &node, double beta, int c, double mu){
    node.var = var_in(node, beta, c, mu);
}


double new_averages(double beta, double lambda, int c, double mu, Tnode &node, double tol_asymp){
    double delta = 0, delta_av, delta_q_sqr, s, hi, hi2, den, av_new, q_sqr_new;

    if (node.var > 0){
        hi2 = node.var * node.field * node.field;
        if (exp(-beta * hi2 / 2) / pow(hi2, beta * lambda / 2) < tol_asymp){
            hi = node.field;
            hi2 = hi * hi;
                den = denominator(beta, lambda, hi, hi2);
            av_new = numerator_av(beta, lambda, hi, hi2) / den;
            q_sqr_new = av_new * av_new;
        }else{
            s = sqrt(node.var);
            hi = s * node.field;
            den = denominator(beta, lambda, hi, hi2);
            av_new = s * numerator_av(beta, lambda, hi, hi2) / den; 
            q_sqr_new = node.var * numerator_q_sqr(beta, lambda, hi, hi2) / den;
        }
    }else{
        hi = node.field;
        hi2 = hi * hi;
        den = denominator(beta, lambda, hi, hi2);
        av_new = numerator_av(beta, lambda, hi, hi2) / den;
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


double new_averages(long N, double beta, double lambda, Tnode *nodes, double tol_asymp){
    double delta = 0, delta_av, delta_q_sqr, s, hi, hi2, den, av_new, q_sqr_new;
    
    for (long i = 0; i < N; i++){
        if (nodes[i].var > 0){
            hi2 = nodes[i].var * nodes[i].field * nodes[i].field;
            if (exp(-beta * hi2 / 2) / pow(hi2, beta * lambda / 2) < tol_asymp){
                hi = nodes[i].field;
                hi2 = hi * hi;
                den = denominator(beta, lambda, hi, hi2);
                av_new = numerator_av(beta, lambda, hi, hi2) / den;
                q_sqr_new = av_new * av_new;
            }else{
                s = sqrt(nodes[i].var);
                hi = s * nodes[i].field;
                den = denominator(beta, lambda, hi, hi2);
                av_new = s * numerator_av(beta, lambda, hi, hi2) / den; 
                q_sqr_new = nodes[i].var * numerator_q_sqr(beta, lambda, hi, hi2) / den;
            }
        }else{
            hi = nodes[i].field;
            hi2 = hi * hi;
            den = denominator(beta, lambda, hi, hi2);
            av_new = numerator_av(beta, lambda, hi, hi2) / den;
            q_sqr_new = av_new * av_new;
        }
        
        delta_av = fabs(av_new - nodes[i].av);
        delta_q_sqr = fabs(q_sqr_new - nodes[i].q_sqr);
        if (delta_av > delta){
            delta = delta_av;
        }
        if (delta_q_sqr > delta){
            delta = delta_q_sqr;
        }
        nodes[i].av = av_new;
        nodes[i].q_sqr = q_sqr_new;

    }
    return delta;
}


int convergence(double beta, double lambda, int c, double mu, Tnode &node, double tol, double tol_asymp, 
                int max_iter){
    double delta = tol + 1;
    int iter = 0;

    comp_field(node, c, mu);
    comp_var(node, beta, c, mu);

    while (delta > tol && iter < max_iter){
        delta = new_averages(beta, lambda, c, mu, node, tol_asymp);
        iter++;
        comp_field(node, c, mu);
        comp_var(node, beta, c, mu);
    }

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
    

    for (double mu = mu0; mu < muf + dmu / 2; mu += dmu) {
        node.av = avn_0;
        node.q_sqr = avn_0 * avn_0;
        iter = convergence(beta, lambda, c, mu, node, tol, tol_asymp, max_iter);
        conv = iter < max_iter;
        cout << mu << "\t" << iter << "\t" << conv << "\t" << node.field << "\t" << node.var << endl;
    }
    
    return 0;
}