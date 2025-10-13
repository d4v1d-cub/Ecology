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


double field_in(double avg, int c, double mu){
    return 1 - mu * c * avg;
}


bool comp_coefficients(double beta, double lambda, double **&coefficients, double *&gamma_vals, 
                       double maximum=1e10){
    bool gamma_diverges = false;
    gamma_vals = new double[2];
    if (isnan(gsl_sf_gamma((1 + beta * lambda) / 2)) || isinf(gsl_sf_gamma((1 + beta * lambda) / 2)) || 
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
    
    while (!(isnan(num) || isinf(num) || isnan(den) || isinf(den) || 
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
        if (isnan(num) || isinf(num) || isnan(den) || isinf(den) || 
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

double new_average(double avg, double beta, double lambda, double mu, int c, 
                   double hmin, double hmax, double **coefficients, double *gamma_vals, 
                   int iter, double damping, double normfactor = 1e-14){
    double field = field_in(avg, c, mu);
    double av_new;
    if (field > hmax){
        av_new = damping * field * (1 - 1.0 / beta / field / field + lambda / field / field) + 
                 (1 - damping) * avg;                      
    }else if (field < hmin)
    {
        av_new = damping * lambda / fabs(field) + (1 - damping) * avg;
    }else if (field == 0){
        av_new = damping * sqrt(2.0 / beta) * gamma_vals[1] / gamma_vals[0] + (1 - damping) * avg;
    }else {
        av_new = damping * numerator_av(beta, lambda, field, coefficients[1]) /
                 denominator(beta, lambda, field, coefficients[0], normfactor) +
                 (1 - damping) * avg;
    }

    if (isnan(av_new) || isinf(av_new)){
        cerr << "Error: av_new is nan or inf at iter=" << iter << endl;   
    }
        
    return av_new;
}


int convergence(double &avg, double beta, double lambda, double mu, int c, double tol, 
                int max_iter, bool &divergence, double hmin, double hmax, 
                double **coefficients, double *gamma_vals, double damping, 
                double maximum=1e10, int min_consecutive=5){
    double avg_new;
    double var = tol + 1;
    int iter = 0;

    int consecutive = 0;
    while (consecutive < min_consecutive && iter < max_iter){
        avg_new = new_average(avg, beta, lambda, mu, c, hmin, hmax, coefficients, 
                              gamma_vals, iter, damping);
        var = fabs(avg_new - avg);
        iter++;
        avg = avg_new;
        if (isinf(var) || isnan(var) || var > maximum){
            divergence = true;
            return iter;
        }
        if (var < tol){
            consecutive++;
        }else{
            consecutive = 0;
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
    int max_iter = atoi(argv[5]);
    double mu0 = atof(argv[6]);
    double dmu = atof(argv[7]);
    double muf = atof(argv[8]);

    int c = atoi(argv[9]);

    double damping = atof(argv[10]);

    gsl_set_error_handler_off();

    double avg, field;

    double beta = 1.0 / T;
    int iter;
    bool conv;
    bool divergence;

    double hmax = find_divergence_max(beta, (1 + beta * lambda) / 2);
    double **coefficients, *gamma_vals;
    comp_coefficients(beta, lambda, coefficients, gamma_vals);
    double hmin = find_divergence_min(beta, lambda, coefficients);

    avg = avn_0;
    for (double mu = mu0; mu < muf + dmu / 2; mu += dmu) {
        iter = convergence(avg, beta, lambda, mu, c, tol, max_iter, divergence, 
                           hmin, hmax, coefficients, gamma_vals, damping);
        field = field_in(avg, c, mu);
        if (divergence){
            cout << mu << "\t" << iter << "\t" << "diverges" << "\t" << avg << "\t" << field << "\t" << endl;
        }else{
            conv = iter < max_iter;
            cout << mu << "\t" << iter << "\t" << conv << "\t" << avg << "\t" << field << "\t" << endl;
        }
    }
    
    return 0;
}