#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include "math.h"

using namespace std;


double field_in(double avg, int c, double mu){
    return 1 - mu * c * avg;
}


double numerator(double beta, double lambda, double hi){
        return gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2) +
     sqrt(2 * beta) * hi * gsl_sf_gamma(1 + beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2);
}


double denominator(double beta, double lambda, double hi){
    return sqrt(beta / 2) * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2) + 
    beta * hi * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2);
}


double new_average(double avg, double beta, double lambda, double mu, int c){
    double field = field_in(avg, c, mu);
    return numerator(beta, lambda, field) / denominator(beta, lambda, field);
}


int convergence(double &avg, double beta, double lambda, double mu, int c, double tol, int max_iter){
    double avg_new;
    double var = tol + 1;
    int iter = 0;

    while (var > tol && iter < max_iter){
        avg_new = new_average(avg, beta, lambda, mu, c);
        var = fabs(avg_new - avg);
        iter++;
        avg = avg_new;
    }

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

    double avg, field;

    double beta = 1.0 / T;
    int iter;
    bool conv;
    for (double mu = mu0; mu < muf + dmu / 2; mu += dmu) {
        avg = avn_0;
        iter = convergence(avg, beta, lambda, mu, c, tol, max_iter);
        conv = iter < max_iter;
        field = field_in(avg, c, mu);
        cout << mu << "\t" << iter << "\t" << conv << "\t" << avg << "\t" << field << "\t" << endl;
    }
    
    return 0;
}