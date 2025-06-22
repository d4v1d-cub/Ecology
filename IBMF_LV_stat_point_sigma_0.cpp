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


double numerator(double beta, double lambda, double hi){
    return gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2) +
            sqrt(2 * beta) * hi * gsl_sf_gamma(1 + beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2);
}


double denominator(double beta, double lambda, double hi, double normfactor = 1e-10){
    return sqrt(beta / 2) * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2) + 
            beta * hi * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2)
            + normfactor;
}


double numerator_assymp(double beta, double lambda, double hi){
    return sqrt(lambda) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2) +
           beta * lambda * hi * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2);
}

double denominator_assymp(double beta, double lambda, double hi, double normfactor = 1e-10){
    return gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2) + 
           beta * sqrt(lambda) * hi * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2) + 
           normfactor;
}

double new_average(double avg, double beta, double lambda, double mu, int c, 
                   double normfactor = 1e-14, double asympthres_1 = 1e-7, double asympthres_2 = 1e-3){
    double field = field_in(avg, c, mu);
    if (exp(-beta * field * field / 2) < asympthres_1){
        if (field < lambda){
            return lambda;
        }else{
            return field;
        }
    }else if(1.0 / lambda / beta < asympthres_2){
        return numerator_assymp(beta, lambda, field) / denominator_assymp(beta, lambda, field, normfactor);
    }else{
        return numerator(beta, lambda, field) / denominator(beta, lambda, field, normfactor);
    }
}


int convergence(double &avg, double beta, double lambda, double mu, int c, double tol, 
                int max_iter, bool &divergence){
    double avg_new;
    double var = tol + 1;
    int iter = 0;

    while (var > tol && iter < max_iter){
        avg_new = new_average(avg, beta, lambda, mu, c);
        var = fabs(avg_new - avg);
        iter++;
        avg = avg_new;
        if (isinf(var)){
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
    int max_iter = atoi(argv[5]);
    double mu0 = atof(argv[6]);
    double dmu = atof(argv[7]);
    double muf = atof(argv[8]);

    int c = atoi(argv[9]);

    double avg, field;

    double beta = 1.0 / T;
    int iter;
    bool conv;
    bool divergence;
    for (double mu = mu0; mu < muf + dmu / 2; mu += dmu) {
        avg = avn_0;
        iter = convergence(avg, beta, lambda, mu, c, tol, max_iter, divergence);
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