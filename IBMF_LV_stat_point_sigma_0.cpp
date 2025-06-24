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
            sqrt(beta / 2) * hi * beta * lambda * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2);
}


double denominator(double beta, double lambda, double hi, double normfactor = 1e-14){
    return sqrt(beta / 2) * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2) + 
            beta * hi * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2)
            + normfactor;
}


double numerator_asymp(double beta, double lambda, double hi){
    return sqrt(lambda) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2) +
           beta * lambda * hi * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2);
}

double denominator_asymp(double beta, double lambda, double hi, double normfactor = 1e-14){
    return gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2) + 
           beta * sqrt(lambda) * hi * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2) + 
           normfactor;
}


int check_wich_diverges(double beta, double lambda, double hi, double limit = 1e+10){
    if (isnan(gsl_sf_gamma(1 + beta * lambda / 2)) || isinf(gsl_sf_gamma(1 + beta * lambda / 2)) || gsl_sf_gamma(1 + beta * lambda / 2) > limit){
        if (isnan(gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2)) 
            || gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        }else if(isnan(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2))
                 || gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        }else if(isnan(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2))
                 || gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        }else if(isnan(gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2)) || 
                 gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2) > limit){
            return 1; // gamma and hypergeometric diverge
        }else{
            return 2; // only gamma diverges
        }
    }else if (isnan(gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2)) 
            || gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta * hi * hi / 2) > limit){
        return 1; // gamma and hypergeometric diverge
    }else if(isnan(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2))
                 || gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta * hi * hi / 2) > limit){
        return 1; // gamma and hypergeometric diverge
    }else if(isnan(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2))
                 || gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta * hi * hi / 2) > limit){
        return 1; // gamma and hypergeometric diverge
    }else if(isnan(gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2)) || isinf(gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2)) || 
                 gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta * hi * hi / 2) > limit){
        return 1; // hypergeometric diverges
    }else{
        return 0; // cannot identify divergence
    }
}


double new_average(double avg, double beta, double lambda, double mu, int c, double normfactor = 1e-14){
    double field = field_in(avg, c, mu);
    int identify_divergence = 0;
    double avg_new = numerator(beta, lambda, field) / denominator(beta, lambda, field, normfactor);
    if (isnan(avg_new) || isinf(avg_new)){
        identify_divergence = check_wich_diverges(beta, lambda, field);
        if (identify_divergence == 1){
            if (field < 0){
                avg_new = 0;
            }else{
                avg_new = field;
            }
        }else if (identify_divergence == 2){
            avg_new = numerator_asymp(beta, lambda, field) / denominator_asymp(beta, lambda, field, normfactor);
        }else{
            cout << "Cannot identify divergence" << endl;
            exit(1);
        }
    }

    return avg_new;
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

    gsl_set_error_handler_off();

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