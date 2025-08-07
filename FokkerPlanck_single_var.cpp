#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include "math.h"
#include <cmath>

using namespace std;

double *n_grid, temp, lambda;
int npoints;
double *P_der_n, *P_der2_n;

void init_arrays(double dn){
    for (int i = 0; i < npoints; i++){
        n_grid[i] = i * dn;
        P_der_n[i] = 0.0;
        P_der2_n[i] = 0.0;
    }
}


void initial_condition(double *&P, double n0, double sigma0){
    P = new double[npoints];
    for (int i = 0; i < npoints; i++){
        P[i] = exp(-(n_grid[i] - n0) * (n_grid[i] - n0) / 2 / sigma0 / sigma0) / 
               (sqrt(M_PI / 2) * sigma0 * (1 + erf(n0 / sqrt(2) / sigma0)));
    }
}


void compute_der(double *P){
    for (int i = 0; i < npoints - 1; i++){
        P_der_n[i] = (P[i + 1] - P[i]) / (n_grid[i + 1] - n_grid[i]);
    }
    P_der_n[npoints - 1] = (P[npoints - 1] - P[npoints - 2]) / (n_grid[npoints - 1] - n_grid[npoints - 2]);
}


void compute_der2(){
    for (int i = 0; i < npoints - 1; i++){
        P_der2_n[i] = (P_der_n[i + 1] - P_der_n[i]) / (n_grid[i + 1] - n_grid[i]);
    }
    P_der2_n[npoints - 1] = (P_der_n[npoints - 1] - P_der_n[npoints - 2]) / (n_grid[npoints - 1] - n_grid[npoints - 2]);
}


int P_der_t(double t, const double P[], double der_t[], void *params){
    (void)(t); /* avoid unused parameter warning */
    for (int i = 0; i < npoints; i++){
        der_t[i] = -(1 - 2 * n_grid[i]) * P[i] - 
                   (n_grid[i] * (1 - n_grid[i]) + lambda - 2 * temp) * P_der_n[i] + 
                    temp * n_grid[i] * P_der2_n[i];
    }
    der_t[0] += (temp - lambda) * P[0];
    return GSL_SUCCESS;
}


int jacobian(double t, const double P[], double *dfdy, double dfdt[], void *params)
{
    gsl_matrix_view dfdy_mat
        = gsl_matrix_view_array (dfdy, npoints, npoints);
    gsl_matrix * m = &dfdy_mat.matrix;

    for (int i = 0; i < npoints - 2; i++){
        for (int j = 0; j < i; j++){
            gsl_matrix_set (m, i, j, 0.0);
        }

        gsl_matrix_set(m, i, i, 
            -(1 - 2 * n_grid[i]) +
            (n_grid[i] * (1 - n_grid[i]) + lambda - 2 * temp) / (n_grid[i + 1] - n_grid[i]) +
            temp * n_grid[i] / (n_grid[i + 1] - n_grid[i]) / (n_grid[i + 1] - n_grid[i]));
        gsl_matrix_set(m, i, i + 1, 
            -(n_grid[i] * (1 - n_grid[i]) + lambda - 2 * temp) / (n_grid[i + 1] - n_grid[i]) -
            temp * n_grid[i] / (n_grid[i + 1] - n_grid[i]) *
            (1.0 / (n_grid[i + 2] - n_grid[i + 1]) + 1.0 / (n_grid[i + 1] - n_grid[i])));
        gsl_matrix_set(m, i, i + 2, 
            temp * n_grid[i] / (n_grid[i + 1] - n_grid[i]) / (n_grid[i + 2] - n_grid[i + 1]));

        for (int j = i + 3; j < npoints; j++){
            gsl_matrix_set (m, i, j, 0.0);
        }
    }

    int pos = npoints - 2;
    gsl_matrix_set(m, pos, pos, 
        -(1 - 2 * n_grid[pos]) +
        (n_grid[pos] * (1 - n_grid[pos]) + lambda - 2 * temp) / (n_grid[pos + 1] - n_grid[pos]));
    gsl_matrix_set(m, pos, pos + 1,
        -(n_grid[pos] * (1 - n_grid[pos]) + lambda - 2 * temp) / (n_grid[pos + 1] - n_grid[pos]));

    pos = npoints - 1;
    gsl_matrix_set(m, pos, pos - 1, 
        (n_grid[pos] * (1 - n_grid[pos]) + lambda - 2 * temp) / (n_grid[pos] - n_grid[pos - 1]));
    gsl_matrix_set(m, pos, pos, 
        -(1 - 2 * n_grid[pos]) -
        (n_grid[pos] * (1 - n_grid[pos]) + lambda - 2 * temp) / (n_grid[pos] - n_grid[pos - 1]));

    gsl_matrix_set(m, 0, 0, gsl_matrix_get(m, 0, 0) + (temp - lambda));

    for (int i = 0; i < npoints; i++){
        dfdt[i] = 0.0;
    }
    
    return GSL_SUCCESS;
}


void fill_gaussian_filter(double *filter, int filter_size, int half_size, double sigma){
    double sum = 0.0;
    for (int i = 0; i < filter_size; i++){
        double x = (i - half_size) / sigma;
        filter[i] = exp(-0.5 * x * x);
        sum += filter[i];
    }
    for (int i = 0; i < filter_size; i++){
        filter[i] /= sum; // Normalize the filter
    }
}


void apply_gaussian_filter(double *P, double *filter, int filter_size, int half_size){
    double sum;
    for (int i = 0; i < half_size; i++){
        sum = 0;
        for (int j = 0; j < half_size - i; j++){
            sum += P[0] * filter[j];
        }
        for (int j = half_size - i; j < filter_size; j++){
            sum += P[i - half_size + j] * filter[j];
        }
        P[i] = sum;
    }

    for (int i = half_size; i < npoints - half_size; i++){
        sum = 0;
        for (int j = 0; j < filter_size; j++){
            sum += P[i - half_size + j] * filter[j];
        }
        P[i] = sum;
    }

    for (int i = npoints - half_size; i < npoints; i++){
        sum = 0;
        for (int j = 0; j < half_size + npoints - i; j++){
            sum += P[i - half_size + j] * filter[j];
        }
        for (int j = half_size + npoints - i; j < filter_size; j++){
            sum += P[npoints - 1] * filter[j];
        }
        P[i] = sum;
    }
}


void averages(double *P, double &nav, double &n2av){
    nav = 0.0;
    n2av = 0.0;
    for (int i = 0; i < npoints - 1; i++){
        nav += (n_grid[i + 1] - n_grid[i]) * (n_grid[i] * P[i] + n_grid[i + 1] * P[i + 1]) / 2;
        n2av += (n_grid[i + 1] - n_grid[i]) * (n_grid[i] * n_grid[i] * P[i] + 
                                               n_grid[i + 1] * n_grid[i + 1] * P[i + 1]) / 2;
    }
}

// Main function to set up the ODE solver and integrate the equations
void integrate(double *P, double t0, double tmax, double dt0, double abstol, double reltol, 
               double tsample, double n0, double sigma0, char *fileP, long maxiter, 
               int filter_size, double sigma_filter) {

    double filter[filter_size];
    int half_size = filter_size / 2;
    fill_gaussian_filter(filter, filter_size, half_size, sigma_filter);
    
    gsl_odeiv2_system sys;
    sys.function = &P_der_t;
    sys.jacobian = &jacobian;
    sys.dimension = npoints;
    sys.params = NULL;

    const gsl_odeiv2_step_type * Type = gsl_odeiv2_step_msbdf;

    gsl_odeiv2_driver * driver = gsl_odeiv2_driver_alloc_y_new (&sys, Type, dt0, abstol, reltol);
    gsl_odeiv2_driver_set_nmax(driver, maxiter);
    

    int status;
    double t = t0;

    ofstream fP(fileP);

    fP << "#params: temp  lambda  n0  sigma0  nmax  npoints  tmax  abstol  reltol  dt0 tsample\n";
    fP << "#\t" << temp << "\t" << lambda << "\t" << n0 << "\t" << sigma0 << "\t" 
       << n_grid[npoints - 1] << "\t" << npoints << "\t" << tmax << "\t" << abstol 
       << "\t" << reltol << "\t" << dt0 << "\t" << tsample << endl;
    fP << "#time";
    for (int i = 0; i < npoints; i++){
        fP << "\t" << n_grid[i];
    }
    fP << endl;
    fP << t;
    for (int i = 0; i < npoints; i++){
        fP << "\t" << P[i];
    }
    fP << endl;

    cout << "# params: temp  lambda  n0  sigma0  nmax  npoints  tmax  abstol  reltol  dt0 tsample" << endl;
    cout << "#\t" << temp << "\t" << lambda << "\t" << n0 << "\t" << sigma0 << "\t" 
         << n_grid[npoints - 1] << "\t" << npoints << "\t" << tmax << "\t" 
         << abstol << "\t" << reltol << "\t" << dt0 << "\t" << tsample << endl;

    double nav, n2av;
    averages(P, nav, n2av);

    cout << "#time\t<n>\t<n^2>" << endl;
    cout << t << "\t" << nav << "\t" << n2av << endl;
    long iter = 1;

    while (t < tmax){
        compute_der(P);
        compute_der2();
        status = gsl_odeiv2_evolve_apply(driver->e, driver->c, driver->s, &sys, &t, tmax, 
                                         &dt0, P);

        for (int i = 0; i < npoints; i++){
            if (P[i] < 0) {
                P[i] = 0.0;
            }
        }
        
        apply_gaussian_filter(P, filter, filter_size, half_size);

        if (status != GSL_SUCCESS) {
            cout << "Error: " << gsl_strerror(status) << endl;
            fP << t;
            for (int i = 0; i < npoints; i++){
                fP << "\t" << P[i];
            }
            fP << endl;
            cout << t << "\t" << nav << "\t" << n2av << endl;

            break;
        }else if (t > iter * tsample) {
            fP << t;
            for (int i = 0; i < npoints; i++){
                fP << "\t" << P[i];
            }
            fP << endl;
            averages(P, nav, n2av);
            cout << t << "\t" << nav << "\t" << n2av << endl;
            
            iter++;
        }
        
    }

    gsl_odeiv2_driver_free (driver);
    fP.close();
}



int main(int argc, char *argv[]) {
    temp = atof(argv[1]);
    lambda = atof(argv[2]);
    double nmax = atof(argv[3]);
    npoints = atoi(argv[4]);
    double tmax = atof(argv[5]);
    double abstol = atof(argv[6]);
    double reltol = atof(argv[7]);
    double dt0 = atof(argv[8]);
    double n0 = atof(argv[9]);
    double sigmasqr0 = atof(argv[10]);
    double tsample = atof(argv[11]);
    long maxiter = atol(argv[12]);
    int filter_size = atoi(argv[13]);
    double sigma_filter = atof(argv[14]);

    double sigma0 = sqrt(sigmasqr0);

    double dn = nmax / (npoints - 1);
    n_grid = new double[npoints];
    P_der_n = new double[npoints];
    P_der2_n = new double[npoints];
    
    init_arrays(dn);

    double *P;
    initial_condition(P, n0, sigma0);
    double t0 = 0;

    char fileP[300];
    sprintf(fileP, "FokkerPlanck_single_var_temp_%.3lf_lambda_%.1e_nmax_%.2lf_npoints_%d_tmax_%.2lf_atol_%.1e_rtol_%.1e_dt0_%.1e_n0_%.2lf_sigma0_%.2lf_ts_%.4lf_maxiter_%li_filtersize_%d_sigmaf_%.3lf.txt",
            temp, lambda, nmax, npoints, tmax, abstol, reltol, dt0, n0, sigma0, tsample, maxiter, filter_size, sigma_filter);

    integrate(P, t0, tmax, dt0, abstol, reltol, tsample, n0, sigma0, fileP, maxiter, 
              filter_size, sigma_filter);

    return 0;
}