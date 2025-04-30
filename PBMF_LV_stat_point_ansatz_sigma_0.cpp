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
    vector <double> mess;
    vector <double> mess_hat;
}Tedge;




Tedge init_edge(double avn_0, long npoints0){
    Tedge edge;
    edge.mess = vector <double> (npoints0, avn_0);
    edge.mess_hat = vector <double> (npoints0, avn_0);
    return edge;
}


void reset_edge(double avn_0, Tedge &edge){
    for (long l = 0; l < edge.mess.size(); l++){
        edge.mess[l] = avn_0;
        edge.mess_hat[l] = avn_0;
    }
}


double A_ki(double nk, double ni, double mhat_ik, double mhat_ki, double mu){
    return pow(mu * nk, 2) + pow(nk + mhat_ki, 2) + 2 * mhat_ki * (mu * ni - 1) + \
           2 * nk * ((mu + mu) * ni + mu * mhat_ik - mu - 1);
}

double integrand_R_ki(double nk, double ni, double mhat_ik, double mhat_ki, double mu, 
                      double beta, double lambda){
    return pow(nk, beta * lambda) * exp(-0.5 * beta * A_ki(nk, ni, mhat_ik, mhat_ki, mu));
    
}

double R_ki(double ni, double mhat_ik, vector <double> mhat_ki, double mu, 
            double beta, double lambda, double dn, vector <double> &saved_integrands, 
            double error, double nmin){
    saved_integrands[0] = integrand_R_ki(nmin, ni, mhat_ik, mhat_ki[0], mu, beta, lambda);
    double integral = saved_integrands[0] * nmin / (beta * lambda + 1);
    double nk = nmin + dn;
    for (long l = 1; l < mhat_ki.size(); l++){
        saved_integrands[l] = integrand_R_ki(nk, ni, mhat_ik, mhat_ki[l], mu, beta, lambda);
        integral += 0.5 * (saved_integrands[l - 1] + saved_integrands[l]) * dn;
        nk += dn;
    }
    if (saved_integrands[saved_integrands.size() - 1] * dn > error){
        double mhat_ki_out = mhat_ki[mhat_ki.size() - 1];
        double mhat_ki_der_ext = (mhat_ki[mhat_ki.size() - 1] - mhat_ki[mhat_ki.size() - 2]) / dn;
        double integrand_prev = saved_integrands[saved_integrands.size() - 1];
        double integrand;
        while (integrand_prev * dn > error){
            mhat_ki_out = mhat_ki_out + mhat_ki_der_ext * dn;
            if (mhat_ki_out < 0){
                mhat_ki_out = 0;
                mhat_ki_der_ext = 0;
            }
            integrand = integrand_R_ki(nk, ni, mhat_ik, mhat_ki_out, mu, beta, lambda);
            integral += 0.5 * (integrand_prev + integrand) * dn;
            integrand_prev = integrand;
            nk += dn;
            // cout << "Integral R_ki did not converge in the selected interval" << endl;
        }
    }
    return integral;
}


double Z_ki(double ni, double mhat_ik, vector <double> mhat_ki, 
            double mu, double beta, double lambda, double dn, 
            vector <double> saved_integrands, double val_R_ki, double error, double nmin){
    double integral = saved_integrands[0] / (beta * lambda);
    double nk = nmin;
    for (long l = 1; l < mhat_ki.size(); l++){
        integral += 0.5 * (saved_integrands[l - 1] / nk + 
                           saved_integrands[l] / (nk + dn)) * dn;
        nk += dn;
    }

    if (saved_integrands[mhat_ki.size() - 1] / nk * dn > error){
        double mhat_ki_out = mhat_ki[mhat_ki.size() - 1];
        double mhat_ki_der_ext = (mhat_ki[mhat_ki.size() - 1] - mhat_ki[mhat_ki.size() - 2]) / dn;
        double integrand_prev = saved_integrands[mhat_ki.size() - 1];
        double integrand;
        while (integrand_prev / nk * dn > error)
        {
            mhat_ki_out = mhat_ki_out + mhat_ki_der_ext * dn;
            if (mhat_ki_out < 0){
                mhat_ki_out = 0;
                mhat_ki_der_ext = 0;
            }
            integrand = integrand_R_ki(nk + dn, ni, mhat_ik, mhat_ki_out, mu, beta, lambda);
            integral += 0.5 * (integrand_prev / nk + integrand / (nk + dn)) * dn;
            nk += dn;
            integrand_prev = integrand;
        }
    }

    return integral;
}


void compute_new_m_ki(vector <double> mhat, double mu, 
                      double beta, double lambda, double dn, double error, vector <double> &mess, double nmin){
    vector <double> saved_integrands = vector <double> (mhat.size(), 0);
    double ni = lambda;
    double val_R_ki, val_Z_ki;
    for (long l = 0; l < mhat.size(); l++){
        val_R_ki = R_ki(ni, mhat[l], mhat, mu, beta, lambda, dn, saved_integrands, error, nmin);
        val_Z_ki = Z_ki(ni, mhat[l], mhat, mu, beta, lambda, dn, saved_integrands, val_R_ki, error, nmin);
        mess[l] = val_R_ki / val_Z_ki;
        if(mess[l] < 0){
            cout << "some  m_ki < 0" << endl;
            exit(1);
        }else if (mess[l] > 100000)
        {
            cout << "some  m_ki > 100000" << endl;
            exit(1);
        }
        
        ni += dn;
    }
}


void update_m(double beta, double lambda, double dn, double error, Tedge &edge, double nmin, double mu){
    compute_new_m_ki(edge.mess_hat, mu, beta, lambda, dn, error, edge.mess, nmin);
}


double update_mhat(Tedge &edge, long size_mess, double mu, int c){
    double var_mhat = 0;
    double mhat_ij_new;
    for (int l = 0; l < size_mess; l++){
        mhat_ij_new = (c - 1) * mu * edge.mess[l];
        if (fabs(mhat_ij_new - edge.mess_hat[l]) > var_mhat){
            var_mhat = fabs(mhat_ij_new - edge.mess_hat[l]);
        }
        edge.mess_hat[l] = mhat_ij_new;
    }
    return var_mhat;
}


double distribution(double ni, double nj, double mhat_ij, double mhat_ji, 
                    double mu, double beta){
    return exp(-0.5 * beta * pow(ni - 1 + mu * nj + mhat_ij, 2)) * 
           exp(-0.5 * beta * pow(nj - 1 + mu * ni + mhat_ji, 2));
}



double get_av(vector <double> mhat, double mu, double beta, double lambda, double dn, double error, double nmin){
    double integral_num = 0, last_integrand_num, last_integrand_den;
    double integral_den = 0;
    double integral_in, integral_in_prev;
    double dist_prev, dist;
    
    dist_prev = distribution(nmin, nmin, mhat[0], mhat[0], mu, beta);
    double nj = nmin + dn;
    integral_in_prev = 0;
    for (long lj = 1; lj < mhat.size(); lj++){
        dist = distribution(nmin, nj, mhat[0], mhat[lj], mu, beta);
        integral_in_prev += 0.5 * (dist_prev + dist) * dn;
        nj += dn;
        dist_prev = dist;
    }

    double ni = nmin + dn;
    for (long li = 1; li < mhat.size(); li++){
        dist_prev = distribution(ni, nmin, mhat[li], mhat[0], mu, beta);
        nj = nmin + dn;
        integral_in = 0;
        for (long lj = 1; lj < mhat.size(); lj++){
            dist = distribution(ni, nj, mhat[li], mhat[lj], mu, beta);
            integral_in += 0.5 * (dist_prev + dist) * dn;
            nj += dn;
            dist_prev = dist;
        }
        last_integrand_num = 0.5 * (integral_in_prev * (ni - dn) + integral_in * ni) * dn;
        last_integrand_den = 0.5 * (integral_in_prev + integral_in) * dn;
        integral_num += last_integrand_num;
        integral_den += last_integrand_den;
        integral_in_prev = integral_in;
        ni += dn;
    }

    
    return integral_num / integral_den;
}


double comp_averages(Tedge edge, double beta, double lambda, double dn, double error, double nmin, double mu){
    return get_av(edge.mess_hat, mu, beta, lambda, dn, error, nmin);
}


int convergence(Tedge &edge, double beta, double lambda, double dn, double tol, int max_iter, 
                double tol_integrals, double nmin, double mu, int c){
    double var_mhat = tol + 1;
    int iter = 0;
    while (var_mhat > tol && iter < max_iter){
        update_m(beta, lambda, dn, tol_integrals, edge, nmin, mu);
        var_mhat = update_mhat(edge, edge.mess.size(), mu, c);
        iter++;
    }
    return iter;
}



void print_results(int iter, Tedge edge, double beta, double lambda, 
                   double dn, int max_iter, char *filemess, 
                   double tol_integrals, double nmin, double mu){
    double field = comp_averages(edge, beta, lambda, dn, tol_integrals, nmin, mu);
    bool conv = iter < max_iter;
    cout << iter << "\t" << conv << "\t" << field << endl;

    ofstream fmess(filemess);
    fmess << "\t" << "mess";
    for (long l = 0; l < edge.mess.size(); l++){
        fmess << "\t" << edge.mess[l];
    }
    fmess << endl;  
    fmess.close();
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
    double nmin = atof(argv[9]);
    double nmax = atof(argv[10]);
    int npoints = atoi(argv[11]);
    double tol_integrals = atof(argv[12]);
    int c = atoi(argv[13]);

    double beta = 1.0 / T;

    char filemess[300];
    sprintf(filemess, "PBMF_Lotka_Volterra_nonoise_steady_state_mess_T_%.2lf_lambda_%.2lf_av0_%.2lf_tol_%.1e_maxiter_%d_mu0_%.4lf_dmu_%.4lf_muf_%.4lf_nmin_%.1e_nmax_%.2lf_npoints_%d_c_%d.txt", 
                      T, lambda, avn_0, tol, max_iter, mu0, dmu, muf, nmin, nmax, npoints, c);

    double dn = (nmax - nmin) / npoints;

    Tedge edge = init_edge(avn_0, npoints);

    int iter;
    double field;
    bool conv;

    ofstream fmess(filemess);

    for (double mu = mu0; mu < muf + dmu / 2; mu += dmu){
        iter = convergence(edge, beta, lambda, dn, tol, max_iter, tol_integrals, nmin, mu, c);

        field = comp_averages(edge, beta, lambda, dn, tol_integrals, nmin, mu);
        conv = iter < max_iter;
        cout << mu << "\t" << iter << "\t" << conv << "\t" << field << endl;

        fmess << "\t" << "mess";
        for (long l = 0; l < edge.mess.size(); l++){
            fmess << "\t" << edge.mess[l];
        }
        fmess << endl;

        reset_edge(avn_0, edge);
    }

    fmess.close();
    
    return 0;
}