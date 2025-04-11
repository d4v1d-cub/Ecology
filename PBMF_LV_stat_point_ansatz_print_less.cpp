#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include "math.h"

using namespace std;

void init_ran(gsl_rng * &r, unsigned long s){
    const gsl_rng_type * T;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, s);
}


typedef struct{
    vector <long> edges_in; // edges that contain the node
    vector <int> pos_there; // position occupied by the node in those edges
    double avn; // average value of n in that node
}Tnode;




typedef struct{
    vector <long> nodes_in; // nodes inside the edge. nodes[i], with i={0, 1}.
    vector <double> links; // links[i], with i={0, 1}. links[i] is the one pointing to the variable in nodes[i]
    vector < vector <double> > mess; // message from edge to node, computed as an integral over 
    // the conditional distribution. mess[i], with i={0, 1}. mess[i] is the one pointing to the variable in nodes[i] coming
    // from the variable in mess[1 - i].
    vector <vector <double> > z_mess; // normalization constants for the message from edge to node. 
    vector < vector <double> > mess_hat;  // message from node to edge, computed as a weighted sum of messages
    // from edge to node. mess_hat[i], with i={0, 1}. mess_hat[i] is the one pointing to the node in nodes[1 - i] from
    // the variable in nodes[i]. 
    vector < vector <long> > edges_except; // edges that contain the node in nodes[i] excepting this edge
    vector < vector <int> > pos_there; // position occupied by the node in nodes[i] in those edges
    vector <int> edge_index; // position of the edge in the list of edges that contain the node
}Tedge;


void fill_except(Tnode *nodes, Tedge *edges, long M){
    for (long e = 0; e < M; e++){
        edges[e].edges_except = vector < vector <long> > (2, vector <long> ());
        edges[e].pos_there = vector < vector <int> > (2, vector <int> ());
        for (int i = 0; i < 2; i++){
            for (int k = 0; k < edges[e].edge_index[i]; k++){
                edges[e].edges_except[i].push_back(nodes[edges[e].nodes_in[i]].edges_in[k]);
                edges[e].pos_there[i].push_back(nodes[edges[e].nodes_in[i]].pos_there[k]);
            }
            for (int k = edges[e].edge_index[i] + 1; k < nodes[edges[e].nodes_in[i]].edges_in.size(); k++){
                edges[e].edges_except[i].push_back(nodes[edges[e].nodes_in[i]].edges_in[k]);
                edges[e].pos_there[i].push_back(nodes[edges[e].nodes_in[i]].pos_there[k]);
            }
        }
    }
}


void init_graph_from_input(Tnode *&nodes, Tedge *&edges, long &N, long &M){
    scanf("%ld %ld", &N, &M);
    nodes = new Tnode[N];
    edges = new Tedge[M];
    long i, j;
    double aij, aji;
    for (long e = 0; e < M; e++){
        scanf("%ld %ld %lf %lf", &i, &j, &aij, &aji);
        edges[e].nodes_in.push_back(i);
        edges[e].nodes_in.push_back(j);
        edges[e].links.push_back(aji);
        edges[e].links.push_back(aij);

        edges[e].edge_index = vector <int> (2);
        edges[e].edge_index[0] = nodes[i].edges_in.size();
        edges[e].edge_index[1] = nodes[j].edges_in.size();
        

        nodes[i].edges_in.push_back(e);
        nodes[j].edges_in.push_back(e);
        nodes[i].pos_there.push_back(0);
        nodes[j].pos_there.push_back(1);
    }

    fill_except(nodes, edges, M);
}


void init_graph_inside_RRG(Tnode *&nodes, Tedge *&edges, long N, int c, double eps,
                           double mu, double sigma, gsl_rng * r){
    // eps is the degree of symmetry of the graph
    if (N * c % 2 != 0){
        cout << "N*c must be even to create a random regular graph" << endl;
        exit(1);
    }else{
        long M = N * c / 2;
        nodes = new Tnode [N];
        edges = new Tedge [M];
        long pos_i, pos_j, i, j;
        double aij, aji;
        vector < long > copies = vector < long > (c * N);
        for (long i = 0; i < N; i++){
            for (int k = 0; k < c; k++){
                copies[i * c + k] = i;
            }
        }

        for (long e = 0; e < M; e++){
            pos_i = gsl_rng_uniform_int(r, copies.size());
            i = copies[pos_i];
            copies.erase(copies.begin() + pos_i);
            pos_j = gsl_rng_uniform_int(r, copies.size());
            j = copies[pos_j];
            while (j == i){
                pos_j = gsl_rng_uniform_int(r, copies.size());
                j = copies[pos_j];
            }
            copies.erase(copies.begin() + pos_j);

            edges[e].nodes_in.push_back(i);
            edges[e].nodes_in.push_back(j);
            
            edges[e].edge_index = vector <int> (2);
            edges[e].edge_index[0] = nodes[i].edges_in.size();
            edges[e].edge_index[1] = nodes[j].edges_in.size();

            aij = mu + gsl_ran_gaussian(r, sigma);
            if (gsl_rng_uniform_pos(r) < eps){
                aji = aij;
            }else{
                aji = mu + gsl_ran_gaussian(r, sigma);
            }
            edges[e].links.push_back(aji);
            edges[e].links.push_back(aij);

            nodes[i].edges_in.push_back(e);
            nodes[j].edges_in.push_back(e);
            nodes[i].pos_there.push_back(0);
            nodes[j].pos_there.push_back(1);
        }
        fill_except(nodes, edges, M);
    }
}


void init_messages(long M, double avn_0, long npoints0, Tedge *edges){
    for (long e = 0; e < M; e++){
        edges[e].mess = vector < vector <double> > (2, vector <double> (npoints0, avn_0));
        edges[e].mess_hat = vector < vector <double> > (2, vector <double> (npoints0, avn_0));
        edges[e].z_mess = vector < vector <double> > (2, vector <double> (npoints0, 1));
    }
}


vector <double> derivative(vector <double> vals, double dx){
    vector <double> der = vector <double> (vals.size(), 0);
    for (long i = 0; i < vals.size() - 1; i++){
        der[i] = (vals[i + 1] - vals[i]) / dx;
    }
    der[vals.size() - 1] = der[vals.size() - 2];
    return der;
}


double A_ki(double nk, double ni, double mhat_ik, double mhat_ki, double aik, double aki){
    return pow(aki * nk, 2) + pow(nk + mhat_ki, 2) + 2 * mhat_ki * (aik * ni - 1) + \
           2 * nk * ((aik + aki) * ni + aki * mhat_ik - aki - 1);
}

double integrand_R_ki(double nk, double ni, double mhat_ik, double mhat_ki, double aik, double aki, 
                      double beta, double lambda){
    return pow(nk, beta * lambda) * exp(-0.5 * beta * A_ki(nk, ni, mhat_ik, mhat_ki, aik, aki));
    
}

double R_ki(double ni, double mhat_ik, vector <double> mhat_ki, double aik, double aki, 
            double beta, double lambda, double dn, vector <double> &saved_integrands, 
            double error, double nmin){
    saved_integrands[0] = integrand_R_ki(nmin, ni, mhat_ik, mhat_ki[0], aik, aki, beta, lambda);
    double integral = saved_integrands[0] * nmin / (beta * lambda + 1);
    double nk = nmin + dn;
    for (long l = 1; l < mhat_ki.size(); l++){
        saved_integrands[l] = integrand_R_ki(nk, ni, mhat_ik, mhat_ki[l], aik, aki, beta, lambda);
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
            integrand = integrand_R_ki(nk, ni, mhat_ik, mhat_ki_out, aik, aki, beta, lambda);
            integral += 0.5 * (integrand_prev + integrand) * dn;
            integrand_prev = integrand;
            nk += dn;
            // cout << "Integral R_ki did not converge in the selected interval" << endl;
        }
    }
    return integral;
}


double Z_ki(double ni, double mhat_ik, vector <double> mhat_ki, 
            double aik, double aki, double beta, double lambda, double dn, 
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
            integrand = integrand_R_ki(nk + dn, ni, mhat_ik, mhat_ki_out, aik, aki, beta, lambda);
            integral += 0.5 * (integrand_prev / nk + integrand / (nk + dn)) * dn;
            nk += dn;
            integrand_prev = integrand;
        }
    }

    return integral;
}


void compute_new_m_ki(vector <double> mhat_ik, vector <double> mhat_ki, double aik, double aki, 
                      double beta, double lambda, double dn, double error, vector <double> &m_ki,
                      vector <double> &z_mess_ki, double nmin){
    vector <double> saved_integrands = vector <double> (mhat_ki.size(), 0);
    double ni = lambda;
    double val_R_ki;
    for (long l = 0; l < mhat_ik.size(); l++){
        val_R_ki = R_ki(ni, mhat_ik[l], mhat_ki, aik, aki, beta, 
                        lambda, dn, saved_integrands, error, nmin);
        z_mess_ki[l] = Z_ki(ni, mhat_ik[l], mhat_ki, aik, aki, beta, lambda, dn, 
                            saved_integrands, val_R_ki, error, nmin);
        m_ki[l] = val_R_ki / z_mess_ki[l];
        if(m_ki[l] < 0){
            cout << "some  m_ki < 0" << endl;
            exit(1);
        }else if (m_ki[l] > 100000)
        {
            cout << "some  m_ki > 100000" << endl;
            exit(1);
        }
        
        ni += dn;
    }
}


void update_all_m(long M, double beta, double lambda, double dn, double error, 
                  Tedge *edges, double nmin){
    for (long e = 0; e < M; e++){
        compute_new_m_ki(edges[e].mess_hat[0], edges[e].mess_hat[1], edges[e].links[1], edges[e].links[0], 
                         beta, lambda, dn, error, edges[e].mess[0], edges[e].z_mess[0], nmin);
        compute_new_m_ki(edges[e].mess_hat[1], edges[e].mess_hat[0], edges[e].links[0], edges[e].links[1], 
                         beta, lambda, dn, error, edges[e].mess[1], edges[e].z_mess[1], nmin);
    }
}


void compute_new_mhat_ij(Tedge *edges, long e, int place, vector <long> edges_except,
                         vector <int> pos_there, long size_mess, double &var_mhat){
    double mhat_ij_new;
    for (int l = 0; l < size_mess; l++){
        mhat_ij_new = 0;
        for (int k = 0; k < edges_except.size(); k++){
            mhat_ij_new += edges[edges_except[k]].links[pos_there[k]] * 
                           edges[edges_except[k]].mess[pos_there[k]][l];
        }
        if (fabs(mhat_ij_new - edges[e].mess_hat[place][l]) > var_mhat){
            var_mhat = fabs(mhat_ij_new - edges[e].mess_hat[place][l]);
        }
        edges[e].mess_hat[place][l] = mhat_ij_new;
    }
}


double update_all_mhat_ij(Tedge *edges, long M){
    double var_mhat = 0;
    for (long e = 0; e < M; e++){
        compute_new_mhat_ij(edges, e, 0, edges[e].edges_except[0], edges[e].pos_there[0], edges[e].mess[0].size(), var_mhat);
        compute_new_mhat_ij(edges, e, 1, edges[e].edges_except[1], edges[e].pos_there[1], edges[e].mess[1].size(), var_mhat);
    }
    return var_mhat;
}


int convergence(Tedge *edges, long M, double beta, double lambda, double dn, double tol, 
                 int max_iter, double tol_integrals, char *filehist, double nmin){
    double var_mhat = tol + 1;
    int iter = 0;
    ofstream fh(filehist);
    fh << "iter\tmax(dmhat)" << endl;
    while (var_mhat > tol && iter < max_iter){
        update_all_m(M, beta, lambda, dn, tol_integrals, edges, nmin);
        var_mhat = update_all_mhat_ij(edges, M);
        iter++;
        fh << iter << "\t" << var_mhat << endl;
    }
    fh.close();
    return iter;
}


double distribution(double ni, double nj, double mhat_ij, double mhat_ji, 
                    double aij, double aji, double beta){
    return exp(-0.5 * beta * pow(ni - 1 + aji * nj + mhat_ij, 2)) * 
           exp(-0.5 * beta * pow(nj - 1 + aij * ni + mhat_ji, 2));
}



double get_av(vector <double> mhat_ij, vector <double> mhat_ji, double aij, double aji, 
              double beta, double lambda, double dn, double error, double nmin){
    double integral_num = 0, last_integrand_num, last_integrand_den;
    double integral_den = 0;
    double integral_in, integral_in_prev;
    double dist_prev, dist;
    
    dist_prev = distribution(nmin, nmin, mhat_ij[0], mhat_ji[0], aij, aji, beta);
    double nj = nmin + dn;
    integral_in_prev = 0;
    for (long lj = 1; lj < mhat_ji.size(); lj++){
        dist = distribution(nmin, nj, mhat_ij[0], mhat_ji[lj], aij, aji, beta);
        integral_in_prev += 0.5 * (dist_prev + dist) * dn;
        nj += dn;
        dist_prev = dist;
    }

    double ni = nmin + dn;
    for (long li = 1; li < mhat_ij.size(); li++){
        dist_prev = distribution(ni, nmin, mhat_ij[li], mhat_ji[0], aij, aji, beta);
        nj = nmin + dn;
        integral_in = 0;
        for (long lj = 1; lj < mhat_ji.size(); lj++){
            dist = distribution(ni, nj, mhat_ij[li], mhat_ji[lj], aij, aji, beta);
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

    if (last_integrand_num > error){
        double mhat_ij_out = mhat_ij[mhat_ij.size() - 1];
        double mhat_ij_der_ext = (mhat_ij[mhat_ij.size() - 1] - mhat_ij[mhat_ij.size() - 2]) / dn;
        double integral_in_prev = integral;
        double integrand;
        while (integrand_prev / ni * dn > error){
            mhat_ij_out = mhat_ij_out + mhat_ij_der_ext * dn;
            if (mhat_ij_out < 0){
                mhat_ij_out = 0;
                mhat_ij_der_ext = 0;
            }
            z_mess_ji_out = z_mess_ji_out + z_mess_ji_der_ext * dn;
            if (z_mess_ji_out < 0){
                z_mess_ji_out = 0;
                z_mess_ji_der_ext = 0;
            }
            integrand = integrand_Rpair(ni + dn, mhat_ij_out, z_mess_ji_out, aij, beta, lambda);
            integral += 0.5 * (integrand_prev / ni + integrand / (ni + dn)) * dn;
            ni += dn;
            integrand_prev = integrand;
        }
    }  

    return integral;
}


double R_ind(double beta, double lambda){
    return gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(-beta * lambda / 2, 0.5, -beta / 2) +
    sqrt(2 * beta) * gsl_sf_gamma(1 + beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 1.5, -beta / 2);
}


double Z_ind(double beta, double lambda){
    return sqrt(beta / 2) * gsl_sf_gamma(beta * lambda / 2) * gsl_sf_hyperg_1F1((1 - beta * lambda) / 2, 0.5, -beta / 2) + 
    beta * gsl_sf_gamma((1 + beta * lambda) / 2) * gsl_sf_hyperg_1F1(1 - beta * lambda / 2, 1.5, -beta / 2);
}


void comp_averages(long N, Tnode *nodes, Tedge *edges, double beta, double lambda, double dn,
                   double error, double nmin){
    double val_Rpair, val_Zpair;
    vector <double> saved_integrands = vector <double> (edges[0].mess[0].size(), 0);
    double val_ind = R_ind(beta, lambda) / Z_ind(beta, lambda);
    for (long i = 0; i < N; i++){
        if (nodes[i].edges_in.size() > 0){
            val_Rpair = Rpair(edges[nodes[i].edges_in[0]].mess_hat[nodes[i].pos_there[0]], 
                              edges[nodes[i].edges_in[0]].z_mess[nodes[i].pos_there[0]], 
                              edges[nodes[i].edges_in[0]].links[1 - nodes[i].pos_there[0]], 
                              beta, lambda, dn, saved_integrands, error, nmin);
            val_Zpair = Zpair(edges[nodes[i].edges_in[0]].mess_hat[nodes[i].pos_there[0]],
                              edges[nodes[i].edges_in[0]].z_mess[nodes[i].pos_there[0]], 
                              edges[nodes[i].edges_in[0]].links[1 - nodes[i].pos_there[0]], 
                              beta, lambda, dn, saved_integrands, val_Rpair, error, nmin);
            nodes[i].avn = val_Rpair / val_Zpair;
        }else{
            nodes[i].avn = val_ind;
        }
    }
}

double average(long N, Tnode *nodes){
    double av = 0;
    for (long i = 0; i < N; i++){
        av += nodes[i].avn;
    }
    return av / N;
}

double average_sqr(long N, Tnode *nodes){
    double av_sqr = 0;
    for (long i = 0; i < N; i++){
        av_sqr += nodes[i].avn * nodes[i].avn;
    }
    return av_sqr / N;
}



void print_results(int iter, Tnode *nodes, Tedge *edges, long N, long M, double beta, double lambda, 
                   double dn, long seed, int max_iter, char *fileavn, char *filemess, 
                   double tol_integrals, double nmin){
    comp_averages(N, nodes, edges, beta, lambda, dn, tol_integrals, nmin);
    double av = average(N, nodes);
    double av_sqr = average_sqr(N, nodes);
    bool conv = iter < max_iter;
    cout << iter << "\t" << conv << "\t" << av << "\t" << sqrt((av_sqr - av * av) / N) << "\t" << seed << endl;

    ofstream favn(fileavn);
    for (long i = 0; i < N; i++){
        favn << i << "\t" << nodes[i].avn << endl;
    }
    favn.close();

    ofstream fmess(filemess);
    for (long e = 0; e < M; e++){
        for (int i = 0; i < 2; i++){
            fmess << e << "\t" << "m_" << edges[e].nodes_in[1 - i] << "to" << edges[e].nodes_in[i];
            for (long l = 0; l < edges[e].mess[i].size(); l++){
                fmess << "\t" << edges[e].mess[i][l];
            }
            fmess << endl;  
        }
    }

    fmess.close();
}

int main(int argc, char *argv[]) {
    unsigned long seed = atoi(argv[1]);
    double avn_0 = atof(argv[2]);
    double T = atof(argv[3]);
    double lambda = atof(argv[4]);
    double tol = atof(argv[5]);
    int max_iter = atoi(argv[6]);
    double eps = atof(argv[7]);
    double mu = atof(argv[8]);
    double sigma = atof(argv[9]);
    double nmin = atof(argv[10]);
    double nmax = atof(argv[11]);
    int npoints = atoi(argv[12]);
    double tol_integrals = atof(argv[13]);
    bool gr_inside = atoi(argv[14]);


    Tnode *nodes;
    Tedge *edges;
    double beta = 1.0 / T;
    long N;
    long M;
    char gr_str[20];
    int c;

    if (gr_inside){
        sprintf(gr_str, "gr_inside_RRG");
        N = atol(argv[15]);
        c = atoi(argv[16]);
        gsl_rng * r;

        init_ran(r, seed);

        init_graph_inside_RRG(nodes, edges, N, c, eps, mu, sigma, r);
        M = N * c / 2;
    }else{
        sprintf(gr_str, "gr_from_input");
        init_graph_from_input(nodes, edges, N, M);
        c = 2 * M / N;
    }


    char filehist[200];
    sprintf(filehist, "PBMF_Lotka_Volterra_steady_state_convergence_%s_T_%.2lf_lambda_%.2lf_av0_%.2lf_tol_%.1e_maxiter_%d_eps_%.2lf_mu_%.2lf_sigma_%.2lf_nmin_%.1e_nmax_%.2lf_npoints_%d_N_%li_c_%d_seed_%li.txt", 
                      gr_str, T, lambda, avn_0, tol, max_iter, eps, mu, sigma, nmin, nmax, npoints, N, c, seed);
    char fileavn[200];
    sprintf(fileavn, "PBMF_Lotka_Volterra_steady_state_avn_%s_T_%.2lf_lambda_%.2lf_av0_%.2lf_tol_%.1e_maxiter_%d_eps_%.2lf_mu_%.2lf_sigma_%.2lf_nmin_%.1e_nmax_%.2lf_npoints_%d_N_%li_c_%d_seed_%li.txt", 
                      gr_str, T, lambda, avn_0, tol, max_iter, eps, mu, sigma, nmin, nmax, npoints, N, c, seed);
    char filemess[200];
    sprintf(filemess, "PBMF_Lotka_Volterra_steady_state_mess_%s_T_%.2lf_lambda_%.2lf_av0_%.2lf_tol_%.1e_maxiter_%d_eps_%.2lf_mu_%.2lf_sigma_%.2lf_nmin_%.1e_nmax_%.2lf_npoints_%d_N_%li_c_%d_seed_%li.txt", 
                      gr_str, T, lambda, avn_0, tol, max_iter, eps, mu, sigma, nmin, nmax, npoints, N, c, seed);

    double dn = (nmax - nmin) / npoints;

    init_messages(M, avn_0, npoints, edges);

    int iter = convergence(edges, M, beta, lambda, dn, tol, max_iter, tol_integrals, filehist, nmin);

    print_results(iter, nodes, edges, N, M, beta, lambda, dn, seed, max_iter, fileavn, filemess, tol_integrals, nmin);
    
    return 0;
}