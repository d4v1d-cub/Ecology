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



void read_mess(Tedge *edges, long M, char *filemess){
    ifstream fmess(filemess);
    int trash;
    string trash2;
    for (long e = 0; e < M; e++){
        for (int i = 0; i < 2; i++){
            fmess >> trash;
            fmess >> trash2;
            for (long l = 0; l < edges[e].mess[i].size(); l++){
                fmess >> edges[e].mess[i][l];
            }
        }
    }
    fmess.close();
}


void print_mhat(Tedge *edges, long M, char *filemess, char *filemhat){
    read_mess(edges, M, filemess);
    update_all_mhat_ij(edges, M);
    ofstream fmess(filemhat);
    for (long e = 0; e < M; e++){
        for (int i = 0; i < 2; i++){
            fmess << e << "\t" << "mhat_" << edges[e].nodes_in[i] << "to" << edges[e].nodes_in[1 - i];
            for (long l = 0; l < edges[e].mess_hat[i].size(); l++){
                fmess << "\t" << edges[e].mess_hat[i][l];
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


    char filemess[200];
    sprintf(filemess, "PBMF_Lotka_Volterra_steady_state_mess_%s_T_%.2lf_lambda_%.2lf_av0_%.2lf_tol_%.1e_maxiter_%d_eps_%.2lf_mu_%.2lf_sigma_%.2lf_nmin_%.1e_nmax_%.2lf_npoints_%d_N_%li_c_%d_seed_%li.txt", 
                      gr_str, T, lambda, avn_0, tol, max_iter, eps, mu, sigma, nmin, nmax, npoints, N, c, seed);

    char filemhat[200];
    sprintf(filemhat, "PBMF_Lotka_Volterra_steady_state_messhat_%s_T_%.2lf_lambda_%.2lf_av0_%.2lf_tol_%.1e_maxiter_%d_eps_%.2lf_mu_%.2lf_sigma_%.2lf_nmin_%.1e_nmax_%.2lf_npoints_%d_N_%li_c_%d_seed_%li.txt", 
                      gr_str, T, lambda, avn_0, tol, max_iter, eps, mu, sigma, nmin, nmax, npoints, N, c, seed);

    init_messages(M, avn_0, npoints, edges);

    print_mhat(edges, M, filemess, filemhat);
    
    return 0;
}