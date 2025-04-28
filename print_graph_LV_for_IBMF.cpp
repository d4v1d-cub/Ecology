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
    vector <long> neighs;
    vector <double> links_in;
    double field; // average value of n in that node
}Tnode;


void init_graph_inside_RRG(Tnode *&nodes, long N, int c, double eps,
                           double mu, double sigma, gsl_rng * r){
    // eps is the degree of symmetry of the graph
    if (N * c % 2 != 0){
        cout << "N*c must be even to create a random regular graph" << endl;
        exit(1);
    }else{
        nodes = new Tnode [N];
        long M = N * c / 2;
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
            nodes[i].neighs.push_back(j);
            nodes[j].neighs.push_back(i);
            aij = mu + gsl_ran_gaussian(r, sigma);
            if (gsl_rng_uniform_pos(r) < eps){
                aji = aij;
            }else{
                aji = mu + gsl_ran_gaussian(r, sigma);
            }
            nodes[i].links_in.push_back(aji);
            nodes[j].links_in.push_back(aij);
        }
    }
}


void print_graph_IBMF(Tnode *nodes, long N, char *filegraph){
    ofstream fgraph(filegraph);
    for (long i = 0; i < N; i++){
        fgraph << i << "\t" << nodes[i].neighs.size() << endl;
        for (long j = 0; j < nodes[i].neighs.size(); j++){
            fgraph << i << "\t" << nodes[i].neighs[j] << "\t" << nodes[i].links_in[j] << endl;
        }
    }
}


int main(int argc, char *argv[]) {
    unsigned long seed = atoi(argv[1]);
    double eps = atof(argv[2]);
    double mu = atof(argv[3]);
    double sigma = atof(argv[4]);


    Tnode *nodes;
    double *avgs;
    long N;
    char gr_str[20];

    sprintf(gr_str, "gr_inside_RRG");
    N = atol(argv[5]);
    int c = atoi(argv[6]);
    gsl_rng * r;

    init_ran(r, seed);

    init_graph_inside_RRG(nodes, N, c, eps, mu, sigma, r);

    char filegraph[200];
    sprintf(filegraph, "LV_Graph_for_IBMF_N_%li_c_%d_eps_%.3lf_mu_%.3lf_sigma_%.3lf_seed_%li.txt", N, c, eps, mu, sigma, seed);
    print_graph_IBMF(nodes, N, filegraph);

    return 0;
}