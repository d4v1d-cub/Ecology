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
    long nodes_in[2]; // nodes inside the edge. nodes_in[i], with i={0, 1}.
    double links[2]; // links[i], with i={0, 1}. links[i] is the one pointing to the variable in nodes[i]
}Tedge;



long init_graph_inside_RRG(Tedge *&edges, long N, int c, double eps,
                           double mu, double sigma, gsl_rng * r){
    // eps is the degree of symmetry of the graph
    if (N * c % 2 != 0){
        cout << "N*c must be even to create a random regular graph" << endl;
        exit(1);
    }else{
        long M = N * c / 2;
        long pos_i, pos_j, i, j;
        double aij, aji; // aij is the coupling that node j sees from node i
        edges = new Tedge[M];

        bool success = false;
        while (!success){

            vector < long > copies = vector < long > (c * N);
            for (long i = 0; i < N; i++){
                for (int k = 0; k < c; k++){
                    copies[i * c + k] = i;
                }
            }

            for (long e = 0; e < M - 1; e++){
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

                edges[e].nodes_in[0] = i;
                edges[e].nodes_in[1] = j;

                aij = mu + gsl_ran_gaussian(r, sigma);
                if (gsl_rng_uniform_pos(r) < eps){
                    aji = aij;
                }else{
                    aji = mu + gsl_ran_gaussian(r, sigma);
                }
                edges[e].links[0] = aji;
                edges[e].links[1] = aij;
            }
            
            pos_i = 0;
            i = copies[pos_i];
            pos_j = 1;
            j = copies[pos_j];
            if (i != j){
                success = true;
                edges[M - 1].nodes_in[0] = i;
                edges[M - 1].nodes_in[1] = j;
                

                aij = mu + gsl_ran_gaussian(r, sigma);
                if (gsl_rng_uniform_pos(r) < eps){
                    aji = aij;
                }else{
                    aji = mu + gsl_ran_gaussian(r, sigma);
                }
                edges[M - 1].links[0] = aji;
                edges[M - 1].links[1] = aij;
            }
        }
        return M;
    }
}


void print_graph(Tedge *edges, long N, long M){
    cout << N << "\t" << M << endl;
    for (long e = 0; e < M; e++){
        cout << edges[e].nodes_in[0] << "\t" << edges[e].nodes_in[1] << "\t" 
             << edges[e].links[0] << "\t" << edges[e].links[1] << endl;
    }
}


int main(int argc, char *argv[]) {
    unsigned long seed = atoi(argv[1]);
    double eps = atof(argv[2]);
    double mu = atof(argv[3]);
    double sigma = atof(argv[4]);

    Tedge *edges;
    long N;
    
    N = atol(argv[5]);
    int c = atoi(argv[6]);
    gsl_rng * r;

    init_ran(r, seed);

    long M = init_graph_inside_RRG(edges, N, c, eps, mu, sigma, r);

    print_graph(edges, N, M);

    return 0;
}