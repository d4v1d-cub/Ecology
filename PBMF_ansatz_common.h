#ifndef __PBMF_ANSATZ_COMMON_H_INCLUDED__
#define __PBMF_ANSATZ_COMMON_H_INCLUDED__

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
    double av_abundance; // average value of n in that node
    vector <double> Psingle; // Psingle[n_index] is the probability of n = n_grid[n_index] in that node
}Tnode;




typedef struct{
    vector <long> nodes_in; // nodes inside the edge. nodes_in[i], with i={0, 1}.
    vector <double> links; // links[i], with i={0, 1}. links[i] is the one pointing to the variable in nodes_in[i]
    vector < vector <double> > cond_av; // cond_av[i], with i = {0, 1}. cond_av[i] is the average of the variable in nodes_in[i]
    // conditioned to the variable in nodes_in[1 - i] being fixed. cond_av[i] is computed as an integral over the conditional distribution.
    vector < vector <double> > integrated_cond_av; // integrated_cond_av[i][n_index], with i = {0, 1}, is the integral of cond_av[i] 
    // from n0 to n=n_grid[n_index]
    vector < vector <long> > edges_except; // edges that contain the node in nodes_in[i] excepting this edge
    vector < vector <int> > pos_there; // position occupied by the node in nodes_in[i] in those edges
    vector <int> edge_index; // position of the edge in the list of edges that contain the node
    vector <bool> converged; // converged[i], with i = {0, 1}. converged[i] is true if the conditional average cond_av[i] arrived to convergence.
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


void init_nodes(Tnode *nodes, long N){
    for (long i = 0; i < N; i++){
        nodes[i].edges_in = vector <long> ();
        nodes[i].pos_there = vector <int> ();
    }
}

void init_graph_inside_RRG(Tnode *&nodes, Tedge *&edges, long N, int c, double eps,
                           double mu, double sigma, gsl_rng * r){
    // eps is the degree of symmetry of the graph
    if (N * c % 2 != 0){
        cout << "N*c must be even to create a random regular graph" << endl;
        exit(1);
    }else{
        long M = N * c / 2;
        long pos_i, pos_j, i, j;
        double aij, aji;
        nodes = new Tnode[N];
        edges = new Tedge[M];

        for (long e = 0; e < M; e++){
            edges[e].nodes_in = vector <long> (2);
            edges[e].links = vector <double> (2);
            edges[e].edge_index = vector <int> (2);
        }

        bool success = false;
        while (!success){
            init_nodes(nodes, N);

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

                edges[e].nodes_in.push_back(i);
                edges[e].nodes_in.push_back(j);
                
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
            
            pos_i = 0;
            i = copies[pos_i];
            pos_j = 1;
            j = copies[pos_j];
            if (i != j){
                success = true;
                edges[M - 1].nodes_in.push_back(i);
                edges[M - 1].nodes_in.push_back(j);
                
                edges[M - 1].edge_index[0] = nodes[i].edges_in.size();
                edges[M - 1].edge_index[1] = nodes[j].edges_in.size();

                aij = mu + gsl_ran_gaussian(r, sigma);
                if (gsl_rng_uniform_pos(r) < eps){
                    aji = aij;
                }else{
                    aji = mu + gsl_ran_gaussian(r, sigma);
                }
                edges[M - 1].links.push_back(aji);
                edges[M - 1].links.push_back(aij);

                nodes[i].edges_in.push_back(M - 1);
                nodes[j].edges_in.push_back(M - 1);
                nodes[i].pos_there.push_back(0);
                nodes[j].pos_there.push_back(1);
            }
        }
        fill_except(nodes, edges, M);
    }
}


vector <double> init_grid(double n0, double n1, double dn, double nmax){
    vector <double> n_grid;
    n_grid.push_back(n0);
    n_grid.push_back(n1);
    for (double n = n1 + dn; n <= nmax; n += dn){
        n_grid.push_back(n);
    }
    return n_grid;
}


vector <double> compute_simpson_weights(int npoints, double dn){
    vector <double> simpson_weights = vector <double> (npoints, dn / 3);
    for (int i = 1; i < npoints - 1; i+=2){
        simpson_weights[i] *= 4;
    }
    for (int i = 2; i < npoints - 1; i+=2){
        simpson_weights[i] *= 2;
    }
}


vector <double> compute_fixed_integrand(double beta, double lambda, int p, vector <double> n_grid){
    vector <double> integrand = vector <double> (n_grid.size() - 1, 0);
    for (int i = 1; i < n_grid.size(); i++){
        integrand[i - 1] = pow(n_grid[i], beta * lambda - 1 + p) * exp(-beta * (n_grid[i] * n_grid[i] - 2 * n_grid[i]) / 2);
    }
    return integrand;
}

void init_cond_av(long M, Tedge *edges, long npoints){
    for (long e = 0; e < M; e++){
        edges[e].cond_av = vector < vector <double> > (2, vector <double> (npoints, 0));
        edges[e].integrated_cond_av = vector < vector <double> > (2, vector <double> (npoints, 0));
    }
}


void init_auxiliary_vectors(vector <double> &n_grid, vector <double> &simpson_weights, vector <double> &fixed_integrand_num, 
                            vector <double> &fixed_integrand_den, double beta, double lambda, double n0, double n1, double dn,
                            double nmax){
    n_grid = init_grid(n0, n1, dn, nmax);
    simpson_weights = compute_simpson_weights(n_grid.size() - 1, dn);
    fixed_integrand_num = compute_fixed_integrand(beta, lambda, 1, n_grid);
    fixed_integrand_den = compute_fixed_integrand(beta, lambda, 0, n_grid);
}




void produce_random_seq(unsigned long seed_seq, long M, long sequence[]){
    gsl_rng * r;
    init_ran(r, seed_seq);
    vector <long> elements(M);
    for (long i = 0; i < M; i++){
        elements[i] = i;
    }
    long pos;
    for (long i = 0; i < M; i++){
        pos = gsl_rng_uniform_int(r, M - i);
        sequence[i] = elements[pos];
        elements.erase(elements.begin() + pos);
    }
    gsl_rng_free(r);
}


#endif