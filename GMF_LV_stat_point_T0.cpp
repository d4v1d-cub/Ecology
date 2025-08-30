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
    double av; // average value of n in that node
    double chi; // beta * (q_sqr - av^2) in that node
    double var; // variance of the perturberd Gaussian in that node
    double field; // field in that node
}Tnode;




typedef struct{
    long nodes_in[2]; // nodes inside the edge. nodes_in[i], with i={0, 1}.
    double links[2]; // links[i], with i={0, 1}. links[i] is the one pointing to the variable in nodes[i]
    double cond_av[2]; // cond_av[i], with i={0,1}, is the average of nodes_in[i] given that node j is zero
    vector < vector <long> > edges_except; // edges that contain the node in nodes[i] excepting this edge
    vector < vector <int> > pos_there; // position occupied by the node in nodes[i] in those edges
    int edge_index[2]; // position of the edge in the list of edges that contain the node
    double chi_cav[2]; // chi_cav[i], with i={0, 1}. chi_cav[i] = beta(cond_q_sqr_i - cond_av_i^2)
    bool chi_cav_converged[2];
    double var_cav[2];
    double fields_cav[2]; // fields_cav[i], with i={0, 1}, is the field in nodes_in[i] given that node j is zero
    bool var_cav_positive[2];
    bool converged[2]; // converged[i], with i={0, 1}, is true if the averages in nodes_in[i] have converged
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
    double aij, aji; // aij is the coupling that node j sees from node i
    for (long e = 0; e < M; e++){
        scanf("%ld %ld %lf %lf", &i, &j, &aij, &aji);
        edges[e].nodes_in[0] = i;
        edges[e].nodes_in[1] = j;
        edges[e].links[0] = aji;
        edges[e].links[1] = aij;

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

long init_graph_inside_RRG(Tnode *&nodes, Tedge *&edges, long N, int c, double eps,
                           double mu, double sigma, gsl_rng * r){
    // eps is the degree of symmetry of the graph
    if (N * c % 2 != 0){
        cout << "N*c must be even to create a random regular graph" << endl;
        exit(1);
    }else{
        long M = N * c / 2;
        long pos_i, pos_j, i, j;
        double aij, aji; // aij is the coupling that node j sees from node i
        nodes = new Tnode[N];
        edges = new Tedge[M];

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

                edges[e].nodes_in[0] = i;
                edges[e].nodes_in[1] = j;
                
                edges[e].edge_index[0] = nodes[i].edges_in.size();
                edges[e].edge_index[1] = nodes[j].edges_in.size();

                aij = mu + gsl_ran_gaussian(r, sigma);
                if (gsl_rng_uniform_pos(r) < eps){
                    aji = aij;
                }else{
                    aji = mu + gsl_ran_gaussian(r, sigma);
                }
                edges[e].links[0] = aji;
                edges[e].links[1] = aij;

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
                edges[M - 1].nodes_in[0] = i;
                edges[M - 1].nodes_in[1] = j;
                
                edges[M - 1].edge_index[0] = nodes[i].edges_in.size();
                edges[M - 1].edge_index[1] = nodes[j].edges_in.size();

                aij = mu + gsl_ran_gaussian(r, sigma);
                if (gsl_rng_uniform_pos(r) < eps){
                    aji = aij;
                }else{
                    aji = mu + gsl_ran_gaussian(r, sigma);
                }
                edges[M - 1].links[0] = aji;
                edges[M - 1].links[1] = aij;

                nodes[i].edges_in.push_back(M - 1);
                nodes[j].edges_in.push_back(M - 1);
                nodes[i].pos_there.push_back(0);
                nodes[j].pos_there.push_back(1);
            }
        }
        fill_except(nodes, edges, M);
        return M;
    }
}


void init_avgs(long M, Tedge *edges, double avn_0){
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            edges[e].cond_av[k] = avn_0;
            edges[e].chi_cav[k] = 0;
            edges[e].var_cav[k] = 1;
        }
    }
}


double field_cav_in(long e, int k, Tedge *edges){
    double sum = 0;
    long edge_neigh;
    int pos_there;
    for (long j = 0; j < edges[e].edges_except[k].size(); j++){
        edge_neigh = edges[e].edges_except[k][j];
        pos_there = edges[e].pos_there[k][j];
        sum += edges[edge_neigh].links[pos_there] * edges[edge_neigh].cond_av[1 - pos_there];
    }
    return 1 - sum;
}


double var_cav_in(long e, int k, Tedge *edges){
    double sum = 0;
    long edge_neigh;
    int pos_there;
    for (long j = 0; j < edges[e].edges_except.size(); j++){
        edge_neigh = edges[e].edges_except[k][j];
        pos_there = edges[e].pos_there[k][j];
        sum += edges[edge_neigh].links[0] * edges[edge_neigh].links[1] * 
               edges[edge_neigh].chi_cav[1 - pos_there];
    }
    return 1.0 / (1 - sum);
}


void comp_fields_cav(long M, Tedge *edges){
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            edges[e].fields_cav[k] = field_cav_in(e, k, edges);
        }
    }
}

void comp_vars_cav(long M, Tedge *edges){
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            edges[e].var_cav[k] = var_cav_in(e, k, edges);
        }
    }
}


double new_averages(long M, Tedge *edges, double tol, int iter, double damping, 
                    double normfactor = 1e-14){
    double delta = 0, delta_av, delta_chi_cav, h, den, av_new, av_new_not_damp, 
           chi_cav_new;

    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            if (edges[e].chi_cav_converged[k]){
                chi_cav_new = edges[e].chi_cav[k];
                if (edges[e].var_cav[k] > 0){
                    edges[e].var_cav_positive[k] = true;
                    h = edges[e].fields_cav[k] * edges[e].var_cav[k];
                    if (h > 0){
                        av_new = damping * h + (1 - damping) * edges[e].cond_av[k];
                    }else {
                        av_new = (1 - damping) * edges[e].cond_av[k];
                    }
                }
            }else if (edges[e].var_cav[k] > 0){
                edges[e].var_cav_positive[k] = true;   
                h = edges[e].fields_cav[k] * edges[e].var_cav[k];
                if (h > 0){
                    av_new = damping * h + (1 - damping) * edges[e].cond_av[k];
                }else if (h < 0){
                    av_new = (1 - damping) * edges[e].cond_av[k];
                }
                chi_cav_new = damping * edges[e].var_cav[k] + (1 - damping) * edges[e].chi_cav[k];
            }else{
                edges[e].var_cav_positive[k] = false;
                h = edges[e].fields_cav[k];
                if (h > 0){
                    av_new = damping * h + (1 - damping) * edges[e].cond_av[k];
                }else if (h < 0){
                    av_new = (1 - damping) * edges[e].cond_av[k];
                }
                chi_cav_new = (1 - damping) * edges[e].chi_cav[k];
            }
            
            if (isnan(av_new) || isinf(av_new) || isnan(chi_cav_new) || isinf(chi_cav_new)){
                cerr << "Error: averages are nan or inf at site e=" << e << "  node=" << edges[e].nodes_in[k] << "   iter=" << iter << endl;
                return sqrt(-1);
            }

            delta_av = fabs(av_new - edges[e].cond_av[k]);
            if (edges[e].var_cav_positive[k]){
                delta_chi_cav = fabs(chi_cav_new - edges[e].chi_cav[k]);
            }else{
                delta_chi_cav = 1;
            }
            
            if (delta_av > delta){
                delta = delta_av;
            }
            if (delta_chi_cav > delta){
                delta = delta_chi_cav;
            }

            if (!edges[e].chi_cav_converged[k] && delta_chi_cav < tol){
                edges[e].chi_cav_converged[k] = true;
            }

            edges[e].cond_av[k] = av_new;
            edges[e].chi_cav[k] = chi_cav_new;

            if (delta_av < tol && delta_chi_cav < tol){
                edges[e].converged[k] = true;
            }else{
                edges[e].converged[k] = false;
            }
        }

    }
    return delta;
}


double field_in(long i, Tnode *nodes, Tedge *edges){
    double sum = 0;
    long e;
    int pos;
    for (long j = 0; j < nodes[i].edges_in.size(); j++){
        e = nodes[i].edges_in[j];
        pos = nodes[i].pos_there[j];
        sum += edges[e].links[pos] * edges[e].cond_av[1 - pos];
    }
    return 1 - sum;
}


double var_in(long i, Tnode *nodes, Tedge *edges){
    double sum = 0;
    long e;
    int pos;
    for (long j = 0; j < nodes[i].edges_in.size(); j++){
        e = nodes[i].edges_in[j];
        pos = nodes[i].pos_there[j];
        sum += edges[e].links[0] * edges[e].links[1] * edges[e].chi_cav[1 - pos];
    }
    return 1.0 / (1 - sum);
}


double average(long N, Tnode *nodes, Tedge *edges, double normfactor = 1e-14){
    double av = 0;
    double h, den;
    for (long i = 0; i < N; i++){
        nodes[i].field = field_in(i, nodes, edges);
        nodes[i].var = var_in(i, nodes, edges);
        if (nodes[i].var > 0){
            h = nodes[i].field * nodes[i].var;
            if (h > 0){
                nodes[i].av = h;
            } else{
                nodes[i].av = 0;
            } 
            nodes[i].chi = nodes[i].var;
        }else{
            h = nodes[i].field;
            if (h > 0){
                nodes[i].av = h;
            }else{
                nodes[i].av = 0;
            }
            nodes[i].chi = nodes[i].var;
        }
        
        av += nodes[i].av;
        
    }
    return av / N;
}

double average_sqr(long N, Tnode *nodes){
    double av_sqr = 0;
    for (long i = 0; i < N; i++){
        av_sqr += nodes[i].av * nodes[i].av;
    }
    return av_sqr / N;
}


double average_chi(long N, Tnode *nodes){
    double av = 0;
    for (long i = 0; i < N; i++){
        av += nodes[i].chi;
    }
    return av / N;
}

double average_chi_sqr(long N, Tnode *nodes){
    double av_sqr = 0;
    for (long i = 0; i < N; i++){
        av_sqr += nodes[i].chi * nodes[i].chi;
    }
    return av_sqr / N;
}


double average_cav(long M, Tedge *edges){
    double av = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            av += edges[e].cond_av[k];
        }
    }
    return av / (2 * M);
}


double average_cav_sqr(long M, Tedge *edges){
    double av_sqr = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            av_sqr += edges[e].cond_av[k] * edges[e].cond_av[k];
        }
    }
    return av_sqr / (2 * M);
}


double average_chi_cav(long M, Tedge *edges){
    double av = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            av += edges[e].chi_cav[k];
        }
    }
    return av / (2 * M);
}


double average_chi_cav_sqr(long M, Tedge *edges){
    double av_sqr = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            av_sqr += edges[e].chi_cav[k] * edges[e].chi_cav[k];
        }
    }
    return av_sqr / (2 * M);
}


int convergence(long M, Tedge *edges, double tol, int max_iter, bool &divergence, 
                double damping, double maximum=1e10, int min_consecutive=5){
    double delta = tol + 1;
    int iter = 0;

    comp_fields_cav(M, edges);
    comp_vars_cav(M, edges);

    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            edges[e].chi_cav_converged[k] = false;
        }
    }

    int consecutive = 0;
    while (consecutive < min_consecutive && iter < max_iter){
        delta = new_averages(M, edges, tol, iter, damping);
        iter++;
        comp_fields_cav(M, edges);
        comp_vars_cav(M, edges);
        if (isinf(delta) || isnan(delta) || delta > maximum){
            divergence = true;
            return iter;
        }
        if (delta < tol){
            consecutive++;
        }else{
            consecutive = 0;
        }
    }
    divergence = false;
    return iter;
}


void print_results(int iter, Tnode *nodes, Tedge *edges, long N, long M, long seed, 
                   int max_iter, bool divergence, double normfactor = 1e-14){
    long counter_diverged = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            if (!edges[e].converged[k]){
                counter_diverged++;
            }
        }
    }

    long counter_varneg = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            if (!edges[e].var_cav_positive[k]){
                counter_varneg++;
            }
        }
    }

    long counter_chi_cav_diverged = 0;
    for (long e = 0; e < M; e++){
        for (int k = 0; k < 2; k++){
            if (!edges[e].chi_cav_converged[k]){
                counter_chi_cav_diverged++;
            }
        }
    }

    double av = average(N, nodes, edges, normfactor);
    double av_sqr = average_sqr(N, nodes);
    double av_chi = average_chi(N, nodes);
    double av_chi_sqr = average_chi_sqr(N, nodes);

    double av_cav = average_cav(M, edges);
    double av_cav_sqr = average_cav_sqr(M, edges);
    double av_chi_cav = average_chi_cav(M, edges);
    double av_chi_cav_sqr = average_chi_cav_sqr(M, edges);
    if (divergence){
        cout << iter << "\t" << "diverges" << "\t" << 
                av_cav << "\t" << sqrt(fabs(av_cav_sqr - av_cav * av_cav) / 2 / M) << "\t" << 
                av_chi_cav << "\t" << sqrt(fabs(av_chi_cav_sqr - av_chi_cav * av_chi_cav) / 2 / M) << "\t" << 
                av << "\t" << sqrt(fabs(av_sqr - av * av) / N) << "\t" << 
                av_chi << "\t" << sqrt(fabs(av_chi_sqr - av_chi * av_chi) / N) << "\t" << 
                counter_diverged << "\t" << counter_varneg << "\t" << counter_chi_cav_diverged << "\t" << seed << endl;
    }else{
        bool conv = iter < max_iter;
        cout << iter << "\t" << conv << "\t" << 
                av_cav << "\t" << sqrt(fabs(av_cav_sqr - av_cav * av_cav) / 2 / M) << "\t" << 
                av_chi_cav << "\t" << sqrt(fabs(av_chi_cav_sqr - av_chi_cav * av_chi_cav) / 2 / M) << "\t" << 
                av << "\t" << sqrt(fabs(av_sqr - av * av) / N) << "\t" << 
                av_chi << "\t" << sqrt(fabs(av_chi_sqr - av_chi * av_chi) / N) << "\t" << 
                counter_diverged << "\t" << counter_varneg << "\t" << counter_chi_cav_diverged << "\t" << seed << endl;
    }

}



int main(int argc, char *argv[]) {
    unsigned long seed = atoi(argv[1]);
    double avn_0 = atof(argv[2]);
    double tol = atof(argv[3]);
    int max_iter = atoi(argv[4]);
    double eps = atof(argv[5]);
    double mu = atof(argv[6]);
    double sigma = atof(argv[7]);
    double damping = atof(argv[8]);
    bool gr_inside = atoi(argv[9]);

    gsl_set_error_handler_off();

    Tnode *nodes;
    Tedge *edges;
    long N, M;

    if (gr_inside){
        N = atol(argv[10]);
        int c = atoi(argv[11]);
        gsl_rng * r;

        init_ran(r, seed);

        M = init_graph_inside_RRG(nodes, edges, N, c, eps, mu, sigma, r);
    }else{
        init_graph_from_input(nodes, edges, N, M);
    }

    init_avgs(M, edges, avn_0);

    bool divergence;

    int iter = convergence(M, edges, tol, max_iter, divergence, damping);

    print_results(iter, nodes, edges, N, M, seed, max_iter, divergence);
    
    return 0;
}