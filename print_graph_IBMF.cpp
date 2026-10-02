// print_graph_IBMF: dump the RRG built by IBMF_LV_sequential (same create_graph routine): one line per node,
// "i  j1 a_ij1  j2 a_ij2 ..." with a_ij the coupling incoming to i from j.
// Usage: print_graph_IBMF.out N mu sigma eps seed_graph
#include "IBMF_common.h"
int main(int argc, char *argv[]) {
    long N = atol(argv[1]); double mu = atof(argv[2]), sigma = atof(argv[3]), eps = atof(argv[4]);
    unsigned long seed = atol(argv[5]);
    Tnode *nodes; char gr_str[200]; char graph_type[10]; sprintf(graph_type, "RRG");
    create_graph(true, seed, N, nodes, eps, mu, sigma, gr_str, graph_type, 3.0, false);
    for (long i = 0; i < N; i++) {
        printf("%ld", i);
        for (size_t k = 0; k < nodes[i].neighs.size(); k++) printf(" %ld %.10f", nodes[i].neighs[k], nodes[i].links_in[k]);
        printf("\n");
    }
    return 0;
}
