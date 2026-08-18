#ifndef _GENERATE_GRAPH_FROM_SEQUENCE_TOOLBOX_H_ // Include Guard

    #define _GENERATE_GRAPH_FROM_SEQUENCE_TOOLBOX_H_ 

    /* PREPROCESSOR INCLUSION */
    #include <complex>
    #include "basic_toolbox.h"
    #include "./eigen-3.4.0/Eigen/Dense"


   /* PREPROCESSOR DEFINITION */

    /* STRUCTURE */


    /* TYPE DEFINITION */
    using Real = float; // This is equivalent to: typedef double real;
    using RealVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using RealMatrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>; // This is equivalent to: typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> RealMatrix;

    /* PROTOTYPES */

    void save_matrix_to_file_sparse_ijaij(RealMatrix& graph, int N, FILE *fp_matrix);
    void save_matrix_to_file_sparse_ijaijaji(RealMatrix& graph, int N, FILE *fp_matrix);
    void load_matrix_to_file_sparse_ijaij(RealMatrix& graph, int N, FILE *fp_matrix);
    void load_matrix_from_file_sparse_ij_aijaji(RealMatrix& graph, int N, FILE *fp_matrix); // Matches save_matrix_to_file_sparse_ijaijaji's format: reads i, j, then graph(i,j), then graph(j,i)
    void load_matrix_from_file_sparse_ij_ajiaij(RealMatrix& graph, int N, FILE *fp_matrix); // Same format, but the two values are swapped: reads i, j, then graph(j,i), then graph(i,j)

    int build_crossing_index_table(int *crossindex, int *deg_seq, int fhs, int N);
    int build_crossing_index_table_remove_zeros(int *crossindex, int *deg_seq, int fhs, int N);
    int build_crossing_index_table_up_kstar(int *crossindex, int *deg_seq, int fhs, int N);
    void build_crossing_index_table_no_kstar(int *crossindex, int *deg_seq, int fhs, int N);
    bool erdos_gallai_test(int *deg_seq, int N, int *crossindex, int kstar);
    void build_dprime(int *rdsp, int *cip, int *ksp, int *fnpc, int *rds, int hs, int fhs, int *crossindex, int kstar);
    void update_rds_connection(int *rds, int *crossindex, int *kstar, int *anc, int *fnc, int *fingerprint, int q, int hs);
    void update_rdsp_connection(int *rdsp, int *cip, int *ksp, int *fnpc, int srd, int dqm1, int hs);
    void update_rdsp_last_connection(int *rdsp, int *cip, int *ksp, int *fnpc, int dqm1, int hs);
    void update_rds_newhub(int *rds, int *crossindex, int *kstar, int *anc, int *fnc, int *hs, int fhs);
    void update_rdsp_newhub(int *rdsp, int *cip, int *ksp, int *fnpc, int hs, int fhs);
    void fail_degree_k(int L, int R, int *fdk, int *k, int ridx, int *rdsp, int *cip, int ciphs, int ksp, int *fnpc);
    int find_maximum_fail_degree(int *rdsp, int *cip, int ciphs, int ksp, int *fnpc);
    void build_allowed_nodes_counter(int *anc, int *crossindex, int *fnc, int hs);
    void build_allowed_nodes_counter_at_beginning(int *anc, int *crossindex, int hs);
    void extract_allowed_node(int *eni, int *tnan, int *crossindex, int hs, int *anc, int *fnc, int mfd);
    int find_smallest_reduced_degree(int *fnc, int *fnpc);
    void graph_from_degree_sequence(RealMatrix& graph, int N, int *rds, int *rdsp, int *fingerprint, int *crossindex, int kstar, void (*weights_generator)(double*, double*, double, double, double), double p1, double p2, double epsilon);

#endif // _GRAPH_LV_TOOLBOX_H_