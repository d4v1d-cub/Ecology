#ifndef _ECOSYSTEM_NUMERICAL_INTEGRATION_TOOLBOX_H_ // Include Guard

    #define _ECOSYSTEM_NUMERICAL_INTEGRATION_TOOLBOX_H_ 

    /* PREPROCESSOR INCLUSION */
    #include "basic_toolbox.h"
    #include <chrono>

    /* PREPROCESSOR DEFINITION */
    // Cask-Karp
    #define CHECK_PERIOD  20
    #define MAX_ADAPTIVE_STEPS 1.E+8
    #define SAFETY 0.9
    #define PGROW -0.2
    #define PSHRNK -0.25
    #define ERRCON 1.89e-4 // The value ERRCON equals (5/SAFETY) raised to the power (1/PGROW), see use in source file.
    // Milstein
    #define HARD_WALL 0.0
    #define NUM_THERM_EQ_INTERVALS 3

    /* STRUCTURE */
    struct vertex_struct{
        double x, logN, r, K;
        int degree;
        struct vertex_struct** connection;
        double* strength;
        int label;
        char status; // 'I' = isolated, 'J' = isolated from evolution, 'E' = extincted, 'S' = survived
    };

    struct graph_struct{
        int size;
        struct vertex_struct** vtx;
        int isolated, extincted, isolated_from_evolution, survived;
    };

    /* TYPE DEFINITION */
    typedef struct vertex_struct vertex;
    typedef struct graph_struct graph;
    typedef double data;

    /* PROTOTYPES */
    // Ecosystem toolbox
    graph* ecosystem_initialization(int S);
    void graph_extraction(graph *ecosystem, double p_legame, double p_A, double p_M, void (*ia_generator)(double*, double*, double, double, double, double, double, double), double ia_p1, double ia_p2, double ia_p3, double ia_p4, double (*gr_generator)(double, double), double gr_p1, double gr_p2, double (*cc_generator)(double, double), double cc_p1, double cc_p2);
    void directed_ER_ecosystem_extraction(graph *ecosystem, double p_legame, /*double p_directed_int,*/ double (*ia_generator)(double, double), double ia_p1, double ia_p2, double (*gr_generator)(double, double), double gr_p1, double gr_p2, double (*cc_generator)(double, double), double cc_p1, double cc_p2);
    void symmetric_gaussian_extraction(double *a_ij, double *a_ji, double mu, double sigma, double mu2, double sigma2, double p_A, double p_M);
    void partyally_asymmetric_gaussian(double* a_ij, double* a_ji, double mu, double sigma, double epsilon);
    void save_dense_interaction_matrix_to_file(graph *ecosystem, FILE *fp_matrix);
    void save_sparse_interaction_matrix_to_file(graph *ecosystem, FILE *fp_matrix);
    void save_ecosystem_to_file(graph *ecosystem, bool sparsity_flag, FILE *fp_r, FILE *fp_K, FILE *fp_matrix);
    void load_matrix_from_file(graph *ecosystem, int S, FILE *fp_matrix_file);
    void load_sparse_matrix_from_file(graph *ecosystem, FILE *fp_matrix_file);
    void load_ecosystem_from_file(graph *ecosystem, bool sparsity_flag, FILE *fp_r, FILE *fp_K, FILE *fp_matrix);
    
    void swap_vertex(graph *ecosystem, int i, int j);
    void vertex_realloc(graph *ecosystem, int vertex_index);
    void vertex_rearrangement(graph *ecosystem, int vertex_index);
    void clean_up_graph(graph *ecosystem);
    void erase_ecosystem(graph *ecosystem);

    void status_set_up_new_measure(graph *ecosystem);
    void extract_and_save_random_initial_conditions(graph *ecosystem, double population_factor, FILE *fp_initial_conditions);
    void extract_and_save_delta_initial_conditions(graph *ecosystem, double population_factor, FILE *fp_initial_conditions);
    void load_initial_conditions(graph *ecosystem, FILE *fp_initial_conditions);
    void load_initial_conditions_from_equilibrium_point(graph *ecosystem, FILE *fp_initial_conditions);
    void generate_file_of_ones(char *file_name, int N);

    void copy_abundances_into_array(graph *ecosystem, double* x);
    bool are_abundances_in_array(graph *ecosystem, double *y, double delta);
    bool are_abundances_in_matrix(graph *ecosystem, double **matrix, int first_dim, double delta, int *index_pointer);
    bool are_abundances_in_array_and_take_max(graph *ecosystem, double *y, double delta, double *max_dist);
    void create_bunch_of_directories(char* dir_name);

    // Numerical integration
    void GLV_equation(double* derivative_func, graph* ecosystem, double lambda);
    void log_LV_gen_eq(double* log_derivative_func, graph* ecosystem, double lambda);
    
    void rkck_algorithm(graph* ecosystem, double **k, double t, double h, double *logN_out, double *logN_lte, double lambda);
    void rk5_quality_control_stepper(graph* ecosystem, double **k, double *t_adr, double h_try, double *h_did, double *h_next, double eps, double *logN_scal, double lambda);
    void log_RungeKutta_adaptive_onestep(graph* ecosystem, double **k, double *t_adr, double lambda, double eps, double *h_try, bool *stepsize_trouble_adr);
    void log_RungeKutta_adaptive_driver(graph* ecosystem, double t_max, double h_first, double h_min, double eps, double threshold, int measure_counter, bool *divergence_adr, bool *stepsize_trouble_adr, int *good, int *bad, double lambda, double deltat_save, void (*save_lv_function)(graph*, double**, double, double*, FILE*, FILE*), FILE* fp_RK, FILE* fp_speed);
    void rk4_algorithm(graph* ecosystem, double **k, double *t_adr, double h, double lambda);
    void log_RungeKutta_adaptive_mixed_static_driver_for_phase_diagram(graph* ecosystem, double t_max, double *t_end, double h_first, double h_min, double h_static, double log_factor, int measure_num, bool *divergence_adr, bool *equilibrated_adr, double *av_extinction_adr, double lambda, double tolerance, double equilibrium_threshold);

    void Milstein_onestep_GLV_demographic_noise(graph* ecosystem, double *k, double *t_adr, double h, double T, double lambda);
    void Milstein_driver_GLV_demographic_noise_with_single_species_measure(graph* ecosystem, double t_max, double h, double log_factor, int measure_num, bool *divergence_adr, double T, double lambda, double deltat_save, void (*save_lv_function)(graph*, double, FILE*), FILE* fp_RK, FILE* fp_eq);
    void Milstein_driver_GLV_demographic_noise_with_single_species_measure_and_summary(graph* ecosystem, double t_max, double h, double log_factor, int measure_num, bool *divergence_adr, double T, double lambda, double deltat_save, void (*save_lv_function)(graph*, double, FILE*), FILE* fp_summary, FILE* fp_RK, FILE* fp_eq, time_t my_seed, int ext_num);
    void Milstein_driver_GLV_demographic_noise_only_with_final_summary(graph* ecosystem, double t_max, double h, double log_factor, int measure_num, bool *divergence_adr, double T, double lambda, double deltat_divergence, FILE *fp_summary, time_t my_seed, int ext_num);
    
    void check_ecosystem_divergence(graph* ecosystem, bool *divergence_adr);
    void check_ecosystem_equilibrium_zero_temperature(double *derivative, int N, double equilibrium_threshold, bool *equilibrium_reached);
    void check_ecosystem_thermal_equilibrium(graph* ecosystem, int *equilibrium_counter, char *species_convergence);

    void save_LotkaVolterra_trajectories_only(graph* ecosystem, double t, FILE* fp_RK);
    void save_LotkaVolterra_nothing(graph* ecosystem, double t, FILE* fp_RK);

#endif // _GRAPH_LV_TOOLBOX_H_