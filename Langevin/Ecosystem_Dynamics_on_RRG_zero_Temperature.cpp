/* PREPROCESSOR INCLUSION */

#include "basic_toolbox.h"
#include "ecosystem_numerical_integration_toolbox.h"
#include "generate_graph_from_sequence_toolbox.h"

/* PREPROCESSOR DEFINITION */

/* STRUCTURE */

/* TYPE DEFINITION */

/* PROTOTYPES */

/* GLOBAL VARIABLES */

/* MAIN BEGINNING */

int main(int argc, char **argv){
    if(argc != 10) {
        fprintf(stderr, "Usage: %s N c mu sigma epsilon T N_ext N_prev_ext N_meas\n", argv[0]);
        return MY_FAIL;
    }
    clock_t begin = clock();
    float time_spent;
    char *N_label = argv[1];
    int N = atoi(N_label);
    char *c_label = argv[2];
    double c = atof(c_label);
    char *mu_label = argv[3];
    double mu = atof(mu_label);
    char *sigma_label = argv[4];
    double sigma = atof(sigma_label);
    char *epsilon_label = argv[5];
    double epsilon = atof(epsilon_label);
    char *T_label = argv[6];
    double T = atof(T_label);
    int N_ext = atoi(argv[7]); 
    int N_previous_ext = atoi(argv[8]);
    int N_meas = atoi(argv[9]);
    // Check T==0
    if(T != 0) {
        fprintf(stderr, "ERROR: T should be zero while T=%lg\n", T);
        return MY_FAIL;
    }

    char ia_label[] = "Partially_AsymGauss";
    void (*ia_generator)(double*, double*, double, double, double) = &partyally_asymmetric_gaussian;
    bool sparsity_flag = MY_TRUE;
    double population_factor = 1.0;
    double lambda = pow(10, -6);
    double h_first = pow(10, -2);
    double h_min = pow(10, -6);
    double h_static = pow(10, -3);
    double tolerance = pow(10, -8); // required tolerance in adaptive RK
    double t_max = 4000, t_end;
    double log_factor = 1.1;
    double stationary_threshold = pow(10, -4);
    double distinction_threshold = 0.01;
    double av_extinction1=0, av_extinction2=0;
    bool equilibrated1=MY_FALSE, equilibrated2=MY_FALSE;

    bool graphical, divergence, multi_eq;
    int j, n, effective_meas;
    int kstar, fhs;
    int *rds, *rdsp, *fingerprint;
    int *crossindex = nullptr;
    double *first_eq_pt;
    double dist_max = 0;
    char *name_buffer, *dir_name, *remove_buffer;
    graph* ecosystem;
    RealMatrix interaction_matrix(N, N);
    time_t my_seed;
    FILE *fp, *fp_r, *fp_K, *fp_summary;

    name_buffer = my_char_malloc(CHAR_LENGHT);
    dir_name = my_char_malloc(CHAR_LENGHT);
    remove_buffer = my_char_malloc(CHAR_LENGHT);
        
    snprintf(name_buffer, sizeof(char) * CHAR_LENGHT, "mkdir epsilon_%s_%s_lambda_%g_tol_%g", epsilon_label, ia_label, lambda, tolerance); // sizeof(char)=1 so it can be omitted
    system(name_buffer); 
    ecosystem = ecosystem_initialization(N);
    // Memory Allocation
    rds = my_int_malloc(N);
    rdsp = my_int_calloc(N);
    fingerprint = my_int_malloc(N);
    first_eq_pt = my_double_malloc(N);
    // Creating directory
    snprintf(dir_name, CHAR_LENGHT, "epsilon_%s_%s_lambda_%g_tol_%g/N_%d_c_%.2f", epsilon_label, ia_label, lambda, tolerance, N, c); // sizeof(char)=1 so it can be omitted
    create_bunch_of_directories(dir_name);
    snprintf(name_buffer, CHAR_LENGHT, "mkdir %s/Equilibrium_Points", dir_name);
    system(name_buffer);
    snprintf(name_buffer, sizeof(char) * CHAR_LENGHT, "mkdir Results"); // sizeof(char)=1 so it can be omitted
    system(name_buffer); 
    snprintf(name_buffer, CHAR_LENGHT, "./Results/Lotka-Volterra_epsilon_%s_%s_lambda_%g_tol_%g_N_%d_c_%.2f_mu_%s_sigma_%s_T_%s_Phase_Diagram.txt", epsilon_label, ia_label, lambda, tolerance, ecosystem->size, c, mu_label, sigma_label, T_label);
    fp_summary = my_open_appending_file(name_buffer);
    fprintf(fp_summary, "# Extraction\tNum_Measure\tSeed\tt_end\tt_max\tfirst_equilibrated\tsecond_equilibrated\tav_extinctions1\tav_extinction2\tdivergence\tmulti_equilibria\tdist_max\tcomputation_time\n");        

    // Extract the graph N_ext times and for each of them, evolve the system N_meas times by changing the initial conditions each time
    for(n=(N_previous_ext+1); n<=(N_previous_ext+N_ext); n++){
        seeds_from_dev_random_and_time(&my_seed); // Generate Random Seed
        srand48(my_seed); // The random seed is given
        // Generate the sequence
        graphical = MY_FALSE;
        while(not graphical){
            fill_array(rds, (int) c, N);
            qsort(rds, N, sizeof(int), compare_desc);
            fhs = rds[0]; // First Hub Size
            if(N > fhs){
                crossindex = my_int_realloc(crossindex, fhs+1);
                kstar = build_crossing_index_table_remove_zeros(crossindex, rds, fhs, N);
                graphical = erdos_gallai_test(rds, N, crossindex, kstar);
            }
        }   // We get a graphical degree sequence
        // GENERATE INTERACTION MATRIX
        graph_from_degree_sequence(interaction_matrix, N, rds, rdsp, fingerprint, crossindex, kstar, ia_generator, mu, sigma, epsilon);
        // Save to file in order to generate ecosystem
        snprintf(name_buffer, CHAR_LENGHT, "%s/Extractions/Extraction_%d_Interaction_Matrix_Sparse_mu_%s_sigma_%s_T_%s.txt", dir_name, n, mu_label, sigma_label, T_label);
        fp = my_open_writing_file(name_buffer);
        save_matrix_to_file_sparse_ijaij(interaction_matrix, N, fp);
        fclose(fp);
        // CREATE ECOSYSTEM
        snprintf(name_buffer, CHAR_LENGHT, "Array_of_ones.txt");
        generate_file_of_ones(name_buffer, N);
        fp_r = my_open_reading_file(name_buffer);
        fp_K = my_open_reading_file(name_buffer);
        snprintf(name_buffer, CHAR_LENGHT, "%s/Extractions/Extraction_%d_Interaction_Matrix_Sparse_mu_%s_sigma_%s_T_%s.txt", dir_name, n, mu_label, sigma_label, T_label);
        fp = my_open_reading_file(name_buffer);
        load_ecosystem_from_file(ecosystem, sparsity_flag, fp_r, fp_K, fp);
        fclose(fp);
        fclose(fp_r);
        fclose(fp_K);
        // Reset booleans
        divergence = MY_FALSE;
        multi_eq = MY_FALSE;
        effective_meas = N_meas; // If not interrupted we are going to have N_meas measures
        time_t computation_time = 0;
        // MEASURES
        for(j=1; j<=N_meas; j++){
            // INITIAL CONDITIONS
            snprintf(name_buffer, CHAR_LENGHT, "%s/Initial_Conditions/Log_Initial_Condition_Extraction_%d_PopFactor_%.2f_mu_%s_sigma_%s_T_%s.txt", dir_name, n, population_factor, mu_label, sigma_label, T_label);
            fp = my_open_writing_file(name_buffer);
            extract_and_save_random_initial_conditions(ecosystem, population_factor, fp);
            fclose(fp);
            // Adaptive+Static Runge-Kutta numerical integration + static
            auto chrono_start = std::chrono::high_resolution_clock::now();
            log_RungeKutta_adaptive_mixed_static_driver_for_phase_diagram(ecosystem, t_max, &t_end, h_first, h_min, h_static, log_factor, j, &divergence, &equilibrated2, &av_extinction2, lambda, tolerance, stationary_threshold);
            auto chrono_end = std::chrono::high_resolution_clock::now();
            computation_time += std::chrono::duration_cast<std::chrono::milliseconds>(chrono_end - chrono_start).count();
            //printf("Extraction %d Measure %d \n", n, j);
            if(j==1){ // We save the first equilibrium configuration
                copy_abundances_into_array(ecosystem, first_eq_pt);
                equilibrated1 = equilibrated2;
                av_extinction1 = av_extinction2;
            }
            else{ // Starting from the second measure we check whether we have multiple equilibria or not
                if(!are_abundances_in_array_and_take_max(ecosystem, first_eq_pt, distinction_threshold, &dist_max))
                    multi_eq = MY_TRUE;
            }
            // If divergence or multi_eq exit, otherwise keep going
            if((divergence||multi_eq) == MY_TRUE){
                effective_meas = j;
                j = N_meas+2;
            }
        }
        computation_time /= effective_meas;
        fprintf(fp_summary, "%d\t%d\t%ld\t%.17f\t%.17f\t%d\t%d\t%lf\t%lf\t%d\t%d\t%lf\t%ld\n", n, effective_meas, my_seed, t_end, t_max, equilibrated1, equilibrated2, av_extinction1, av_extinction2, divergence, multi_eq, dist_max, computation_time);
        fflush(fp_summary);
        // CLEANING
        clean_up_graph(ecosystem);
        snprintf(name_buffer, CHAR_LENGHT, "%s/Initial_Conditions/Log_Initial_Condition_Extraction_%d_PopFactor_%.2f_mu_%s_sigma_%s_T_%s.txt", dir_name, n, population_factor, mu_label, sigma_label, T_label);
        snprintf(remove_buffer, CHAR_LENGHT, "rm %s", name_buffer);
        system(remove_buffer);
        snprintf(name_buffer, CHAR_LENGHT, "%s/Extractions/Extraction_%d_Interaction_Matrix_Sparse_mu_%s_sigma_%s_T_%s.txt", dir_name, n, mu_label, sigma_label, T_label);
        snprintf(remove_buffer, CHAR_LENGHT, "rm %s", name_buffer);
        system(remove_buffer);
    }
    fclose(fp_summary);
    // CLEANING MEMORY
    erase_ecosystem(ecosystem);
    free(first_eq_pt);
    free(rds);
    free(rdsp);
    free(fingerprint);
    free(name_buffer); 
    free(dir_name); 
    free(remove_buffer);
    clock_t end = clock();
    time_spent = ((float)(end - begin)) / CLOCKS_PER_SEC;

    return MY_SUCCESS;
}

/* MAIN - THE END */

/* FUNCTIONS */