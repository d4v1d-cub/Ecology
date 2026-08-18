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
        fprintf(stderr,
                "Usage: %s N c mu sigma epsilon T N_ext N_prev_ext N_meas\n",
                argv[0]);
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

    char ia_label[] = "Partially_AsymGauss";
    void (*ia_generator)(double*, double*, double, double, double) = &partyally_asymmetric_gaussian;
    void (*save_lv_function)(graph*, double, FILE*) = &save_LotkaVolterra_trajectories_only; // &save_LotkaVolterra_trajectories_only // &save_LotkaVolterra_nothing

    bool sparsity_flag = MY_TRUE;
    double population_factor = 1.0;
    double lambda = 1e-06;
    double h = 1e-2;
    double t_max = 4000;
    double deltat_save = 2.0;
    double log_factor = 1.1;

    int *rds, *rdsp, *fingerprint;
    int *crossindex = NULL;
    int kstar, fhs;
    RealMatrix interaction_matrix(N, N);

    bool graphical, divergence;
    int j, n;  
    int divergence_counter;
    char *name_buffer, *dir_name, *remove_buffer;
    graph* ecosystem;
    time_t my_seed;
    FILE *fp, *fp_RK, *fp_eq, *fp_r, *fp_K, *fp_summary; 

    printf("Stiamo eseguendo: %s\n", argv[0]);
    printf("Orario di inizio: \n");
    system("date");
    name_buffer = my_char_malloc(CHAR_LENGHT);
    dir_name = my_char_malloc(CHAR_LENGHT);
    remove_buffer = my_char_malloc(CHAR_LENGHT);
        
    // Creating directory mu, sigma, epsilon and T
    snprintf(name_buffer, sizeof(char) * CHAR_LENGHT, "mkdir epsilon_%s_%s_lambda_%g_h_%g", epsilon_label, ia_label, lambda, h); // sizeof(char)=1 so it can be omitted
    system(name_buffer); 
    // Init Ecosystem
    ecosystem = ecosystem_initialization(N);
    snprintf(dir_name, CHAR_LENGHT, "epsilon_%s_%s_lambda_%g_h_%g/N_%d_c_%.2f", epsilon_label, ia_label, lambda, h, ecosystem->size, c); // sizeof(char)=1 so it can be omitted
    create_bunch_of_directories(dir_name);
    snprintf(name_buffer, CHAR_LENGHT, "mkdir %s/Equilibrium_Points", dir_name);
    system(name_buffer);
    // Memory Allocation
    rds = my_int_malloc(N);
    rdsp = my_int_calloc(N);
    fingerprint = my_int_malloc(N);
    divergence_counter = 0;
    snprintf(name_buffer, sizeof(char) * CHAR_LENGHT, "mkdir Results"); // sizeof(char)=1 so it can be omitted
    system(name_buffer); 
    snprintf(name_buffer, CHAR_LENGHT, "./Results/Lotka-Volterra_epsilon_%s_%s_lambda_%g_h_%g_N_%d_c_%.2f_mu_%s_sigma_%s_T_%s_Equilibrium_Points.txt", epsilon_label, ia_label, lambda, h, ecosystem->size, c, mu_label, sigma_label, T_label);
    fp_summary = my_open_appending_file(name_buffer);
    fprintf(fp_summary, "# Extraction\tMeasure\tSeed\ttime\tNum_NonEq\tav(#extinctions)\tav(<n_i>)\tstd(<n_i>)\tav(<(n_i - <n_i>)^{2}>)\tstd(<(n_i - <n_i>)^{2}>)\tcomputation time(ms)\n");        
    
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
        // MEASURES
        for(j=1; j<=N_meas; j++){
            if(j!=1){ // Use a different seed for measures after the first one
                seeds_from_dev_random_and_time(&my_seed); // Generate Random Seed
                srand48(my_seed); // The random seed is given
            }
            // INITIAL CONDITIONS
            snprintf(name_buffer, CHAR_LENGHT, "%s/Initial_Conditions/Log_Initial_Condition_Extraction_%d_PopFactor_%.2f_mu_%s_sigma_%s_T_%s.txt", dir_name, n, population_factor, mu_label, sigma_label, T_label);
            fp = my_open_writing_file(name_buffer);
            extract_and_save_random_initial_conditions(ecosystem, population_factor, fp);
            fclose(fp);
            snprintf(name_buffer, CHAR_LENGHT, "%s/Evolutions/Lotka-Volterra_Extraction_%d_Measure_%d_mu_%s_sigma_%s_T_%s.txt", dir_name, n, j, mu_label, sigma_label, T_label);
            fp_RK = my_open_writing_file(name_buffer);
            snprintf(name_buffer, CHAR_LENGHT, "%s/Equilibrium_Points/Lotka-Volterra_mu_%s_sigma_%s_T_%s_Extraction_%d_Measure_%d_Equilibrium_Points.txt", dir_name, mu_label, sigma_label, T_label, n, j);
            fp_eq = my_open_writing_file(name_buffer);
            // NUMERICAL INTEGRAION
            Milstein_driver_GLV_demographic_noise_with_single_species_measure_and_summary(ecosystem, t_max, h, log_factor, j, &divergence, T, lambda, deltat_save, save_lv_function, fp_summary, fp_RK, fp_eq, my_seed, n);
            fflush(fp_RK);
            fclose(fp_RK);
            fflush(fp_eq);
            fclose(fp_eq);
            fflush(fp_summary);
            printf("Extraction %d Measure %d \n", n, j);
            if(divergence == MY_TRUE){
                divergence_counter++;
            }
        } 
        // Pulizia interazioni grafo
        clean_up_graph(ecosystem);
        // Cleaning files
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
    free(rds);
    free(rdsp);
    free(fingerprint);
    free(name_buffer);
    free(dir_name);
    free(remove_buffer);
    clock_t end = clock(); 
    time_spent = ((float)(end - begin)) / CLOCKS_PER_SEC;
    printf("\n # Il programma ha impiegato %f secondi.\n # A breve dovrebbe arrivare la mail (se opportunamente richiesto presso i nostri uffici).\n \n", time_spent);
    printf("Orario di conclusione: \n");
    system("date");

    return MY_SUCCESS;
}

/* MAIN - THE END */

/* FUNCTIONS */