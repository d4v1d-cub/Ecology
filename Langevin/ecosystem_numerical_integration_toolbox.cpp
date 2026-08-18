#include "ecosystem_numerical_integration_toolbox.h"

// Inizializzazione del grafo sparso
graph* ecosystem_initialization(int S){
    graph* ecosystem;
    int i;
    // Ecosystem Memory Allocation
    ecosystem = (graph*) malloc(1 * sizeof(graph));
    if (ecosystem == NULL){
        printf("\n ERROR: ECOSYSTEM MALLOC HAS FAILED \n");
        exit(MY_MEMORY_FAIL);
    }
    // Ecosystem fill in 
    ecosystem->size = S;
    ecosystem->vtx = (vertex**) malloc(ecosystem->size * sizeof(vertex*));
    if (ecosystem->vtx == NULL){
        printf("\n ERROR: ECOSYSTEM VERTEX POINTERS MALLOC HAS FAILED \n");
        exit(MY_MEMORY_FAIL);
    }
    for(i=0; i<ecosystem->size; i++){
        *(ecosystem->vtx+i) = (vertex*) malloc(1 * sizeof(vertex));
        if ( *(ecosystem->vtx+i) == NULL ){
            printf("\n ERROR: ECOSYSTEM VERTEX MALLOC HAS FAILED \n");
            exit(MY_MEMORY_FAIL);
        }
        (*(ecosystem->vtx+i))->label = i;
    }
    return ecosystem;
}

void graph_extraction(graph *ecosystem, double p_legame, double p_A, double p_M, void (*ia_generator)(double*, double*, double, double, double, double, double, double), double ia_p1, double ia_p2, double ia_p3, double ia_p4, double (*gr_generator)(double, double), double gr_p1, double gr_p2, double (*cc_generator)(double, double), double cc_p1, double cc_p2){
    int i, j;
    double a_ij, a_ji;
    int scartati = 0;
    
    ecosystem->isolated = 0;
    ecosystem->extincted = 0;
    ecosystem->isolated_from_evolution = 0;
    ecosystem->survived = ecosystem->size;

    for(i=0; i<ecosystem->size; i++){
        (*(ecosystem->vtx+i))->degree = 0;
        (*(ecosystem->vtx+i))->status = 'S';
        // Extract growth rate and carrying capacity
        (*(ecosystem->vtx+i))->r = gr_generator(gr_p1, gr_p2);
        (*(ecosystem->vtx+i))->K = cc_generator(cc_p1, cc_p2);
        // Init connections
        (*(ecosystem->vtx+i))->connection = (vertex**) malloc(ecosystem->size * sizeof(vertex*));
        if ( (*(ecosystem->vtx+i))->connection == NULL ){
            printf("\n ERROR: ECOSYSTEM VTX CONNECTION MALLOC HAS FAILED \n");
            exit(MY_MEMORY_FAIL);
        }
        // Init strenghts
        (*(ecosystem->vtx+i))->strength = (double*) malloc(ecosystem->size * sizeof(double));
        if ( (*(ecosystem->vtx+i))->strength == NULL ){
            printf("\n ERROR: ECOSYSTEM VTX STRENGHT MALLOC HAS FAILED \n");
            exit(MY_MEMORY_FAIL);
        }
    }
    // Extract interactions
    for(i=ecosystem->size-1; i>=0; i--){
        for(j=0; j<i-1; j++){ 
            if(RNG_0() <= p_legame){
                // Connect
                *( ( *(ecosystem->vtx+i) )->connection + ( *(ecosystem->vtx+i) )->degree ) = ( *(ecosystem->vtx+j) ); // ecosystem->vtx è un array di puntatotori dunque *(ecosystem->vtx+j) è l'elemento j ovvero il puntatore alla memoria del vertice j (e non il vertice j)
                *( ( *(ecosystem->vtx+j) )->connection + ( *(ecosystem->vtx+j) )->degree ) = ( *(ecosystem->vtx+i) );
                // Extract strenghts
                ia_generator(&a_ij, &a_ji, ia_p1, ia_p2, ia_p3, ia_p4, p_A, p_M); // Interactions extraction is not optimal but it increases readibility and flexibility
                *( ( *(ecosystem->vtx+i) )->strength + ( *(ecosystem->vtx+i) )->degree ) = a_ij;
                *( ( *(ecosystem->vtx+j) )->strength + ( *(ecosystem->vtx+j) )->degree ) = a_ji;
                // Increment degrees
                ( *(ecosystem->vtx+i) )->degree ++; 
                ( *(ecosystem->vtx+j) )->degree ++;
            }
            else {
                scartati++;
            }
        }
        vertex_rearrangement(ecosystem, i);
    }
    scartati = scartati * 2;
    return;
}

void directed_ER_ecosystem_extraction(graph *ecosystem, double p_legame, double (*ia_generator)(double, double), double ia_p1, double ia_p2, double (*gr_generator)(double, double), double gr_p1, double gr_p2, double (*cc_generator)(double, double), double cc_p1, double cc_p2){
    int i, j;
    double a_ij;
    
    ecosystem->isolated = 0;
    ecosystem->extincted = 0;
    ecosystem->isolated_from_evolution = 0;
    ecosystem->survived = ecosystem->size;

    for(i=0; i<ecosystem->size; i++){
        (*(ecosystem->vtx+i))->degree = 0;
        (*(ecosystem->vtx+i))->status = 'S';
        (*(ecosystem->vtx+i))->r = gr_generator(gr_p1, gr_p2);
        (*(ecosystem->vtx+i))->K = cc_generator(cc_p1, cc_p2);
        (*(ecosystem->vtx+i))->connection = (vertex**) malloc(ecosystem->size * sizeof(vertex*));
        if ( (*(ecosystem->vtx+i))->connection == NULL ){
            printf("\n ERROR: ECOSYSTEM VTX CONNECTION MALLOC HAS FAILED \n");
            exit(MY_MEMORY_FAIL);
        }
        (*(ecosystem->vtx+i))->strength = (double*) malloc(ecosystem->size * sizeof(double));
        if ( (*(ecosystem->vtx+i))->strength == NULL ){
            printf("\n ERROR: ECOSYSTEM VTX STRENGHT MALLOC HAS FAILED \n");
            exit(MY_MEMORY_FAIL);
        }
    }
    for(i=ecosystem->size-1; i>=0; i--){ // The start from the end because we put the isolated nodes at the end
        for(j=0; j<i; j++){ // First part of the interactions
            if(RNG_0() <= p_legame){
                *( ( *(ecosystem->vtx+i) )->connection + ( *(ecosystem->vtx+i) )->degree ) = ( *(ecosystem->vtx+j) ); // ecosystem->vtx è un array di puntatotori dunque *(ecosystem->vtx+j) è l'elemento j ovvero il puntatore alla memoria del vertice j (e non il vertice j)
                a_ij = ia_generator(ia_p1, ia_p2); // Generator for directed, independent interactions
                *( ( *(ecosystem->vtx+i) )->strength + ( *(ecosystem->vtx+i) )->degree ) = a_ij;
                ( *(ecosystem->vtx+i) )->degree ++; 
            }
        }
        for(j=i+1; j<ecosystem->size; j++){ // Second part of the interactions
            if(RNG_0() <= p_legame){
                *( ( *(ecosystem->vtx+i) )->connection + ( *(ecosystem->vtx+i) )->degree ) = ( *(ecosystem->vtx+j) ); // ecosystem->vtx è un array di puntatotori dunque *(ecosystem->vtx+j) è l'elemento j ovvero il puntatore alla memoria del vertice j (e non il vertice j)
                a_ij = ia_generator(ia_p1, ia_p2); // Generator for directed, independent interactions
                *( ( *(ecosystem->vtx+i) )->strength + ( *(ecosystem->vtx+i) )->degree ) = a_ij;
                ( *(ecosystem->vtx+i) )->degree ++; 
            }
        }
        vertex_rearrangement(ecosystem, i);
    }
    return;
}

void symmetric_gaussian_extraction(double *a_ij, double *a_ji, double mu, double sigma, double mu2, double sigma2, double p_A, double p_M){
    *a_ij = GNG(mu, sigma);
    *a_ji = *a_ij;
    return;
}

void partyally_asymmetric_gaussian(double* a_ij, double* a_ji, double mu, double sigma, double epsilon){
    *a_ij = GNG(mu, sigma);
    if(RNG_0() <= epsilon)
        *a_ji = *a_ij;
    else
        *a_ji = GNG(mu, sigma);
    return;
}

void save_dense_interaction_matrix_to_file(graph *ecosystem, FILE *fp_matrix){
    int i, j;
    matrix *alpha;
    alpha = my_double_matrix_calloc(ecosystem->size, ecosystem->size);
    for(i=0; i<ecosystem->size; i++){
        for(j=0; j<( (*(ecosystem->vtx+i))->degree ); j++){
            alpha->element[(*(ecosystem->vtx+i))->label][(*( (*(ecosystem->vtx+i))->connection + j ))->label] = *( (*(ecosystem->vtx+i))->strength + j ); // Un po' di solita confusione con le struct ma ragionandoci ci arrivi
        }
    }
    for(i=0; i<ecosystem->size; i++){
        for(j=0; j<ecosystem->size; j++){
            fprintf(fp_matrix, "%.17f\t", alpha->element[i][j]);
        }
        fprintf(fp_matrix, "\n");
    }
    my_free_matrix(alpha);
    return;
}

void save_sparse_interaction_matrix_to_file(graph *ecosystem, FILE *fp_matrix){
    int i, j;
    for(i=0; i<ecosystem->size; i++){ 
      for(j=0; j<(*(ecosystem->vtx+i))->degree; j++){
        fprintf(fp_matrix, "%d\t%d\t%.17f\n", (*(ecosystem->vtx+i))->label, (*( (*(ecosystem->vtx+i))->connection + j ))->label, *( (*(ecosystem->vtx+i))->strength + j ) ); 
      }
    }
    return;
}

void save_ecosystem_to_file(graph *ecosystem, bool sparsity_flag, FILE *fp_r, FILE *fp_K, FILE *fp_matrix){
    int i;
    for(i=0; i<ecosystem->size; i++){
        fprintf(fp_r, "%.17f\n", (*(ecosystem->vtx+i))->r);
        fprintf(fp_K, "%.17f\n", (*(ecosystem->vtx+i))->K);
    }
    if(sparsity_flag == MY_TRUE){
        save_sparse_interaction_matrix_to_file(ecosystem, fp_matrix);
    }
    else{
        save_dense_interaction_matrix_to_file(ecosystem, fp_matrix);
    }    
    return;
}

void load_dense_interaction_matrix_from_file(graph *ecosystem, FILE *fp_matrix){
    int i, j;
    double r_strength;
    for(i=0; i<ecosystem->size; i++){
        for(j=0; j<ecosystem->size; j++){
            fscanf(fp_matrix, "%lf", &r_strength);
            if(r_strength!=0){
                *( ( *(ecosystem->vtx+i) )->connection + ( *(ecosystem->vtx+i) )->degree ) = ( *(ecosystem->vtx+j) );
                *( ( *(ecosystem->vtx+i) )->strength + ( *(ecosystem->vtx+i) )->degree ) = r_strength;
                ( *(ecosystem->vtx+i) )->degree += 1;
            }
        }
    }
    for(i=ecosystem->size-1; i>=0; i--){
        vertex_rearrangement(ecosystem, i);
    }
    return;
}

void load_sparse_interaction_matrix_from_file(graph *ecosystem, FILE *fp_matrix){
    int i, j;
    double r_strength; 
    while(fscanf(fp_matrix, "%d\t%d\t%lf", &i, &j, &r_strength) > 0){
        *( ( *(ecosystem->vtx+i) )->connection + ( *(ecosystem->vtx+i) )->degree ) = ( *(ecosystem->vtx+j) );
        *( ( *(ecosystem->vtx+i) )->strength + ( *(ecosystem->vtx+i) )->degree ) = r_strength;
        ( *(ecosystem->vtx+i) )->degree += 1;
    }
    for(i=ecosystem->size-1; i>=0; i--){
        vertex_rearrangement(ecosystem, i);
    }
    return;
}

void load_ecosystem_from_file(graph *ecosystem, bool sparsity_flag, FILE *fp_r, FILE *fp_K, FILE *fp_matrix){
    int i;
    ecosystem->isolated = 0;
    ecosystem->extincted = 0;
    ecosystem->isolated_from_evolution = 0;
    ecosystem->survived = ecosystem->size;
    for(i=0; i<ecosystem->size; i++){
        (*(ecosystem->vtx+i))->degree = 0;
        (*(ecosystem->vtx+i))->status = 'S';
        fscanf(fp_r, "%lf", &((*(ecosystem->vtx+i))->r));
        fscanf(fp_K, "%lf", &((*(ecosystem->vtx+i))->K));
        (*(ecosystem->vtx+i))->connection = (vertex**) malloc(ecosystem->size * sizeof(vertex*));
        if ((*(ecosystem->vtx+i))->connection == NULL){
            printf("\n ERROR: ECOSYSTEM CONNECTION POINTERS MALLOC HAS FAILED \n");
            exit(MY_MEMORY_FAIL);
        }
        (*(ecosystem->vtx+i))->strength = (double*) malloc(ecosystem->size * sizeof(double));
        if ((*(ecosystem->vtx+i))->strength == NULL){
            printf("\n ERROR: ECOSYSTEM STRENGTH POINTERS MALLOC HAS FAILED \n");
            exit(MY_MEMORY_FAIL);
        }
    }
    if(sparsity_flag == MY_TRUE){
        load_sparse_interaction_matrix_from_file(ecosystem, fp_matrix);
    }
    else{
        load_dense_interaction_matrix_from_file(ecosystem, fp_matrix);
    }    
    return;
}

void swap_vertex(graph *ecosystem, int i, int j){
    vertex* tmp_vtx_ptr;
    tmp_vtx_ptr = *(ecosystem->vtx + i);
    *(ecosystem->vtx + i) = *(ecosystem->vtx + j);
    *(ecosystem->vtx + j) = tmp_vtx_ptr; 
    ( *(ecosystem->vtx + i) )->label = i;
    ( *(ecosystem->vtx + j) )->label = j;
    return;
}

void vertex_realloc(graph *ecosystem, int vertex_index){
    vertex** appoggio_connection;
    double* appoggio_strength;
    if(( *(ecosystem->vtx + vertex_index) )->degree == 0){
        free((*(ecosystem->vtx + vertex_index))->connection);
        free((*(ecosystem->vtx + vertex_index))->strength);
    } else{
        appoggio_connection = (vertex**) realloc( (*(ecosystem->vtx + vertex_index))->connection, (*(ecosystem->vtx + vertex_index))->degree * sizeof(vertex*) );
        if (appoggio_connection == NULL){
            printf("\n ERROR: CONNECTION REALLOC HAS FAILED \n");
            free((*(ecosystem->vtx + vertex_index))->connection);
            exit(MY_MEMORY_FAIL);
        } else {
            (*(ecosystem->vtx + vertex_index))->connection = appoggio_connection;
        }

        appoggio_strength = (double*) realloc( (*(ecosystem->vtx + vertex_index))->strength, (*(ecosystem->vtx + vertex_index))->degree * sizeof(double) );
        if (appoggio_strength == NULL){
            printf("\n ERROR: STRENGTH REALLOC HAS FAILED \n");
            free((*(ecosystem->vtx + vertex_index))->strength);
            exit(MY_MEMORY_FAIL);
        } else {
            (*(ecosystem->vtx + vertex_index))->strength = appoggio_strength;
        }
    }
    return;
}

void vertex_rearrangement(graph *ecosystem, int vertex_index){
    int btm_idx; // Bottom index
    vertex* tmp_vtx_ptr;
    vertex** tmp_connection;
    double* tmp_strength;
    if(( *(ecosystem->vtx + vertex_index) )->degree == 0){
        free((*(ecosystem->vtx + vertex_index))->connection);
        free((*(ecosystem->vtx + vertex_index))->strength);
        ( *(ecosystem->vtx + vertex_index) )->status = 'I';
        btm_idx = ecosystem->survived - 1;
        tmp_vtx_ptr = *(ecosystem->vtx + vertex_index);
        *(ecosystem->vtx + vertex_index) = *(ecosystem->vtx + btm_idx);
        *(ecosystem->vtx + btm_idx) = tmp_vtx_ptr; 
        ( *(ecosystem->vtx + vertex_index) )->label = vertex_index;
        ( *(ecosystem->vtx + btm_idx) )->label = btm_idx;
        ecosystem->isolated += 1;
        ecosystem->survived -= 1;
    } else{
        tmp_connection = (vertex**) realloc( (*(ecosystem->vtx + vertex_index))->connection, (*(ecosystem->vtx + vertex_index))->degree * sizeof(vertex*) );
        if (tmp_connection == NULL){
            printf("\n ERROR: CONNECTION REALLOC HAS FAILED \n");
            free((*(ecosystem->vtx + vertex_index))->connection);
            exit(MY_MEMORY_FAIL);
        } else {
            (*(ecosystem->vtx + vertex_index))->connection = tmp_connection;
        }
        tmp_strength = (double*) realloc( (*(ecosystem->vtx + vertex_index))->strength, (*(ecosystem->vtx + vertex_index))->degree * sizeof(double) );
        if (tmp_strength == NULL){ 
            printf("\n ERROR: STRENGTH REALLOC HAS FAILED \n");
            free((*(ecosystem->vtx + vertex_index))->strength);
            exit(MY_MEMORY_FAIL);
        } else {
            (*(ecosystem->vtx + vertex_index))->strength = tmp_strength;
        }
    }
    return;
}

void clean_up_graph(graph *ecosystem){
    int i;
    for(i=0; i<ecosystem->size; i++){
        if((*(ecosystem->vtx+i))->degree != 0){
            free((*(ecosystem->vtx+i))->connection);
            free((*(ecosystem->vtx+i))->strength);
        }
    }
    return;
}

void erase_ecosystem(graph *ecosystem){
    int i;
    for(i=0; i<ecosystem->size; i++){
        free((*(ecosystem->vtx+i)));
    }
    free(ecosystem->vtx);
    free(ecosystem);
    return;
}

void status_set_up_new_measure(graph *ecosystem){
    int i;
    for(i=0; i<ecosystem->size - ecosystem->isolated; i++){
        (*(ecosystem->vtx+i))->status = 'S'; 
    }
    ecosystem->extincted = 0;
    ecosystem->isolated_from_evolution = 0;
    ecosystem->survived = ecosystem->size - ecosystem->isolated; // Gli unici che non cambiano sono gli isolati
    return;
}

void extract_and_save_random_initial_conditions(graph *ecosystem, double population_factor, FILE *fp_initial_conditions){
    int i;
    status_set_up_new_measure(ecosystem);
    for(i=0; i<ecosystem->size; i++){
        (*(ecosystem->vtx+i))->x = RNG() * population_factor;
        (*(ecosystem->vtx+i))->logN = log( (*(ecosystem->vtx+i))->x );
        fprintf(fp_initial_conditions, "%.17f\n", (*(ecosystem->vtx+i))->logN); 
    }
    return;
}

void extract_and_save_delta_initial_conditions(graph *ecosystem, double population_factor, FILE *fp_initial_conditions){
    int i;
    status_set_up_new_measure(ecosystem);
    for(i=0; i<ecosystem->size; i++){
        (*(ecosystem->vtx+i))->x = population_factor;
        (*(ecosystem->vtx+i))->logN = log( (*(ecosystem->vtx+i))->x );
        fprintf(fp_initial_conditions, "%.17f\n", (*(ecosystem->vtx+i))->logN); 
    }
    return;
}

void load_initial_conditions(graph *ecosystem, FILE *fp_initial_conditions){
    int i;
    status_set_up_new_measure(ecosystem);
    for(i=0; i<ecosystem->size; i++){
        fscanf(fp_initial_conditions, "%lf", &((*(ecosystem->vtx+i))->logN));
        (*(ecosystem->vtx+i))->x = exp((*(ecosystem->vtx+i))->logN);
    }
    return;
}

void load_initial_conditions_from_equilibrium_point(graph *ecosystem, FILE *fp_initial_conditions){
    int i;
    status_set_up_new_measure(ecosystem);
    for(i=0; i<ecosystem->size; i++){
        fscanf(fp_initial_conditions, "%*f\t%lf\t%*c", &((*(ecosystem->vtx+i))->logN));
        (*(ecosystem->vtx+i))->x = exp((*(ecosystem->vtx+i))->logN);
    }
    return;
}

void generate_file_of_ones(char *file_name, int N){
    FILE *fp = my_open_writing_file(file_name);
    for(int i=0; i<N; i++){
        fprintf(fp, "1\n");
    }
    fclose(fp);
    return;
}

void copy_abundances_into_array(graph *ecosystem, double* x){
    int i;
    for(i=0; i<ecosystem->size; i++){
        *(x+i) = (*(ecosystem->vtx+i))->x;
    }
    return;
}

bool are_abundances_in_array(graph *ecosystem, double *y, double delta){
    bool answer = MY_TRUE;
    int i;
    for(i=0; i<ecosystem->size; i++){
        if(fabs( (*(ecosystem->vtx+i))->x - *(y+i) ) > delta){
            answer = MY_FALSE;
            i = ecosystem->size;
        }
    }
    return answer;
}

bool are_abundances_in_matrix(graph *ecosystem, double **matrix, int first_dim, double delta, int *index_pointer){
    bool answer = MY_FALSE;
    int k;
    for(k=0; k<first_dim; k++){
        if( are_abundances_in_array(ecosystem, *(matrix+k), delta) ){
            answer = MY_TRUE;
            *(index_pointer) = k;
            k = first_dim;
        }
    }
    return answer;
}

bool are_abundances_in_array_and_take_max(graph *ecosystem, double *y, double delta, double *max_dist){
    bool answer = MY_TRUE;
    int i;
    *max_dist = 0;
    for(i=0; i<ecosystem->size; i++){
        *max_dist = fmax((ecosystem->vtx[i])->x - y[i], *max_dist);
        if(fabs(*max_dist) > delta){ 
            answer = MY_FALSE;
        }
    }
    return answer;
}

void create_bunch_of_directories(char* dir_name){
    char *name_buffer;
    name_buffer = my_char_malloc(CHAR_LENGHT);
    snprintf(name_buffer, CHAR_LENGHT, "mkdir -p %s", dir_name);
    system(name_buffer);
    snprintf(name_buffer, CHAR_LENGHT, "mkdir -p %s/Extractions", dir_name);
    system(name_buffer);
    snprintf(name_buffer, CHAR_LENGHT, "mkdir -p %s/Initial_Conditions", dir_name);
    system(name_buffer);
    snprintf(name_buffer, CHAR_LENGHT, "mkdir -p %s/Evolutions", dir_name);
    system(name_buffer);
    free(name_buffer);
    return;
}

void GLV_equation(double* derivative_func, graph* ecosystem, double lambda){
    int i, j;
    double double_appoggio = 0;
    for(i=0; i<ecosystem->size; i++){
        *(derivative_func+i) = (*(ecosystem->vtx+i))->r * ( 1 - (*(ecosystem->vtx+i))->x / (*(ecosystem->vtx+i))->K );
        double_appoggio = 0;
        for(j=0; j < (*(ecosystem->vtx+i))->degree; j++){
            double_appoggio += (*( (*(ecosystem->vtx+i))->strength + j )) * (*((*(ecosystem->vtx+i))->connection+j))->x; 
        }
        *(derivative_func+i) = *(derivative_func+i) - double_appoggio;
        *(derivative_func+i) = (ecosystem->vtx[i])->x * derivative_func[i] + lambda;
    }
    return;
}

void log_LV_gen_eq(double* log_derivative_func, graph* ecosystem, double lambda){
    int i, j;
    double double_appoggio = 0;
    for(i=0; i<ecosystem->size; i++){
        *(log_derivative_func+i) = (*(ecosystem->vtx+i))->r * ( 1 - (*(ecosystem->vtx+i))->x / (*(ecosystem->vtx+i))->K ) + lambda/(*(ecosystem->vtx+i))->x;
        double_appoggio = 0;
        for(j=0; j < (*(ecosystem->vtx+i))->degree; j++){
            double_appoggio += (*( (*(ecosystem->vtx+i))->strength + j )) * (*((*(ecosystem->vtx+i))->connection+j))->x; 
        }
        *(log_derivative_func+i) = *(log_derivative_func+i) - double_appoggio; // E' importante che qui ci sia il meno in quanto abbiamo preso la convenzione che interazione positive siano competitive (negative saranno mutualistiche)
    }
    return;
}

// Cash-Karp Runge-Kutta Step
void rkck_algorithm(graph* ecosystem, double **k, double t, double h, double *logN_out, double *logN_lte, double lambda){
    static const double b10=0.2, b20=3.0/40.0, b21=9.0/40.0, b30=0.3, b31 = -0.9, b32=1.2, b40 = -11.0/54.0, b41=2.5, b42 = -70.0/27.0, b43=35.0/27.0;
    static const double b50=1631.0/55296.0, b51=175.0/512.0, b52=575.0/13824.0, b53=44275.0/110592.0, b54=253.0/4096.0;
    static const double c0=37.0/378.0, c2=250.0/621.0, c3=125.0/594.0, c5=512.0/1771.0;
    static const double dc0=37.0/378.0-2825.0/27648.0, dc2=250.0/621.0-18575.0/48384.0, dc3=125.0/594.0-13525.0/55296.0, dc4=-277.0/14336.0, dc5=512.0/1771.0-0.25;
    int i;

    for(i=0; i<ecosystem->size; i++){
        (*(ecosystem->vtx+i))->x = (*(ecosystem->vtx+i))->logN + h * b10*(*(*k+i));
        (*(ecosystem->vtx+i))->x = exp((*(ecosystem->vtx+i))->x); 
    }
    log_LV_gen_eq(*(k+1), ecosystem, lambda);
    
    for(i=0; i<ecosystem->size; i++){
        (*(ecosystem->vtx+i))->x = (*(ecosystem->vtx+i))->logN + h * (b20*(*(*k+i)) + b21*(*(*(k+1)+i)));  
        (*(ecosystem->vtx+i))->x = exp((*(ecosystem->vtx+i))->x); 
    }
    log_LV_gen_eq(*(k+2), ecosystem, lambda);
    
    for(i=0; i<ecosystem->size; i++){
        (*(ecosystem->vtx+i))->x = (*(ecosystem->vtx+i))->logN + h * (b30*(*(*k+i)) + b31*(*(*(k+1)+i)) + b32*(*(*(k+2)+i)));  
        (*(ecosystem->vtx+i))->x = exp((*(ecosystem->vtx+i))->x); 
    }
    log_LV_gen_eq (*(k+3), ecosystem, lambda);
    
    for(i=0; i<ecosystem->size; i++){
        (*(ecosystem->vtx+i))->x = (*(ecosystem->vtx+i))->logN + h * (b40*(*(*k+i)) + b41*(*(*(k+1)+i)) + b42*(*(*(k+2)+i)) + b43*(*(*(k+3)+i)));  
        (*(ecosystem->vtx+i))->x = exp((*(ecosystem->vtx+i))->x); 
    }
    log_LV_gen_eq(*(k+4), ecosystem, lambda); 
    
    for(i=0; i<ecosystem->size; i++){
        (*(ecosystem->vtx+i))->x = (*(ecosystem->vtx+i))->logN + h * (b50*(*(*k+i)) + b51*(*(*(k+1)+i)) + b52*(*(*(k+2)+i)) + b53*(*(*(k+3)+i)) + b54*(*(*(k+4)+i)));  
        (*(ecosystem->vtx+i))->x = exp((*(ecosystem->vtx+i))->x); 
    }
    log_LV_gen_eq(*(k+5), ecosystem, lambda); 

    for(i=0; i<ecosystem->size; i++){
        logN_out[i] = (*(ecosystem->vtx+i))->logN + h * (c0*(*(*k+i)) + c2*(*(*(k+2)+i)) + c3*(*(*(k+3)+i)) + c5*(*(*(k+5)+i)));
        //N_out[i] = exp(N_out[i]); // It is better to return LogN_out
        logN_lte[i] = h*(dc0*(*(*k+i)) + dc2*(*(*(k+2)+i)) + dc3*(*(*(k+3)+i)) + dc4*(*(*(k+4)+i)) + dc5*(*(*(k+5)+i))); // Estimate error as diﬀerence between fourth and ﬁfth order methods.
        //N_lte[i] = exp(N_lte[i]); // It is better to return LogN_lte
    }
    return;
}

// Fifth-order Runge-Kutta step with monitoring of local truncation error to ensure accuracy and adjust stepsize.
// logN_scal is the array against which the error is scaled. 
// On output, logN and t are replaced by their new values, h_did is the stepsize that was actually accomplished, and h_next is the estimated next stepsize. 
void rk5_quality_control_stepper(graph* ecosystem, double **k, double *t_adr, double h_try, double *h_did, double *h_next, double eps, double *logN_scal, double lambda){
    int i;
    bool step_fail = MY_TRUE; // We make at least the first step (MY_TRUE = True = 1)
    double err_max, h, h_temp, t_new;
    double *logN_out, *logN_lte;

    logN_out = my_double_malloc(ecosystem->size);
    logN_lte = my_double_malloc(ecosystem->size);
    h = h_try; // Set stepsize to the initial trial value.
    while(step_fail){
        rkck_algorithm(ecosystem, k, *t_adr, h, logN_out, logN_lte, lambda); // Take a step attempt.
        // Evaluate accuracy/error.
        err_max = 0.0;
        if(logN_scal==nullptr){ // if logN_scal is nullptr take a trivial error_scale
            for(i=0; i<ecosystem->size; i++){
                err_max = fmax(err_max, fabs(logN_lte[i]));
            }
        }
        else{ // else use logN_scal
            for(i=0; i<ecosystem->size; i++){
                err_max = fmax(err_max, fabs(logN_lte[i]/logN_scal[i]));
            }
        }
        err_max /= eps; // Scale relative to required tolerance.
        if(err_max <= 1.0){ // Step succeeded.
            step_fail = MY_FALSE;
            for(i=0; i<ecosystem->size; i++){ // Update logN values and compute new N values.
                (*(ecosystem->vtx+i))->logN = logN_out[i];
                (*(ecosystem->vtx+i))->x = exp((*(ecosystem->vtx+i))->logN);
            }
        }
        else{ // Truncation error too large.
            for(i=0; i<ecosystem->size; i++){ // Reset original N values.
                (*(ecosystem->vtx+i))->x = exp((*(ecosystem->vtx+i))->logN);
            }
            h_temp = SAFETY * h * pow(err_max, PSHRNK); // Reduce stepsize.
            h = fmax(h_temp, 0.1*h); // No more than a factor of 10 shrinking.
            // h = (h > 0.0 ? fmax(h_temp, 0.1*h) : fmin(h_temp, 0.1*h)); // No more than a factor of 10 shrinking. Remember that "condition ? true_value : false_value" is the ternary operation but I don't think it is needed for us (h is always positive)
            t_new = (*t_adr) + h;
            if(t_new == *t_adr){ // Check underflow.
                printf("\n UNDERFLOW ERROR: Stepsize underflow in rk5_quality_control_stepper!\n");
                free(logN_out);
                free(logN_lte);
                exit(MY_TROUBLE);
            }
        }
    }
    // Compute size of next step h_next and update time t.
    if(err_max > ERRCON){ // No more than a factor of 5 increase (We apply a conservative approach and we do not want h_next to grow too much).
        *h_next = SAFETY * h * pow(err_max, PGROW);
    }
    else{
        *h_next = 5.0 * h;
    }
    *h_did = h;
    *t_adr += h;
    // Free allocated memory.
    free(logN_out);
    free(logN_lte);
    return;
}

void log_RungeKutta_adaptive_onestep(graph* ecosystem, double **k, double *t_adr, double lambda, double eps, double *h_try, bool *stepsize_trouble_adr){
    double h_next, h_did;
    log_LV_gen_eq(*k, ecosystem, lambda); // First derivative evaluation
    // If stepsize must not overshoot (we have to end precisely at t_max), here is the place to check. Anyway, I don't think this is important for us. Example: if((t_max-t) < h){h = t_max-t;}
    rk5_quality_control_stepper(ecosystem, k, t_adr, *h_try, &h_did, &h_next, eps, nullptr, lambda);
    *h_try = h_next; // Stepsize for next step
    return;
}

// Runge-Kutta driver with adaptive stepsize control. Starting abundances in ecosystems are replaced with the integrated values at time t_max.
// h_first is the first stepsize, h_min the minimum allowed stepsize, eps the accuracy, good and bad the (address of) number of good and bad steps taken.
void log_RungeKutta_adaptive_driver(graph* ecosystem, double t_max, double h_first, double h_min, double eps, double threshold, int measure_counter, bool *divergence_adr, bool *stepsize_trouble_adr, int *good, int *bad, double lambda, double deltat_save, void (*save_lv_function)(graph*, double**, double, double*, FILE*, FILE*), FILE* fp_RK, FILE* fp_speed){
    int i, j, effective_steps; 
    int evaluations_needed = 6;
    double t = 0;
    double t_save = -2 * deltat_save; // Assures storage of initial values.
    double h, h_next, h_did;
    double speed;
    double **k;
    double *logN_scal;
    bool equilibrium_reached;

    h = h_first;
    *(divergence_adr) = MY_FALSE; // 
    *(stepsize_trouble_adr) = MY_FALSE;
    *good = *bad = 0;
    k = my_double_arofars_malloc(evaluations_needed, ecosystem->size);
    // Set error scale (to have a trivial error scale leave logN_scale=nullptr)
    logN_scal = nullptr;
    // Let's start with the steps
    for(j=1; j<=MAX_ADAPTIVE_STEPS; j++){ // Take at most MAX_ADAPTIVE_STEPS adaptive steps.
        log_LV_gen_eq(*k, ecosystem, lambda); // First derivative evaluation
        // Save intermediate results (abundances and speed) into file with the argument function save_lv_function.
        if(t-t_save > deltat_save){
            save_lv_function(ecosystem, k, t, &speed, fp_RK, fp_speed);
            t_save = t;
        }
        // If stepsize must not overshoot (we have to end precisely at t_max), here is the place to check. Anyway, I don't think this is important for us. Example: if((t_max-t) < h){h = t_max-t;}
        rk5_quality_control_stepper(ecosystem, k, &t, h, &h_did, &h_next, eps, logN_scal, lambda);
        // Check if the step was successful or not
        if(h_did == h){
            (*good)++; 
        }
        else{
            (*bad)++;
        }
        // Check stepsize is not too small
        if(h_next < h_min){
            printf(" STEPSIZE WARNING: Step size %lg too small in log_RungeKutta4_adaptive_driver!\n Checking divergence using constant stepsize...\n", h_next);
            *(stepsize_trouble_adr) = MY_TRUE;
            effective_steps = j;
            j = MAX_ADAPTIVE_STEPS+2; // We get out of the loop
        }
        h = h_next; // Stepsize for next step
        // Check whetever we have done or not
        if(t>t_max){ 
            printf("# Everything fine at time %lf, step %d on measure %d \n", t, j, measure_counter);
            effective_steps = j;
            j = MAX_ADAPTIVE_STEPS+2; // We get out of the loop
        }
        // Each period check for divergence or equilibrium
        if(j%CHECK_PERIOD==0){  
            for(i=0; i<ecosystem->size; i++){
                if(std::isfinite((*(ecosystem->vtx+i))->x) == MY_FALSE){ // Check if the ecosystem has diverged
                    *(divergence_adr) = MY_TRUE;
                    i = ecosystem->size;
                }
            }
            if(*(divergence_adr) == MY_TRUE){
                printf("# Divergence on rkck_measure %d \n", measure_counter);
                effective_steps = j;
                j = MAX_ADAPTIVE_STEPS+2; // We get out of the loop
            }
            else{ // If no divergence
                // Check if the equilibrium has been reached.
                equilibrium_reached = MY_TRUE;
                i = 0;
                while( (equilibrium_reached == MY_TRUE) && (i<ecosystem->size) ){ // Once we found the equilibrium is not reached yet we stop checking all the species
                    if(fabs(*(*k+i)) > threshold){
                        equilibrium_reached = MY_FALSE;
                    }
                    i++;
                }
                if(equilibrium_reached == MY_TRUE){
                    printf("# Equilibrium reached on measure %d \n", measure_counter);
                    effective_steps = j;
                    j = MAX_ADAPTIVE_STEPS+2; // We get out of the loop at the end of this iteration.
                }
            }
        }
    } // End of the loop on the steps.
    // Last check for divergence on final results
    for(i=0; i<ecosystem->size; i++){
        if(std::isfinite((*(ecosystem->vtx+i))->x) == MY_FALSE){ // Check if the ecosystem has diverged
            *(divergence_adr) = MY_TRUE;
            i = ecosystem->size;
        }
    }
    if(*(divergence_adr) == MY_TRUE){
        printf("# Divergence on rkck_measure %d \n", measure_counter);
    }
    // Save final results (abundances and speed) into file with the argument function save_lv_function.
    log_LV_gen_eq(*k, ecosystem, lambda); // Compute speed in final point
    save_lv_function(ecosystem, k, t, &speed, fp_RK, fp_speed);
    t_save = t;
    
    my_double_arofars_free(k, evaluations_needed);
    free(logN_scal);
    // Check on max number of steps
    if(j == MAX_ADAPTIVE_STEPS+1){
        printf("\n NUMBER OF STEPS ERROR: Maximum number of steps reached in log_RungeKutta4_adaptive_driver!\n");
        exit(MY_TROUBLE);  
    }
    return;
}

// Standard Runge-Kutta 4 Step
void rk4_algorithm(graph* ecosystem, double **k, double *t_adr, double h, double lambda){
    int i;
    double h2 = 0.5*h; 
    double h6 = h/6.0;
    for(i=0; i<ecosystem->size; i++){
        (*(ecosystem->vtx+i))->x = (*(ecosystem->vtx+i))->logN + h2 * ( *(*k + i) );
        (*(ecosystem->vtx+i))->x = exp((*(ecosystem->vtx+i))->x);
    }
    log_LV_gen_eq (*(k+1), ecosystem, lambda);
    for(i=0; i<ecosystem->size; i++){
        (*(ecosystem->vtx+i))->x = (*(ecosystem->vtx+i))->logN + h2 * ( *(*(k+1) + i) );
        (*(ecosystem->vtx+i))->x = exp((*(ecosystem->vtx+i))->x); 
    }
    log_LV_gen_eq (*(k+2), ecosystem, lambda);
    for(i=0; i<ecosystem->size; i++){
        (*(ecosystem->vtx+i))->x = (*(ecosystem->vtx+i))->logN + h*( *(*(k+2) + i) );
        (*(ecosystem->vtx+i))->x = exp((*(ecosystem->vtx+i))->x); 
    }
    log_LV_gen_eq (*(k+3), ecosystem, lambda);
    
    for(i=0; i<ecosystem->size; i++){       
        (*(ecosystem->vtx+i))->logN = (*(ecosystem->vtx+i))->logN + h6 * ( 1. * ( *(*(k+0) + i) ) + 2. * ( *(*(k+1) + i) ) + 2. * ( *(*(k+2) + i) ) + 1. * ( *(*(k+3) + i) ) );
        (*(ecosystem->vtx+i))->x = exp((*(ecosystem->vtx+i))->logN);
    }
    *(t_adr) += h;
    return;
}

void log_RungeKutta_adaptive_mixed_static_driver_for_phase_diagram(graph* ecosystem, double t_max, double *t_end, double h_first, double h_min, double h_static, double log_factor, int measure_num, bool *divergence_adr, bool *equilibrated_adr, double *av_extinction_adr, double lambda, double tolerance, double equilibrium_threshold){
    
    int j, effective_steps;
    int evaluations_needed = 6;
    double h;
    double t = 0;
    double t_check = 0;
    double deltat_check = 5.0;
    double **k;
    bool stepsize_trouble, equilibrium_reached;
    int *extinction_counter_circular, circular_idx=0;

    k = my_double_arofars_malloc(evaluations_needed, ecosystem->size);
    extinction_counter_circular = my_int_calloc(NUM_THERM_EQ_INTERVALS);

    h = h_first;
    *(divergence_adr) = MY_FALSE; // 
    equilibrium_reached = MY_FALSE; // 
    stepsize_trouble = MY_FALSE;

    // Let's start with the steps
    for(j=1; j<=MAX_ADAPTIVE_STEPS; j++){ // Take at most MAX_ADAPTIVE_STEPS adaptive steps.
        if(!stepsize_trouble){
            log_RungeKutta_adaptive_onestep(ecosystem, k, &t, lambda, tolerance, &h, &stepsize_trouble);
            if(h < h_min){ // Check whether stepsize is too small and eventually rise stepsize_trouble
                stepsize_trouble = MY_TRUE;
                printf(" STEPSIZE WARNING: Step size %lg smaller than minimum %lg in log_RungeKutta4_adaptive_driver!\n Checking divergence using constant stepsize...\n", h, h_min);
                check_ecosystem_divergence(ecosystem, divergence_adr);
            }
        }
        else{ // Static RK Algorithm
            log_LV_gen_eq(*k, ecosystem, lambda);
            rk4_algorithm(ecosystem, k, &t, h_static, lambda);            
        }
        // Check divergence and equilibrium in log interval
        if(t-t_check > deltat_check){
            check_ecosystem_divergence(ecosystem, divergence_adr);
            if(*(divergence_adr) == MY_TRUE){
                //printf("# Divergence on RK at %lf, measure %d \n", t, measure_num);
                effective_steps = j;
                j = MAX_ADAPTIVE_STEPS+2; // We get out of the loop
            }
            else{ 
                // Check extinctions
                extinction_counter_circular[circular_idx] = 0;
                for(int i=0; i<ecosystem->size; i++){
                    if((ecosystem->vtx[i])->x < lambda)
                        extinction_counter_circular[circular_idx]++;
                }
                circular_idx = (circular_idx + 1) % NUM_THERM_EQ_INTERVALS;
                // Equilibrium Check
                check_ecosystem_equilibrium_zero_temperature(k[0], ecosystem->size, equilibrium_threshold, &equilibrium_reached);
                *equilibrated_adr = equilibrium_reached;
                if(equilibrium_reached == MY_TRUE){
                    //printf("# Equilibrium reached on measure %d \n", measure_num);
                    effective_steps = j;
                    j = MAX_ADAPTIVE_STEPS+2; // We get out of the loop
                }
            }
            t_check = t;
            deltat_check = deltat_check * log_factor;
        }
        // Check whether we reached t_max
        if(t>t_max){ 
            effective_steps = j;
            j = MAX_ADAPTIVE_STEPS+2; // We get out of the loop
        }
    } // End of the loop on the steps
    
    // Compute average extinctions
    *av_extinction_adr = 0;
    for(int i=0; i<NUM_THERM_EQ_INTERVALS; i++){
        *av_extinction_adr += extinction_counter_circular[i];
    }
    *t_end = t;
    *av_extinction_adr /= NUM_THERM_EQ_INTERVALS;

    my_double_arofars_free(k, evaluations_needed);
    free(extinction_counter_circular);
    return;
}

// Milstein stochastic integration
void Milstein_onestep_GLV_demographic_noise(graph* ecosystem, double *k, double *t_adr, double h, double T, double lambda){
    int i;
    double w_i, diffusion_term, b;
    double w_sigma = sqrt(h);
    GLV_equation(k, ecosystem, lambda); // Compute deterministic derivative
    for(i=0; i<ecosystem->size; i++){
        b = sqrt(2*T*(ecosystem->vtx[i])->x);
        w_i = GNG(0, 1) * w_sigma;
        diffusion_term = b * w_i + (T/2.0) * (w_i*w_i - h);
        // In principle the variable update should be done at the end, after evaluating all the terms but the diffusion is diagonal and the deterministic part has already been evaluated
        (ecosystem->vtx[i])->x = (ecosystem->vtx[i])->x + h * k[i] + diffusion_term;
        // Wall or Reflecting wall after noise
        if((*(ecosystem->vtx+i))->x < HARD_WALL){
            (ecosystem->vtx[i])->x = HARD_WALL;
        }
    }
    *(t_adr) += h;
    return;
}

void Milstein_driver_GLV_demographic_noise_with_single_species_measure(graph* ecosystem, double t_max, double h, double log_factor, int measure_num, bool *divergence_adr, double T, double lambda, double deltat_save, void (*save_lv_function)(graph*, double, FILE*), FILE* fp_RK, FILE* fp_eq){
    int i, j1, j2, effective_steps;
    int initial_steps = ceil(10/h);
    int max_steps = ceil(t_max/h)-initial_steps;
    int eqmeas_steps;
    double t = 0;
    double t_save = -2 * deltat_save; // Assures storage of initial values
    double t_eqcheck; // = 0;
    double eq_check = 5;
    double *k;
    double *eq_pt, *eq_pt_previous, *eq_pt_std, *eq_pt_std_previous;
    char *species_convergence;
    int equilibrium_counter = 0;

    k = my_double_malloc(ecosystem->size);
    eq_pt = my_double_calloc(ecosystem->size);
    eq_pt_previous = my_double_calloc(ecosystem->size);
    eq_pt_std = my_double_calloc(ecosystem->size);
    eq_pt_std_previous = my_double_calloc(ecosystem->size);
    species_convergence = my_char_malloc(ecosystem->size);
    *(divergence_adr) = MY_FALSE; // 
    
    j2=1; // We initialise here j2 so that we can change it in the following loop
    // Let's start with the initial transient steps (no check on equilibrium)
    for(j1=1; j1<=initial_steps; j1++){
        // Save intermediate abundances into file with the argument function save_lv_function and check divergence
        if(t-t_save > deltat_save){
            save_lv_function(ecosystem, t, fp_RK);
            check_ecosystem_divergence(ecosystem, divergence_adr);
            if(*(divergence_adr) == MY_TRUE){
                printf("# Divergence on milstein at %lf, measure %d \n", t, measure_num);
                effective_steps = j1;
                j1 = max_steps+2; // We get out of the first loop
                j2 = max_steps+2; // We get out of the second loop
            }
            t_save = t;
        }
        Milstein_onestep_GLV_demographic_noise(ecosystem, k, &t, h, T, lambda);
    }
    // Reset check equilibrium counters
    t_eqcheck = t;
    eqmeas_steps = 0;
    // Let's start with the steps in which we check equilibrium
    // We have initialise j2 before, j2=j2
    for(; j2<=max_steps; j2++){ // Take at most max_steps adaptive steps.
        // Save intermediate abundances into file with the argument function save_lv_function.
        if(t-t_save > deltat_save){
            save_lv_function(ecosystem, t, fp_RK);
            t_save = t;
        }
        Milstein_onestep_GLV_demographic_noise(ecosystem, k, &t, h, T, lambda);
        // Compute average
        eqmeas_steps++;
        for(i=0; i<ecosystem->size; i++){
            eq_pt[i] += (ecosystem->vtx[i])->x;
            eq_pt_std[i] += (ecosystem->vtx[i])->x * (ecosystem->vtx[i])->x;
        }
        if(t-t_eqcheck > eq_check){ // Each logarithmic interval we check for divergence or equilibrium
            // Divergence Check
            check_ecosystem_divergence(ecosystem, divergence_adr);
            if(*(divergence_adr) == MY_TRUE){
                printf("# Divergence on milstein at %lf, measure %d \n", t, measure_num);
                effective_steps = j2+initial_steps;
                j2 = max_steps+2; // We get out of the loop
            }
            else{ // Equilibrium Check
                // Compute average and check species equilibrium
                for(i=0; i<ecosystem->size; i++){
                    eq_pt[i] /= eqmeas_steps;
                    eq_pt_std[i] /= eqmeas_steps;
                    eq_pt_std[i] = sqrt(fabs(eq_pt_std[i]-eq_pt[i]*eq_pt[i]));
                    if(fabs(eq_pt[i]-eq_pt_previous[i]) < eq_pt_std[i]+eq_pt_std_previous[i])
                        species_convergence[i] = 'T';
                    else
                        species_convergence[i] = 'F';
                }
                // Check if the equilibrium has been reached for each species
                check_ecosystem_thermal_equilibrium(ecosystem, &equilibrium_counter, species_convergence);
                if(equilibrium_counter == NUM_THERM_EQ_INTERVALS){
                    printf("# Equilibrium reached on measure %d \n", measure_num);
                    effective_steps = j2+initial_steps;
                    j2 = max_steps+2; // We get out of the loop at the end of this iteration.
                }
                // Reset averages
                for(i=0; i<ecosystem->size; i++){
                    eq_pt_previous[i] = eq_pt[i];
                    eq_pt_std_previous[i] = eq_pt_std[i];
                    eq_pt[i] = 0;
                    eq_pt_std[i] = 0;
                }
                eqmeas_steps = 0;
                t_eqcheck = t;
                eq_check = eq_check * log_factor;
            }
        }
    } // End of the loop on the steps.
    // Save final results (abundances and speed) into file with the argument function save_lv_function.
    save_lv_function(ecosystem, t, fp_RK);
    t_save = t;
    // Save final data in David format
    for(i=0; i<ecosystem->size; i++){
        fprintf(fp_eq, "%d\t%c\t%.17f\t%.17f\n", i, species_convergence[i], eq_pt_previous[i], eq_pt_std_previous[i]);
    }
    free(k);
    free(eq_pt);
    free(eq_pt_previous);
    free(eq_pt_std);
    free(eq_pt_std_previous);
    free(species_convergence);
    return;
}

void Milstein_driver_GLV_demographic_noise_with_single_species_measure_and_summary(graph* ecosystem, double t_max, double h, double log_factor, int measure_num, 
        bool *divergence_adr, double T, double lambda, double deltat_save, void (*save_lv_function)(graph*, double, FILE*), 
        FILE *fp_summary, FILE* fp_RK, FILE* fp_eq, time_t my_seed, int ext_num){
    
    int i, j1, j2, effective_steps;
    int initial_steps = ceil(10/h);
    int max_steps = ceil(t_max/h)-initial_steps;
    int eqmeas_steps;
    double t = 0;
    double t_save = -2 * deltat_save; // Assures storage of initial values
    double t_eqcheck; // = 0;
    double eq_check = 5;
    double *k;
    double *eq_pt, *eq_pt_previous, *eq_pt_std, *eq_pt_std_previous;
    char *species_convergence;
    int *extinction_counter_circular, circular_idx=0;
    int equilibrium_counter = 0;
    auto chrono_start = std::chrono::high_resolution_clock::now();

    k = my_double_malloc(ecosystem->size);
    eq_pt = my_double_calloc(ecosystem->size);
    eq_pt_previous = my_double_calloc(ecosystem->size);
    eq_pt_std = my_double_calloc(ecosystem->size);
    eq_pt_std_previous = my_double_calloc(ecosystem->size);
    species_convergence = my_char_malloc(ecosystem->size);
    extinction_counter_circular = my_int_calloc(NUM_THERM_EQ_INTERVALS);
    *(divergence_adr) = MY_FALSE;
    
    j2=1; // We initialise here j2 so that we can change it in the following loop
    // Let's start with the initial transient steps (no check on equilibrium)
    for(j1=1; j1<=initial_steps; j1++){
        // Save intermediate abundances into file with the argument function save_lv_function and check divergence
        if(t-t_save > deltat_save){
            save_lv_function(ecosystem, t, fp_RK);
            check_ecosystem_divergence(ecosystem, divergence_adr);
            if(*(divergence_adr) == MY_TRUE){
                printf("# Divergence on milstein at %lf, measure %d \n", t, measure_num);
                effective_steps = j1;
                j1 = max_steps+2; // We get out of the first loop
                j2 = max_steps+2; // We get out of the second loop
            }
            t_save = t;
        }
        Milstein_onestep_GLV_demographic_noise(ecosystem, k, &t, h, T, lambda);
    }
    // Reset check equilibrium counters
    t_eqcheck = t;
    eqmeas_steps = 0;
    // Let's start with the steps in which we check equilibrium
    // We have initialise j2 before, j2=j2
    for(; j2<=max_steps; j2++){ // Take at most max_steps adaptive steps.
        // Save intermediate abundances into file with the argument function save_lv_function.
        if(t-t_save > deltat_save){
            save_lv_function(ecosystem, t, fp_RK);
            t_save = t;
        }
        Milstein_onestep_GLV_demographic_noise(ecosystem, k, &t, h, T, lambda);
        // Compute average
        eqmeas_steps++;
        for(i=0; i<ecosystem->size; i++){
            eq_pt[i] += (ecosystem->vtx[i])->x;
            eq_pt_std[i] += (ecosystem->vtx[i])->x * (ecosystem->vtx[i])->x;
        }
        if(t-t_eqcheck > eq_check){ // Each logarithmic interval we check for divergence or equilibrium
            // Divergence Check
            check_ecosystem_divergence(ecosystem, divergence_adr);
            if(*(divergence_adr) == MY_TRUE){
                printf("# Divergence on milstein at %lf, measure %d \n", t, measure_num);
                effective_steps = j2+initial_steps;
                j2 = max_steps+2; // We get out of the loop
            }
            else{ // Equilibrium Check
                // Compute average and check species equilibrium
                for(i=0; i<ecosystem->size; i++){
                    eq_pt[i] /= eqmeas_steps;
                    eq_pt_std[i] /= eqmeas_steps;
                    eq_pt_std[i] = sqrt(fabs(eq_pt_std[i]-eq_pt[i]*eq_pt[i]));
                    if(fabs(eq_pt[i]-eq_pt_previous[i]) < eq_pt_std[i]+eq_pt_std_previous[i])
                        species_convergence[i] = 'T';
                    else
                        species_convergence[i] = 'F';
                }
                // Check extinctions
                extinction_counter_circular[circular_idx] = 0;
                for(i=0; i<ecosystem->size; i++){
                    if((ecosystem->vtx[i])->x < lambda)
                        extinction_counter_circular[circular_idx]++;
                }
                circular_idx = (circular_idx + 1) % NUM_THERM_EQ_INTERVALS;
                // Check if the equilibrium has been reached for each species
                check_ecosystem_thermal_equilibrium(ecosystem, &equilibrium_counter, species_convergence);
                if(equilibrium_counter == NUM_THERM_EQ_INTERVALS){
                    printf("# Equilibrium reached on measure %d \n", measure_num);
                    effective_steps = j2+initial_steps;
                    j2 = max_steps+2; // We get out of the loop at the end of this iteration.
                }
                // Reset averages
                for(i=0; i<ecosystem->size; i++){
                    eq_pt_previous[i] = eq_pt[i];
                    eq_pt_std_previous[i] = eq_pt_std[i];
                    eq_pt[i] = 0;
                    eq_pt_std[i] = 0;
                }
                eqmeas_steps = 0;
                t_eqcheck = t;
                eq_check = eq_check * log_factor;
            }
        }
    } // End of the loop on the steps.
    // Save final results (abundances and speed) into file with the argument function save_lv_function.
    
    save_lv_function(ecosystem, t, fp_RK);
    t_save = t;
    // Save final data in David format
    for(i=0; i<ecosystem->size; i++){
        fprintf(fp_eq, "%d\t%c\t%.17f\t%.17f\n", i, species_convergence[i], eq_pt_previous[i], eq_pt_std_previous[i]);
    }

    // SUMMARY
    double av_abundance = 0, av_abundance_sqr = 0, av_abundance_std = 0, 
           av_abundance_std_sqr = 0;
    double av_extinction = 0;
    int num_noneq = 0;
    // Compute averages
    for(i=0; i<ecosystem->size; i++){
        av_abundance += eq_pt_previous[i];
        av_abundance_sqr += eq_pt_previous[i] * eq_pt_previous[i];
        av_abundance_std += eq_pt_std_previous[i];
        av_abundance_std_sqr += eq_pt_std_previous[i] * eq_pt_std_previous[i];
        if (species_convergence[i] == 'F'){
            num_noneq++;
        }
    }
    av_abundance /= ecosystem->size;
    av_abundance_sqr /= ecosystem->size;
    av_abundance_std /= ecosystem->size;
    av_abundance_std_sqr /= ecosystem->size;
    for(i=0; i<NUM_THERM_EQ_INTERVALS; i++){
        av_extinction += extinction_counter_circular[i];
    }
    av_extinction /= NUM_THERM_EQ_INTERVALS;
    // Compute errors
    double av_abundance_error = sqrt(av_abundance_sqr - av_abundance * av_abundance);
    double av_abundance_std_error = sqrt(av_abundance_std_sqr - av_abundance_std * av_abundance_std);
    auto chrono_end = std::chrono::high_resolution_clock::now();
    time_t computation_time = std::chrono::duration_cast<std::chrono::milliseconds>(chrono_end - chrono_start).count();
    // Print summary
    fprintf(fp_summary, "%d\t%d\t%ld\t%.17lf\t%d\t%.17f\t%.17f\t%.17f\t%.17f\t%.17f\t%ld\n", ext_num, measure_num, my_seed, t, num_noneq, av_extinction, av_abundance, av_abundance_error, av_abundance_std, av_abundance_std_error, computation_time);
    
    free(k);
    free(eq_pt);
    free(eq_pt_previous);
    free(eq_pt_std);
    free(eq_pt_std_previous);
    free(species_convergence);
    free(extinction_counter_circular);
    return;
}

void Milstein_driver_GLV_demographic_noise_only_with_final_summary(graph* ecosystem, double t_max, double h, double log_factor, int measure_num, 
        bool *divergence_adr, double T, double lambda, double deltat_divergence, FILE *fp_summary, time_t my_seed, int ext_num){
    
    int i, j1, j2, effective_steps;
    int initial_steps = ceil(10/h);
    int max_steps = ceil(t_max/h)-initial_steps;
    int eqmeas_steps;
    double t = 0;
    double t_divcheck = 0;
    double t_eqcheck; // = 0;
    double eq_check = 5;
    double *k;
    double *eq_pt, *eq_pt_previous, *eq_pt_std, *eq_pt_std_previous;
    char *species_convergence;
    int *extinction_counter_circular, circular_idx=0;
    int equilibrium_counter = 0;

    k = my_double_malloc(ecosystem->size);
    eq_pt = my_double_calloc(ecosystem->size);
    eq_pt_previous = my_double_calloc(ecosystem->size);
    eq_pt_std = my_double_calloc(ecosystem->size);
    eq_pt_std_previous = my_double_calloc(ecosystem->size);
    species_convergence = my_char_malloc(ecosystem->size);
    extinction_counter_circular = my_int_calloc(NUM_THERM_EQ_INTERVALS);
    *(divergence_adr) = MY_FALSE; // 
    
    j2=1; // We initialise here j2 so that we can change it in the following loop
    // Let's start with the initial transient steps (no check on equilibrium)
    for(j1=1; j1<=initial_steps; j1++){
        // Check divergence
        if(t-t_divcheck > deltat_divergence){
            check_ecosystem_divergence(ecosystem, divergence_adr);
            if(*(divergence_adr) == MY_TRUE){
                printf("# Divergence on milstein at %lf, measure %d \n", t, measure_num);
                effective_steps = j1;
                j1 = max_steps+2; // We get out of the first loop
                j2 = max_steps+2; // We get out of the second loop
            }
            t_divcheck = t;
        }
        Milstein_onestep_GLV_demographic_noise(ecosystem, k, &t, h, T, lambda);
    }
    // Reset check equilibrium counters
    t_eqcheck = t;
    eqmeas_steps = 0;
    // Let's start with the steps in which we check equilibrium
    // We have initialise j2 before, j2=j2
    for(; j2<=max_steps; j2++){ // Take at most max_steps adaptive steps.
        Milstein_onestep_GLV_demographic_noise(ecosystem, k, &t, h, T, lambda);
        // Compute average
        eqmeas_steps++;
        for(i=0; i<ecosystem->size; i++){
            eq_pt[i] += (ecosystem->vtx[i])->x;
            eq_pt_std[i] += (ecosystem->vtx[i])->x * (ecosystem->vtx[i])->x;
        }
        if(t-t_eqcheck > eq_check){ // Each logarithmic interval we check for divergence or equilibrium
            // Divergence Check
            check_ecosystem_divergence(ecosystem, divergence_adr);
            if(*(divergence_adr) == MY_TRUE){
                printf("# Divergence on milstein at %lf, measure %d \n", t, measure_num);
                effective_steps = j2+initial_steps;
                j2 = max_steps+2; // We get out of the loop
            }
            else{ // Equilibrium Check
                // Compute average and check species equilibrium
                for(i=0; i<ecosystem->size; i++){
                    eq_pt[i] /= eqmeas_steps;
                    eq_pt_std[i] /= eqmeas_steps;
                    eq_pt_std[i] = sqrt(fabs(eq_pt_std[i]-eq_pt[i]*eq_pt[i]));
                    if(fabs(eq_pt[i]-eq_pt_previous[i]) < eq_pt_std[i]+eq_pt_std_previous[i])
                        species_convergence[i] = 'T';
                    else
                        species_convergence[i] = 'F';
                }
                // Check extinctions
                extinction_counter_circular[circular_idx] = 0;
                for(i=0; i<ecosystem->size; i++){
                    if((ecosystem->vtx[i])->x < lambda)
                        extinction_counter_circular[circular_idx]++;
                }
                circular_idx = (circular_idx + 1) % NUM_THERM_EQ_INTERVALS;
                // Check if the equilibrium has been reached for each species
                check_ecosystem_thermal_equilibrium(ecosystem, &equilibrium_counter, species_convergence);
                if(equilibrium_counter == NUM_THERM_EQ_INTERVALS){
                    printf("# Equilibrium reached on measure %d \n", measure_num);
                    effective_steps = j2+initial_steps;
                    j2 = max_steps+2; // We get out of the loop at the end of this iteration.
                }
                // Reset averages
                for(i=0; i<ecosystem->size; i++){
                    eq_pt_previous[i] = eq_pt[i];
                    eq_pt_std_previous[i] = eq_pt_std[i];
                    eq_pt[i] = 0;
                    eq_pt_std[i] = 0;
                }
                eqmeas_steps = 0;
                t_eqcheck = t;
                eq_check = eq_check * log_factor;
            }
        }
    } // End of the loop on the steps.

    // SUMMARY
    double av_abundance = 0, av_abundance_sqr = 0, av_abundance_std = 0, 
           av_abundance_std_sqr = 0;
    double av_extinction = 0;
    int num_noneq = 0;
    // Compute averages
    for(i=0; i<ecosystem->size; i++){
        av_abundance += eq_pt_previous[i];
        av_abundance_sqr += eq_pt_previous[i] * eq_pt_previous[i];
        av_abundance_std += eq_pt_std_previous[i];
        av_abundance_std_sqr += eq_pt_std_previous[i] * eq_pt_std_previous[i];
        if (species_convergence[i] == 'F'){
            num_noneq++;
        }
    }
    av_abundance /= ecosystem->size;
    av_abundance_sqr /= ecosystem->size;
    av_abundance_std /= ecosystem->size;
    av_abundance_std_sqr /= ecosystem->size;
    for(i=0; i<NUM_THERM_EQ_INTERVALS; i++){
        av_extinction += extinction_counter_circular[i];
    }
    av_extinction /= NUM_THERM_EQ_INTERVALS;
    // Compute errors
    double av_abundance_error = sqrt(av_abundance_sqr - av_abundance * av_abundance);
    double av_abundance_std_error = sqrt(av_abundance_std_sqr - av_abundance_std * av_abundance_std);
    // Print summary
    fprintf(fp_summary, "%d\t%d\t%ld\t%.17lf\t%d\t%.17f\t%.17f\t%.17f\t%.17f\t%.17f\n", ext_num, measure_num, my_seed, t, 
                                                                num_noneq, av_extinction, av_abundance, av_abundance_error, av_abundance_std, av_abundance_std_error);
    free(k);
    free(eq_pt);
    free(eq_pt_previous);
    free(eq_pt_std);
    free(eq_pt_std_previous);
    free(species_convergence);
    free(extinction_counter_circular);
    return;
}

void check_ecosystem_divergence(graph* ecosystem, bool *divergence_adr){
    for(int i=0; i<ecosystem->size; i++){
        if(std::isfinite((*(ecosystem->vtx+i))->x) == MY_FALSE){ // Check if the ecosystem has diverged
            *(divergence_adr) = MY_TRUE;
            i = ecosystem->size;
        }
    }
    return;
}

void check_ecosystem_equilibrium_zero_temperature(double *derivative, int N, double equilibrium_threshold, bool *equilibrium_reached){
    int i=0;
    *equilibrium_reached = MY_TRUE;
    while( ((*equilibrium_reached) == MY_TRUE) && (i<N) ){ // Once we found the equilibrium is not reached yet we stop checking all the species
        if(fabs(derivative[i]) > equilibrium_threshold){
            *equilibrium_reached = MY_FALSE;
        }
        i++;
    }
    return;
}

void check_ecosystem_thermal_equilibrium(graph* ecosystem, int *equilibrium_counter, char *species_convergence){
    int i = 0;
    bool equilibrium_reached = MY_TRUE;
    while((equilibrium_reached == MY_TRUE) && (i<ecosystem->size)){ // Once we found the equilibrium is not reached yet we stop checking all the species
        if(species_convergence[i] == 'F'){
            equilibrium_reached = MY_FALSE;
            *equilibrium_counter = 0;
        }
        i++;
    }
    if(equilibrium_reached == MY_TRUE)
        *equilibrium_counter = *equilibrium_counter + 1;
    return;
}

// Save LotkaVolterra trajectories only
void save_LotkaVolterra_trajectories_only(graph* ecosystem, double t, FILE* fp_RK){
    int i; // Passing speed as a pointer in order to use always the same variable and spare time. The same could be done for i but it's boring.
    fprintf (fp_RK, "%.8f\t", t);
    for(i=0; i<ecosystem->size; i++){
        fprintf (fp_RK, "%.8f\t", (*(ecosystem->vtx+i))->x);
    }
    fprintf (fp_RK, "\n");
    return;
}

// Don't save anything 
void save_LotkaVolterra_nothing(graph* ecosystem, double t, FILE* fp_RK){
    //printf(" # king of broccoli is whistling\n");
    return;
}
