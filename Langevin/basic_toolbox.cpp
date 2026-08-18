#include "basic_toolbox.h"

int compare_desc(const void *a, const void *b){
    return *(const uint *)b - *(const uint *)a;
}

int my_max(int a, int b){
    int max = a;
    if(b>a){
        max = b;
    }
    return max;
}

double sum_array(double *x, int N){
    int i;
    double sum=0;
    for(i=0; i<N;i++){
        sum += x[i];
    }
    return sum;
}

int sum_array(int *n, int N){
    int i;
    int sum=0;
    for(i=0; i<N;i++){
        sum += n[i];
    }
    return sum;
}

int max_array(int *n, int N){
    int i;
    int max=n[0];
    for(i=1; i<N;i++){
        if(n[i] > max)
            max = n[i];
    }
    return max;
}

void loadArray(FILE *fp, int *x, int size){
    for(int i = 0; i < size; i++)
        //fscanf(fp, "%d", &x[i]);
        if(fscanf(fp, "%d", &x[i]) != 1){
            printf("\n ERROR: Could not read integer at line %d in file.\n", i + 1);
            exit(MY_TROUBLE);
        }
    return;
}

void fill_array(int *x, int m, int N){
    int i;
    for(i=0; i<N; i++){
        x[i] = m;
    }
    return;
}

bool are_arrays_equal(double *x, double *y, int dimension, double delta){
    bool answer = MY_TRUE;
    int i;
    for(i=0; i<dimension; i++){
        if(fabs( *(x+i) - *(y+i) ) > delta){
            answer = MY_FALSE;
            i = dimension;
        }
    }
    return answer;
}

bool is_array_in(double *array, double **matrix, int first_dim, int second_dim, double delta, int *index_pointer){
    bool answer = MY_FALSE;
    int k;
    for(k=0; k<first_dim; k++){
        if( are_arrays_equal(array, *(matrix+k), second_dim, delta) ){
            answer = MY_TRUE;
            *(index_pointer) = k;
            k = first_dim;
        }
    }
    return answer;
}

void initialize_double_array (double* x, double a, int x_dimension){
    int i;
    double* appoggio_double;
    appoggio_double = (double*) realloc( x, x_dimension*sizeof(double));
    if (appoggio_double == NULL){
        printf("\n ERROR: DOUBLE REALLOC HAS FAILED \n");
        free(x);
        exit(MY_MEMORY_FAIL);
    } else {
        x = appoggio_double;
    }
    for(i=0; i<x_dimension; i++){
        *(x+i) = a;
    }
    return;
}

void equalize_double_array (double* x_1, double* x_2, int x_dimension){
    int i;
    for(i=0; i<x_dimension; i++){
        *(x_1+i) = *(x_2+i);
    }
    return;
}

void combine_array (double alfa, double* x_1, double beta, double* x_2, double* x, int x_dimension){
    int i;
    for(i=0; i<x_dimension; i++){
        *(x+i) = alfa * ( *(x_1+i) ) + beta * ( *(x_2+i) );
    }
    return;
}

double* exp_double_array(double *x, int size){
    int i;
    double *x_exp;
    x_exp = my_double_malloc(size);
    for(i=0; i<size; i++){
        *(x_exp+i) = exp(*(x+i));
    }
    return x_exp;
}

// Swap double variables
void swap(double *x_ptr, double *y_ptr){
    double tmp = *x_ptr;
    *x_ptr = *y_ptr;
    *y_ptr = tmp;
    return;
}

void swap(int *x_ptr, int *y_ptr){
    int tmp = *x_ptr;
    *x_ptr = *y_ptr;
    *y_ptr = tmp;
    return;
}

void BubbleSort(double *x, int n, char order/*='D'*/){ // The default parameters should only be stated in the function declaration, not in the definition
    int i, j;
    if(order=='D'){
    for(i = 0; i < n-1; i++)    
        // Last i elements are already in place
        for(j = 0; j < n-i-1; j++)
            if(x[j] < x[j+1])
                swap(&x[j], &x[j+1]);
    }
    else if(order=='A'){
    for(i = 0; i < n-1; i++)    
        // Last i elements are already in place
        for(j = 0; j < n-i-1; j++)
            if(x[j] > x[j+1])
                swap(&x[j], &x[j+1]);
    }
    else{
        fprintf(stderr, "\n ERROR: UNKNOWN SORTING ORDER \n");
        exit(MY_VALUE_ERROR);
    }
    return;
}

void BubbleSort(int *x, int n, char order/*='D'*/){ // The default parameters should only be stated in the function declaration, not in the definition
    int i, j;
    if(order=='D'){
    for(i = 0; i < n-1; i++)    
        // Last i elements are already in place
        for(j = 0; j < n-i-1; j++)
            if(x[j] < x[j+1])
                swap(&x[j], &x[j+1]);
    }
    else if(order=='A'){
    for(i = 0; i < n-1; i++)    
        // Last i elements are already in place
        for(j = 0; j < n-i-1; j++)
            if(x[j] > x[j+1])
                swap(&x[j], &x[j+1]);
    }
    else{
        fprintf(stderr, "\n ERROR: UNKNOWN SORTING ORDER \n");
        exit(MY_VALUE_ERROR);
    }
    return;
}

unsigned int max_unsint(unsigned int a, unsigned int b){
    unsigned int max = a;
    if(b>a){
        max = b;
    }
    return max;
}

int linear_search_interval(unsigned int p, unsigned int *A, int N){
    int i=0;
    while (p > A[i])
        i++;
    if(i>N){
        fprintf(stderr,"\n ERROR: VALUE OUTSIDE ARRAY \n");
        exit(MY_VALUE_ERROR);
    }
    return i;
}

int linear_search_interval(double p, double *A, int N){
    int i=0;
    while (p > A[i])
        i++;
    if(i>N){
        fprintf(stderr,"\n ERROR: VALUE OUTSIDE ARRAY \n");
        exit(MY_VALUE_ERROR);
    }
    return i;
}

int binary_search_interval(double p, double *A, int N){
    int L, R, k;
    L = 0;
    R = N-1;
    if((p>A[R])||(p<A[L])){
        printf("\n ERROR: VALUE OUTSIDE ARRAY \n");
        exit(MY_VALUE_ERROR);
    }
    while(L<R-1){
        k = (int) ((L+R)/2);
        if(p < A[k])
            R = k;
        else if(p > A[k])
            L = k;
        else
            L = R-1;
    }
    return L+1;
}

int int_sequence_remove_zeros(int *deg_seq, int N){
    int Nm1 = N-1;
    while(deg_seq[Nm1]==0){
        N = Nm1;
        Nm1--;
    }
    deg_seq = my_int_realloc(deg_seq, N);
    return N; // Here N is the number of non-zero degrees
}

int sign(double x){
    return (x >= 0) ? +1 : -1;
}

bool is_power_of_two(int n){
    return (n > 0) && ((n & (n - 1)) == 0);
}

bool almost_equal(double a, double b, double tol){
    return fabs(a - b) < tol;
}

// Random Number Generator [0, 1]
double RNG(){
    double x;
    x = ( (double) lrand48() ) / ((double) RAND_MAX);
    return x;
}

// Random Number Generator (0, 1]
double RNG_0(){
    double x;
    x = ( (double) lrand48() + 1.) / ((double) RAND_MAX + 1. );
    return x;
}

// Random Number Generator [0, 1)
double RNG_1(){
    double x;
    x = ( (double) lrand48() ) / ((double) RAND_MAX + 1. );
    return x;
}

double UNG(double a, double b){
    double x;
    x = a + (b-a) * ( (double) lrand48() + 1.) / ((double) RAND_MAX + 1. );
    return x;
}

double GNG(double mean, double std_dev){
    double x_1, x_2;
    double s, a;
    do{
        x_1 = ((double) mrand48())/((double) RAND_MAX);
        x_2 = ((double) mrand48())/((double) RAND_MAX);
        s = x_1*x_1 + x_2*x_2;
    } while (s==0 || s>=1);
    a = sqrt((-2*log(s))/s);
    return (x_1 * a * std_dev + mean);
    // We are wasting the other gaussian number (x_2 * a * dev_std + media)
}

double Positive_GNG(double mean, double std_dev){
    double x;
    do{
        x = GNG(mean, std_dev);
    } while (x < 0);
    return x;
}

double trivial_generator(double x, double y){ // Trivial Generator which always extract x
    return x;
}


// Integer RNG between min and max - induces small skew (see RandIntegers to avoid that)
int IRNG(int min, int max){
    return (int)(RNG_1() * (double)(max+1-min)) + min;
}

int RandIntegers(int nmin, int nmax){ // Inspired by https://www.cs.yale.edu/homes/aspnes/pinewiki/C(2f)Randomization.html
    long int n;
    int range = nmax - nmin + 1;
    long int rejection_lim = RAND_MAX - (RAND_MAX % range);
    while((n = lrand48()) >= rejection_lim);
    n = nmin + (n % range); 
    return (int) n;
}

int Init_Poisson(unsigned int *PoissTable, int psize, float lambda){
    int k=0;
    unsigned int zk; // Probability of k-th event
    double lok = 1.0; // lambda over k
    double eml = exp(-lambda); // exp minus lambda
    zk = eml * RAND_MAX + 0.5;
    PoissTable[0] = eml * RAND_MAX + 0.5;
    while(zk<RAND_MAX){
        PoissTable[k] = zk;   
        k++;
        lok = lok * lambda/k; 
        zk = zk + max_unsint((long int) (eml  * lok * RAND_MAX + 0.5), 1);
    }
    PoissTable[k] = zk; // RAND_MAX
    k++;
    PoissTable[k] = RAND_MAX + 1;
    k++; // PoissTable[k] is significant so size is k+1
    if(k>psize){
        printf("\n ERROR: POISSONIAN TABLE SIZE NOT SUFFICIENT \n");
        exit(MY_VALUE_ERROR);
    }
    PoissTable = my_unsigned_int_realloc(PoissTable, k);
    return k;
}

int poisson_RNG(unsigned int *PoissTable, int psize){
    int k;
    unsigned int q = lrand48();
    k = linear_search_interval(q, PoissTable, psize);
    return k;
}

void Init_PowerLaw(double *PLTable, double gamma, int k_min, int tsize){
    int i;
    double s = 0;
    // Create the table
    for(i=0; i<tsize; i++){
        PLTable[i] = pow((k_min + i), -gamma);
        s += PLTable[i];
    }
    // Normalise the table
    for(i=0; i<tsize; i++)
        PLTable[i] /= s;
    // Create the cumulative
    for(i=1; i<tsize; i++)
        PLTable[i] += PLTable[i-1];
    return;
}

void powerlaw_even_degree_seq(int *k, double *PLTable, int N, int k_min, int tsize){
    int ksum, i;
    double xi;
    bool odd = MY_TRUE;
    while(odd){
        ksum = 0; // Sum of all the degrees
        for(i=0; i<N; i++){
            xi = RNG_1(); // Uniform in [0, 1)
            k[i] = k_min + linear_search_interval(xi, PLTable, tsize);
            ksum  += k[i];
        }
        if((ksum%2) == 0) // This is 0 if the degree sequence is even, or 1 otherwise
            odd = MY_FALSE;
    }
    return;
}

int Init_Geometric(unsigned int *GTable, double c, int tsize){
    int k=0;
    unsigned int zk; // Probability of k-th event
    double coc1 = c / (c+1.0); // c over c+1
    double pk = 1.0 / (c+1.0);
    double s = 0;
    zk = pk * RAND_MAX + 0.5; // Il +0.5 serve per arrotondare alla maniera usuale (credo)
    //GTable[0] = zk;
    while(zk<RAND_MAX){
        GTable[k] = zk;
        k++;
        pk = pk * coc1;
        zk = zk + max_unsint((unsigned int) (pk * RAND_MAX + 0.5), 1);        
    }
    GTable[k] = zk; // zk >= RAND_MAX
    GTable[k+1] = RAND_MAX + 1;
    if(k>tsize){
        printf("\n ERROR: GEOMETRIC TABLE SIZE NOT SUFFICIENT \n");
        exit(MY_VALUE_ERROR);
    }
    GTable = my_unsigned_int_realloc(GTable, k);
    for(int i=0; i<k; i++){
    }
    return k;
}

int Geometric_RNG(unsigned int *GTable, int gsize){
    int k;
    unsigned int q = lrand48();
    k = linear_search_interval(q, GTable, gsize);
    return k;
}

void geometric_even_degree_seq(int *k, unsigned int *GTable, int N, int tsize){
    int ksum, i;
    unsigned int q;
    bool odd = MY_TRUE;
    while(odd){
        ksum = 0; // Sum of all the degrees
        for(i=0; i<N; i++){
            q = lrand48();
            k[i] = linear_search_interval(q, GTable, tsize);
            ksum  += k[i];
        }
        if((ksum%2) == 0) // This is 0 if the degree sequence is even, or 1 otherwise
            odd = MY_FALSE;
    }
    return;
}

double RNG_Exponential(double d){
    return - log(1 - RNG_1()) * d;
}

void Fisher_Yates_Shuffle(double *x, int n){ 
    int i, j; 
    double tmp; // create local variables to hold values for shuffle
    for (i=n-1; i>0; i--) { // for loop to shuffle
        j = lrand48() % (i+1); // Numero casuale tra 0 ed i
        tmp = x[j];
        x[j] = x[i];
        x[i] = tmp;
    }
    return;
}

void Uint_FY_Reshuffle(unsigned int *idx, int n){ 
    int i, j; 
    unsigned int tmp; // create local variables to hold values for shuffle
    for (i=n-1; i>0; i--) { // for loop to shuffle
        j = lrand48() % (i+1); // Numero casuale tra 0 ed i
        tmp = idx[j];
        idx[j] = idx[i];
        idx[i] = tmp;
    }
    return;
}

int random_sign(){ // Random +/- 1
    return (RNG() < 0.5) ? +1 : -1;
}

bool my_isfinite(double x){
    return (x * 0) == 0;
}

void time_from_seconds_to_hours(int s){
    int m, h;
    m = ((int) (s / 60));
    s = s % 60;
    h = ((int) (m / 60));
    m = m % 60;
    printf("%d h %d m %d s", h, m, s);
    return;
}

double* my_double_calloc(int size){
    double *dp;
    dp = (double*) calloc(size, sizeof(double));
    if (dp == NULL){
        printf("\n ERROR: DOUBLE CALLOC HAS FAILED \n");
        exit(MY_MEMORY_FAIL);
    }
    return dp;
}

double* my_double_malloc(int size){
    double *dp;
    dp = (double*) malloc(size * sizeof(double));
    if (dp == NULL){
        printf("\n ERROR: DOUBLE MALLOC HAS FAILED \n");
        exit(MY_MEMORY_FAIL);
    }
    return dp;
}

double* my_double_realloc(double *prev_dp, int new_size){
    double *new_dp; // Sfruttiamo dei puntatori di appoggio per evitare memory leak nel caso realloc fallisse
    new_dp = (double*) realloc(prev_dp, new_size * sizeof(double));
    if (new_dp == NULL){ 
        printf("\n ERROR: DOUBLE REALLOC HAS FAILED \n");
        free(prev_dp);
        exit(MY_MEMORY_FAIL);
    }
    return new_dp; // IMPORTANTE: Dobbiamo returnare un puntatore, una funzione void non funziona in questo caso (come già osservato con my_malloc, my_calloc e my_file)
}

double** my_double_arofars_malloc(int first_dim, int second_dim){
    double **arofars;
    int l;
    arofars = (double**) malloc(first_dim * sizeof(double*));
    if (arofars == NULL){
        printf("\n ERROR: DOUBLE ARRAY OF ARRAYS MALLOC HAS FAILED \n");
        exit(MY_MEMORY_FAIL);
    }
    for (l=0; l<first_dim; l++){
        *(arofars+l) = my_double_malloc(second_dim);
    }
    return arofars;
}

void my_double_arofars_reset(double** arofars, int first_dim, int second_dim){
    int i, j;
    for(i=0; i<first_dim; i++)
        for(j=0; j<second_dim; j++)
            arofars[i][j] = 0;
    return;
}

void my_double_arofars_free(double** arofars, int first_dim){
    int l;
    for (l=0; l<first_dim; l++){
        free(*(arofars+l));
    }
    free(arofars);
    return;
}

matrix* my_double_matrix_malloc(int first_dim, int second_dim){
    matrix *my_matrix;
    int l;
    my_matrix = (matrix*) malloc(sizeof(matrix));
    my_matrix->N_rows = first_dim;
    my_matrix->N_columns = second_dim;
    my_matrix->element = (double**) malloc(my_matrix->N_rows * sizeof(double*));
    if (my_matrix->element == NULL){
        printf("\n ERROR: DOUBLE MATRIX MALLOC HAS FAILED \n");
        exit(MY_MEMORY_FAIL);
    }
    for (l=0; l<my_matrix->N_rows; l++){
        *(my_matrix->element+l) = (double*) malloc(my_matrix->N_columns * sizeof(double));
        if (*(my_matrix->element+l) == NULL){
            printf("\n ERROR: DOUBLE MATRIX SUB-ARRAY MALLOC HAS FAILED \n");
            exit(MY_MEMORY_FAIL);
        }
    }
    return my_matrix;
}

matrix* my_double_matrix_calloc(int first_dim, int second_dim){
    matrix *my_matrix;
    int l;
    my_matrix = (matrix*) malloc(sizeof(matrix));
    my_matrix->N_rows = first_dim;
    my_matrix->N_columns = second_dim;
    my_matrix->element = (double**) malloc(my_matrix->N_rows * sizeof(double*));
    if (my_matrix->element == NULL){
        printf("\n ERROR: DOUBLE MATRIX MALLOC HAS FAILED \n");
        exit(MY_MEMORY_FAIL);
    }
    for (l=0; l<my_matrix->N_rows; l++){
        *(my_matrix->element+l) = (double*) calloc(my_matrix->N_columns, sizeof(double));
        if (*(my_matrix->element+l) == NULL){
            printf("\n ERROR: DOUBLE MATRIX SUB-ARRAY MALLOC HAS FAILED \n");
            exit(MY_MEMORY_FAIL);
        }
    }
    return my_matrix;
}

matrix* my_double_matrix_realloc(matrix *my_matrix, int new_dim){
    double **pdp_tmp;
    int old_dim;
    int l;
    old_dim = my_matrix->N_rows;
    my_matrix->N_rows = new_dim;
    // IMPORTANTE: In questo caso sfruttiamo dei puntatori di appoggio non solo per evitare memory leak nel caso malloc fallisse ma anche perché altrimenti il contenuto dei puntatori puntati andrebbe perso ed incorreremmo in segmentation fault (oltre a perdere i dati)
    pdp_tmp = (double**) realloc(my_matrix->element, my_matrix->N_rows * sizeof(double*));
    if (pdp_tmp == NULL){ 
        printf("\n ERROR: DOUBLE MATRIX REALLOC HAS FAILED \n");
        free(my_matrix->element);
        exit(MY_MEMORY_FAIL);
    }
    my_matrix->element = pdp_tmp;
    for(l=old_dim; l<my_matrix->N_rows; l++){
        *(my_matrix->element+l) = my_double_malloc(my_matrix->N_columns);
    }
    // IMPORTANTE: Dobbiamo returnare un puntatore, una funzione void non funziona in questo caso (come già osservato con my_malloc, my_calloc e my_file) ---> In caso dovremmo usare un puntatore al puntatore
    return my_matrix;
}

void my_free_matrix(matrix *my_matrix){
    int k;
    for (k=0; k<my_matrix->N_rows; k++){
        free(*(my_matrix->element+k));
    }
    free(my_matrix->element);
    free(my_matrix);
    return;
}

void my_save_dense_matrix_to_file(matrix *my_matrix, FILE *fp){
    int i, j;
    // Stampa su file della matrice interagente J_surv
    for(i=0; i<my_matrix->N_rows; i++){
        for(j=0; j<my_matrix->N_columns; j++){ // Questa volta stampiamo tutta la matrice quindi i cicli vanno fino a ecosystem->size
            fprintf(fp, "%lf\t", my_matrix->element[i][j]);
            //printf("%lf \t", jac->element[i][j]);
        }
        fprintf(fp, "\n");
        //printf("\n");
    }
    return;
}

int* my_int_malloc(int size){
    int *pint;
    pint = (int*) malloc(size * sizeof(int));
    if (pint==NULL) {
        printf ("\nERROR: Failed allocayion of int array.\n");
        exit (MY_MEMORY_FAIL);
    }
    return pint;
}

int* my_int_calloc(int size){
    int *pint;
    pint = (int*) calloc(size, sizeof(int));
    if (pint==NULL) {
        printf ("\nERROR: Failed calloc of int array.\n");
        exit (MY_MEMORY_FAIL);
    }
    return pint;
}

int* my_int_realloc(int *pint, int new_size){
    int *pint_appoggio;
    pint_appoggio = (int*) realloc(pint, new_size * sizeof(int));
    if (pint_appoggio == NULL){ 
        printf("\n ERROR: INT REALLOC HAS FAILED \n");
        free(pint);
        exit(MY_MEMORY_FAIL);
    }
    return pint_appoggio;
}

unsigned int* my_unsigned_int_malloc(int size){
    unsigned int *p;
    p = (unsigned int*) malloc(size * sizeof(unsigned int));
    if (p == NULL){
        printf("\n ERROR: UNSIGNED INT MALLOC HAS FAILED \n");
        exit(MY_MEMORY_FAIL);
    }
    return p;
}

unsigned int* my_unsigned_int_realloc(unsigned int *prev_p, int new_size){
    unsigned int *new_p;
    new_p = (unsigned int*) realloc(prev_p, new_size * sizeof(unsigned int));
    if (new_p == NULL){ 
        printf("\n ERROR: UNSIGNED INT REALLOC HAS FAILED \n");
        free(prev_p);
        exit(MY_MEMORY_FAIL);
    }
    return new_p;
}

char* my_char_malloc(int size){
    char *buffer;
    buffer = (char*) malloc(size * sizeof(char));
    if (buffer==NULL) {
        printf ("\nERROR: Failed malloc of file buffer.\n");
        exit (MY_MEMORY_FAIL);
    }
    return buffer;
}

char* my_char_realloc(char *prev_buffer, int new_size){
    char *new_buffer;
    new_buffer = (char*) realloc(prev_buffer, new_size * sizeof(char));
    if (new_buffer==NULL) {
        printf ("\nERROR: Failed realloc of char buffer.\n");
        free(prev_buffer);
        exit (MY_MEMORY_FAIL);
    }
    return new_buffer;
}

const char* my_int_to_str(int n, int lenght){
    char *buffer;
    buffer = my_char_malloc(lenght);
    snprintf(buffer, sizeof(char) * lenght, "%d", n);
    return (const char*) buffer;
}

FILE* my_open_writing_file(const char* file_name){
    FILE* file_pointer;
    file_pointer = fopen(file_name, "w");
    if (file_pointer==NULL) {
        printf("Error while opening %s \n", file_name);
        exit(OPEN_FILE_ERROR);
    }
    return file_pointer;
}

FILE* my_open_appending_file(const char* file_name){
    FILE* file_pointer;
    file_pointer = fopen(file_name, "a");
    if (file_pointer==NULL) {
        printf("Error while opening %s \n", file_name);
        exit(OPEN_FILE_ERROR);
    }
    return file_pointer;
}

FILE* my_open_reading_file(const char* file_name){
    FILE* file_pointer;
    file_pointer = fopen(file_name, "r");
    if (file_pointer==NULL) {
        printf("Error while opening %s \n", file_name);
        exit(OPEN_FILE_ERROR);
    }
    return file_pointer;
}

FILE* my_open_writing_binary_file(const char* file_name){
    FILE* file_pointer;
    file_pointer = fopen(file_name, "wb");
    if (file_pointer==NULL) {
        printf("Error while opening %s \n", file_name);
        exit(OPEN_FILE_ERROR);
    }
    return file_pointer;
}

FILE* my_open_reading_binary_file(const char* file_name){
    FILE* file_pointer;
    file_pointer = fopen(file_name, "rb");
    if (file_pointer==NULL) {
        printf("Error while opening %s \n", file_name);
        exit(OPEN_FILE_ERROR);
    }
    return file_pointer;
}

void seeds_from_dev_random_and_time(time_t *my_seed) {
    FILE* fp = fopen("/dev/urandom", "rb");
    if (fp){ // fp is a valid pointer
        int read_bytes = fread(my_seed, sizeof(my_seed), 1, fp); // Returns number of read element, in our case is one or zero 
        fclose(fp);
        if (read_bytes != 1)
            *my_seed = 0;
    }
    else
        *my_seed = 0;

    // Use XOR to mix with current time for extra entropy
    *my_seed ^= time(0); // *my_seed = *my_seed ^ time(0);
    return;
}

int count_pattern_in_dir(const char* file_pattern, const char* directory){
    int count=0;
    char *command_string;
    FILE* fp;
    command_string = my_char_malloc(CHAR_LENGHT);
    snprintf(command_string, CHAR_LENGHT, "ls -1f %s | grep '%s' | wc -l", directory, file_pattern);
    // Open a pipe to read the command's output
    fp = popen(command_string, "r");
    if(fp == NULL){
        perror("popen failed");
        return MY_FAIL;
    }
    //fscanf(fp, "%d", &count);
    if(fscanf(fp, "%d", &count) != 1){
        fprintf(stderr, "Failed to parse count\n");
        pclose(fp);
        return MY_FAIL;
    }
    pclose(fp);
    free(command_string);
    return count;
}

void my_create_directory(const char* dir_name, int lenght){
    char *buffer;
    buffer = my_char_malloc(lenght);
    snprintf(buffer, sizeof(char) * lenght, "mkdir %s", dir_name);
    system(buffer);
    return;
}

/**
 * @brief Parses the command line arguments of Ecosystem_Dynamics_on_Graphs_in_Temperature.cpp.
 *
 * Every parameter must already hold its default value when this function is called (the caller
 * sets the defaults, this function only overwrites the ones for which a flag is found in argv).
 * For the non-integer (double) parameters, both the numeric value AND its exact command-line
 * string ("label") are kept: the label is reused verbatim to build output file names, so that
 * file names reflect exactly what the user typed instead of a value reformatted by printf
 * (which could otherwise silently change file names when re-running with the same arguments).
 * The label buffers (c_label, mu_label, sigma_label, epsilon_label, T_label) must be pre-allocated
 * by the caller (e.g. with my_char_malloc(CHAR_LENGHT)) before calling this function, since it
 * writes into them with snprintf.
 *
 * @param argc                number of command line arguments, as received by main
 * @param argv                command line arguments, as received by main
 * @param N              [in/out] number of species in the ecosystem
 * @param c              [in/out] average connectivity of the interaction graph
 * @param c_label        [out]    string representation of c, exactly as passed on the command line
 * @param mu             [in/out] average strength of the interactions
 * @param mu_label       [out]    string representation of mu, exactly as passed on the command line
 * @param sigma          [in/out] standard deviation of the interactions
 * @param sigma_label    [out]    string representation of sigma, exactly as passed on the command line
 * @param epsilon        [in/out] level of asymmetry of the interactions
 * @param epsilon_label  [out]    string representation of epsilon, exactly as passed on the command line
 * @param T              [in/out] temperature of the dynamics
 * @param T_label        [out]    string representation of T, exactly as passed on the command line
 * @param N_ext          [in/out] number of graph extractions to perform
 * @param N_previous_ext [in/out] number of extractions already performed in a previous run (offset used for numbering output files)
 * @param N_meas         [in/out] number of measures (independent initial conditions) per extraction
 * @param lambda         [in/out] immigration rate / extinction threshold of the demographic noise dynamics
 * @param t_max          [in/out] maximum simulated time for each measure
 * @param deltat_save    [in/out] simulated time between two consecutive trajectory samples
 * @param print_hist     [out]    set to true if --print_hist is found; if true, species trajectories are written to file
 * @param print_avgs     [out]    set to true if --print_avgs is found; if true, final average abundances are written to file
 * @param print_graph    [out]    set to true if --print_graph is found; if true, the extracted interaction graph is additionally written to file for inspection (the ecosystem itself is always loaded directly from memory)
 * @param print_init     [out]    set to true if --print_init is found; if true, the initial conditions are additionally written to a log file for inspection (the ecosystem state is always set directly in memory)
 * @param print_parameters [out]  set to true if --print_parameters is found; the caller should then call print_params()
 */
void parse_arguments(int argc, char *argv[], int &N, double &c, char *c_label,
                      double &mu, char *mu_label, double &sigma, char *sigma_label,
                      double &epsilon, char *epsilon_label, double &T, char *T_label,
                      int &N_ext, int &N_previous_ext, int &N_meas, double &lambda,
                      double &t_max, double &deltat_save, bool &print_hist, bool &print_avgs,
                      bool &print_graph, bool &print_init, bool &print_parameters){
    int arg_index = 1;
    while(arg_index < argc){
        if(strcmp(argv[arg_index], "-h") == 0 || strcmp(argv[arg_index], "--help") == 0){
            fprintf(stderr, "Usage: %s [options]\n", argv[0]);
            fprintf(stderr, "The following list describes the command line arguments\n");
            fprintf(stderr, "the structure is --arg_name  [data_type: default]  ::  description\n");
            fprintf(stderr, "-N or --size  [int: 256]  ::  number of species in the ecosystem\n");
            fprintf(stderr, "-c or --connect  [double: 3.000]  ::  average connectivity of the interaction graph\n");
            fprintf(stderr, "--mu  [double: 0.100]  ::  average strength of the interactions\n");
            fprintf(stderr, "--sigma  [double: 0.000]  ::  standard deviation of the interactions\n");
            fprintf(stderr, "--epsilon  [double: 1.000]  ::  level of asymmetry of the interactions\n");
            fprintf(stderr, "-T or --temp  [double: 0.010]  ::  temperature of the dynamics\n");
            fprintf(stderr, "--N_ext  [int: 1]  ::  number of graph extractions to perform\n");
            fprintf(stderr, "--N_prev_ext  [int: 0]  ::  number of extractions already performed in a previous run (offset used for numbering output files)\n");
            fprintf(stderr, "--N_meas  [int: 1]  ::  number of measures (independent initial conditions) per extraction\n");
            fprintf(stderr, "--lambda  [double: 1e-06]  ::  immigration rate / extinction threshold of the demographic noise dynamics\n");
            fprintf(stderr, "--t_max  [double: 4000]  ::  maximum simulated time for each measure\n");
            fprintf(stderr, "--deltat_save  [double: 2.0]  ::  simulated time between two consecutive trajectory samples\n");
            fprintf(stderr, "--print_hist  ::  if this flag is added to the arguments, the program will print the species trajectories to file\n");
            fprintf(stderr, "--print_avgs  ::  if this flag is added to the arguments, the program will print the final average abundances to file\n");
            fprintf(stderr, "--print_graph  ::  if this flag is added to the arguments, the extracted interaction graph is additionally written to file for inspection\n");
            fprintf(stderr, "--print_init  ::  if this flag is added to the arguments, the initial conditions are additionally written to a log file for inspection\n");
            fprintf(stderr, "--print_parameters  ::  if this flag is added to the arguments, the program will print the parameters used for the run\n");
            exit(MY_SUCCESS);
        }else if(strcmp(argv[arg_index], "-N") == 0 || strcmp(argv[arg_index], "--size") == 0){
            arg_index++;
            N = atoi(argv[arg_index]);
            arg_index++;
        }else if(strcmp(argv[arg_index], "-c") == 0 || strcmp(argv[arg_index], "--connect") == 0){
            arg_index++;
            snprintf(c_label, CHAR_LENGHT, "%s", argv[arg_index]);
            c = atof(c_label);
            arg_index++;
        }else if(strcmp(argv[arg_index], "--mu") == 0){
            arg_index++;
            snprintf(mu_label, CHAR_LENGHT, "%s", argv[arg_index]);
            mu = atof(mu_label);
            arg_index++;
        }else if(strcmp(argv[arg_index], "--sigma") == 0){
            arg_index++;
            snprintf(sigma_label, CHAR_LENGHT, "%s", argv[arg_index]);
            sigma = atof(sigma_label);
            arg_index++;
        }else if(strcmp(argv[arg_index], "--epsilon") == 0){
            arg_index++;
            snprintf(epsilon_label, CHAR_LENGHT, "%s", argv[arg_index]);
            epsilon = atof(epsilon_label);
            arg_index++;
        }else if(strcmp(argv[arg_index], "-T") == 0 || strcmp(argv[arg_index], "--temp") == 0){
            arg_index++;
            snprintf(T_label, CHAR_LENGHT, "%s", argv[arg_index]);
            T = atof(T_label);
            arg_index++;
        }else if(strcmp(argv[arg_index], "--N_ext") == 0){
            arg_index++;
            N_ext = atoi(argv[arg_index]);
            arg_index++;
        }else if(strcmp(argv[arg_index], "--N_prev_ext") == 0){
            arg_index++;
            N_previous_ext = atoi(argv[arg_index]);
            arg_index++;
        }else if(strcmp(argv[arg_index], "--N_meas") == 0){
            arg_index++;
            N_meas = atoi(argv[arg_index]);
            arg_index++;
        }else if(strcmp(argv[arg_index], "--lambda") == 0){
            arg_index++;
            lambda = atof(argv[arg_index]);
            arg_index++;
        }else if(strcmp(argv[arg_index], "--t_max") == 0){
            arg_index++;
            t_max = atof(argv[arg_index]);
            arg_index++;
        }else if(strcmp(argv[arg_index], "--deltat_save") == 0){
            arg_index++;
            deltat_save = atof(argv[arg_index]);
            arg_index++;
        }else if(strcmp(argv[arg_index], "--print_hist") == 0){
            print_hist = true;
            arg_index++;
        }else if(strcmp(argv[arg_index], "--print_avgs") == 0){
            print_avgs = true;
            arg_index++;
        }else if(strcmp(argv[arg_index], "--print_graph") == 0){
            print_graph = true;
            arg_index++;
        }else if(strcmp(argv[arg_index], "--print_init") == 0){
            print_init = true;
            arg_index++;
        }else if(strcmp(argv[arg_index], "--print_parameters") == 0){
            print_parameters = true;
            arg_index++;
        }else{
            fprintf(stderr, "\n ERROR: Unknown argument: %s\n", argv[arg_index]);
            exit(MY_VALUE_ERROR);
        }
    }
    return;
}

/**
 * @brief Prints to stderr the parameters that will be used for the run.
 *
 * Meant to be called right after parse_arguments(), and only when the user requested it
 * via the --print_parameters flag, so that a run's parameters can be logged/checked without
 * having to grep them back out of the output file names.
 *
 * @param N              number of species in the ecosystem
 * @param c              average connectivity of the interaction graph
 * @param c_label        string representation of c, as used to build output file names
 * @param mu             average strength of the interactions
 * @param mu_label       string representation of mu, as used to build output file names
 * @param sigma          standard deviation of the interactions
 * @param sigma_label    string representation of sigma, as used to build output file names
 * @param epsilon        level of asymmetry of the interactions
 * @param epsilon_label  string representation of epsilon, as used to build output file names
 * @param T              temperature of the dynamics
 * @param T_label        string representation of T, as used to build output file names
 * @param N_ext          number of graph extractions to perform
 * @param N_previous_ext number of extractions already performed in a previous run
 * @param N_meas         number of measures (independent initial conditions) per extraction
 * @param lambda         immigration rate / extinction threshold of the demographic noise dynamics
 * @param t_max          maximum simulated time for each measure
 * @param deltat_save    simulated time between two consecutive trajectory samples
 * @param print_hist     whether species trajectories are written to file
 * @param print_avgs     whether final average abundances are written to file
 * @param print_graph    whether the extracted interaction graph is additionally written to file for inspection
 * @param print_init     whether the initial conditions are additionally written to a log file for inspection
 */
void print_params(int N, double c, char *c_label, double mu, char *mu_label,
                   double sigma, char *sigma_label, double epsilon, char *epsilon_label,
                   double T, char *T_label, int N_ext, int N_previous_ext, int N_meas,
                   double lambda, double t_max, double deltat_save, bool print_hist,
                   bool print_avgs, bool print_graph, bool print_init){
    fprintf(stderr, "\n ### RUN PARAMETERS ###\n");
    fprintf(stderr, "Number of species (N): %d\n", N);
    fprintf(stderr, "Average connectivity (c): %s (%g)\n", c_label, c);
    fprintf(stderr, "Average interaction strength (mu): %s (%g)\n", mu_label, mu);
    fprintf(stderr, "Interaction standard deviation (sigma): %s (%g)\n", sigma_label, sigma);
    fprintf(stderr, "Interaction asymmetry (epsilon): %s (%g)\n", epsilon_label, epsilon);
    fprintf(stderr, "Temperature (T): %s (%g)\n", T_label, T);
    fprintf(stderr, "Number of graph extractions (N_ext): %d\n", N_ext);
    fprintf(stderr, "Number of extractions already performed (N_prev_ext): %d\n", N_previous_ext);
    fprintf(stderr, "Number of measures per extraction (N_meas): %d\n", N_meas);
    fprintf(stderr, "Immigration rate / extinction threshold (lambda): %g\n", lambda);
    fprintf(stderr, "Maximum simulated time per measure (t_max): %g\n", t_max);
    fprintf(stderr, "Time between trajectory samples (deltat_save): %g\n", deltat_save);
    fprintf(stderr, "Print species trajectories (print_hist): %s\n", BOOL_STR(print_hist));
    fprintf(stderr, "Print final average abundances (print_avgs): %s\n", BOOL_STR(print_avgs));
    fprintf(stderr, "Write the interaction graph to file (print_graph): %s\n", BOOL_STR(print_graph));
    fprintf(stderr, "Write the initial conditions to file (print_init): %s\n", BOOL_STR(print_init));
    fprintf(stderr, " #######################\n\n");
    return;
}

