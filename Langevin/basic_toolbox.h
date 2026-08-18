#ifndef _BASIC_TOOLBOX_H_ // Include Guard

    #define _BASIC_TOOLBOX_H_ 

    /* PREPROCESSOR INCLUSION */
    #include <cmath>
    #include <ctime>
    #include <cstring>
    #include <cstdio> 
    #include <cstdlib>

    /* PREPROCESSOR DEFINITION */
    #define MY_SUCCESS  1
    #define MY_FAIL   0
    #define MY_TRUE  1
    #define MY_FALSE   0
    #define MY_TROUBLE   -7
    #define MY_MEMORY_FAIL   -8
    #define MY_VALUE_ERROR   -9
    #define OPEN_FILE_ERROR   -10
    #define CHAR_LENGHT     250

    #define CHECK_ALLOC(ptr, ptr_type)                                                 \
        if(ptr == NULL){                                                              \
            fprintf(stderr, "Memory allocation of %s failed\n", ptr_type);             \
            exit(MY_FAIL);                                                             \
        }

    #define BOOL_STR(b) ((b) ? "true" : "false")

    /* STRUCTURE */
    struct matrix_struct{
        double **element;
        int N_rows;
        int N_columns;
    };

    /* TYPE DEFINITION */
    typedef struct matrix_struct matrix;

    /* PROTOTYPES */

    int compare_desc(const void *a, const void *b);
    int my_max(int a, int b);
    double sum_array(double *x, int N);
    int sum_array(int *n, int N);
    int max_array(int *n, int N);
    void loadArray(FILE *fp, int *x, int size);
    void fill_array(int *x, int m, int N);
    bool are_arrays_equal(double *x, double *y, int dimension, double delta);
    bool is_array_in(double *array, double **matrix, int first_dim, int second_dim, double delta, int *index_pointer);
    void initialize_double_array(double* x, double a, int x_dimension);
    void equalize_double_array(double* x_1, double* x_2, int x_dimension);
    void combine_double_array(double alfa, double* x_1, double beta, double* x_2, double* x, int x_dimension);
    double* exp_double_array(double *x, int size);
    void swap(double *x_ptr, double *y_ptr);
    void swap(int *x_ptr, int *y_ptr);
    void BubbleSort(double *x, int n, char order='D');
    void BubbleSort(int *x, int n, char order='D');
    unsigned int max_unsint(unsigned int a, unsigned int b);
    int linear_search_interval(unsigned int p, unsigned int *A, int N);
    int linear_search_interval(double p, double *A, int N);
    int binary_search_interval(double p, double *A, int N);
    int int_sequence_remove_zeros(int *deg_seq, int N);
    int sign(double x);
    bool is_power_of_two(int n);
    bool almost_equal(double a, double b, double tol);

    double RNG(); // Generatore numeri casuali [0, 1]
    double RNG_0(); // Generatore numeri casuali (0, 1]
    double RNG_1(); // Generatore numeri casuali [0, 1)
    double UNG(double a, double b);
    double GNG(double mean, double std_dev); // Potremmo facilmente generarne due ma per ora ci limitiamo a sprecare quest'opportunità
    double Positive_GNG(double mean, double std_dev);
    double trivial_generator(double x, double y);
    int IRNG(int min, int max);
    int RandIntegers(int nmin, int nmax);
    int Init_Poisson(unsigned int *PoissTable, int psize, float lambda);
    int poisson_RNG(unsigned int *PoissTable, int psize);
    void Init_PowerLaw(double *PLTable, double gamma, int k_min, int tsize);
    void powerlaw_even_degree_seq(int *k, double *PLTable, int N, int k_min, int tsize);
    int Init_Geometric(unsigned int *GTable, double c, int tsize);
    int Geometric_RNG(unsigned int *GTable, int gsize);
    void geometric_even_degree_seq(int *k, unsigned int *GTable, int N, int tsize);
    double RNG_Exponential(double d);
    void Fisher_Yates_Shuffle(double *x, int n);
    void Uint_FY_Reshuffle(unsigned int *idx, int n);
    int random_sign();

    bool my_isfinite(double x);
    void time_from_seconds_to_hours(int s);

    double* my_double_calloc(int size);
    double* my_double_malloc(int size);
    double* my_double_realloc(double *prev_dp, int new_size);
    double** my_double_arofars_malloc(int first_dim, int second_dim);
    void my_double_arofars_reset(double** arofars, int first_dim, int second_dim);
    void my_double_arofars_free(double** arofars, int first_dim);
    matrix* my_double_matrix_malloc(int first_dim, int second_dim);
    matrix* my_double_matrix_calloc(int first_dim, int second_dim);
    matrix* my_double_matrix_realloc(matrix *my_matrix, int new_dim);
    void my_free_matrix(matrix *my_matrix);
    void my_save_dense_matrix_to_file(matrix *my_matrix, FILE *fp);

    int* my_int_malloc(int size);
    int* my_int_calloc(int size);
    int* my_int_realloc(int *pint, int new_size);
    unsigned int* my_unsigned_int_malloc(int size);
    unsigned int* my_unsigned_int_realloc(unsigned int *prev_p, int new_size);
    char* my_char_malloc(int size);
    char* my_char_realloc(char *prev_buffer, int new_size);

    const char* my_int_to_str(int n, int lenght);
    FILE* my_open_writing_file(const char* file_name);
    FILE* my_open_appending_file(const char* file_name);
    FILE* my_open_reading_file(const char* file_name);
    FILE* my_open_writing_binary_file(const char* file_name);
    FILE* my_open_reading_binary_file(const char* file_name);

    void seeds_from_dev_random_and_time(time_t *my_seed);
    int count_pattern_in_dir(const char* file_pattern, const char* directory);
    void my_create_directory(const char* dir_name, int lenght);

#endif // _GRAPH_LV_TOOLBOX_H_