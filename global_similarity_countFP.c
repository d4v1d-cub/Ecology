#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_FILENAME 512
#define MAX_LINE 1024
#define HISTOGRAM_BINS 100

// Funzione per creare il nome del file con i parametri
void create_filename(char* filename, size_t size, 
                     double eps, double mu, double sigma, int N, int c, int seedgraph,
                     double av0, double dn, double tol, int maxiter, double damping,
                     int fixedpoint) {
    snprintf(filename, size,
             "IBMF_T0_seq_gr_inside_RRG_eps_%.3f_mu_%.3f_sigma_%.3f_N_%d_c_%d_seedgraph_%d_"
             "Lotka_Volterra_final_av0_%.3f_dn_%.3f_tol_%.1e_maxiter_%d_damping_%.2f_"
             "fixedpoint_%d.txt",
             eps, mu, sigma, N, c, seedgraph, av0, dn, tol, maxiter, damping, fixedpoint);
}

// Funzione per leggere un file e memorizzare i valori in un array
int read_file_into_array(const char* filename, double* values, int N) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("  Errore: impossibile aprire il file %s\n", filename);
        return 0;
    }
    
    char line[MAX_LINE];
    int line_count = 0;
    
    // Inizializza l'array con un valore sentinella (es. NAN) per controllare dopo
    for (int i = 0; i < N; i++) {
        values[i] = -999.0;  // Valore sentinella per indicare mancante
    }
    
    // Leggi i dati
    while (fgets(line, sizeof(line), file)) {
        // Salta righe vuote o che iniziano con #
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        
        int index;
        double value;
        
        if (sscanf(line, "%d %lf", &index, &value) == 2) {
            if (index >= 0 && index < N) {
                values[index] = value;
                line_count++;
            }
        }
    }
    
    fclose(file);
    
    // Verifica che abbiamo letto tutti gli N valori
    for (int i = 0; i < N; i++) {
        if (values[i] == -999.0) {
            printf("  Attenzione: file %s manca del valore per indice %d\n", filename, i);
            return 0;  // File incompleto
        }
    }
    
    return line_count;
}

int main(int argc, char* argv[]) {
    // Parametri da riga di comando
    if (argc !=13) {
        printf("Uso: %s eps mu sigma N c seedgraph av0 dn tol maxiter damping n_fixedpoints\n", argv[0]);
        printf("Esempio: %s 0.000 0.400 0.000 1024 3 3 0.500 0.400 1.0e-06 10000 1.00 2\n", argv[0]);
        return 1;
    }
    
    double eps = atof(argv[1]);
    double mu = atof(argv[2]);
    double sigma = atof(argv[3]);
    int N = atoi(argv[4]);
    int c = atoi(argv[5]);
    int seedgraph = atoi(argv[6]);
    double av0 = atof(argv[7]);
    double dn = atof(argv[8]);
    double tol = atof(argv[9]);
    int maxiter = atoi(argv[10]);
    double damping = atof(argv[11]);
    int fixedpoints = atoi(argv[12]);
    int total_files_expected=fixedpoints;
    
    // Calcola il numero totale di file
    
    printf("Parametri:\n");
    printf("  eps = %.3f, mu = %.3f, sigma = %.3f\n", eps, mu, sigma);
    printf("  N = %d, c = %d, seedgraph = %d\n", N, c, seedgraph);
    printf("  av0 = %.3f, dn = %.3f, tol = %.1e, maxiter = %d, damping = %.2f\n", 
           av0, dn, tol, maxiter, damping);
    printf("  Totale file attesi: %d\n", fixedpoints);
    
    if (fixedpoints < 2) {
        printf("Errore: servono almeno 2 file per calcolare le similarità\n");
        return 1;
    }
    
    // Per memorizzare i nomi dei file e i loro dati
    char** filenames = malloc(total_files_expected * sizeof(char*));
    double** file_data = malloc(total_files_expected * sizeof(double*));
    int files_read = 0;
    
    if (!filenames || !file_data) {
        printf("Errore di allocazione memoria per gli array di file\n");
        return 1;
    }
    
    // Leggi tutti i file
    int file_index = 0;
    for (fixedpoints=1;fixedpoints<= total_files_expected; fixedpoints++){
      char filename[MAX_FILENAME];
      create_filename(filename, sizeof(filename), 
		      eps, mu, sigma, N, c, seedgraph, av0, dn, tol, maxiter, damping,fixedpoints);
      
      printf("Leggo file %d/%d: %s\n", file_index + 1, total_files_expected, filename);
      
      // Alloca memoria per i dati di questo file
      double* values = malloc(N * sizeof(double));
      if (!values) {
	printf("  Errore di allocazione per i dati del file\n");
	file_index++;
	continue;
      }
      
      if (read_file_into_array(filename, values, N) == N) {
	// File letto con successo
	filenames[files_read] = malloc(strlen(filename) + 1);
	strcpy(filenames[files_read], filename);
	file_data[files_read] = values;
	files_read++;
	printf("  File letto con successo (%d valori)\n", N);
      } else {
	// File non valido, libera la memoria
	free(values);
	printf("  File saltato (incompleto o non leggibile)\n");
      }
      
      file_index++;
    }
    
    
    printf("\nTotale file letti con successo: %d\n", files_read);
    
    if (files_read < 2) {
        printf("Errore: servono almeno 2 file validi per calcolare le similarità\n");
        for (int i = 0; i < files_read; i++) {
            free(file_data[i]);
            free(filenames[i]);
        }
        free(file_data);
        free(filenames);
        return 1;
    }
    
    // Calcola similarità tra tutte le coppie di file
    int num_pairs = (files_read * (files_read - 1)) / 2;
    printf("Calcolo similarità tra %d file (%d coppie)\n", files_read, num_pairs);
    printf("Similarità = sum_i (1 - |x_i^a - x_i^b|)\n\n");
    
    double* similarities = malloc(num_pairs * sizeof(double));
    if (!similarities) {
        printf("Errore di allocazione per similarities\n");
        for (int i = 0; i < files_read; i++) {
            free(file_data[i]);
            free(filenames[i]);
        }
        free(file_data);
        free(filenames);
        return 1;
    }
    
    int sim_idx = 0;
    for (int i = 0; i < files_read; i++) {
        for (int j = i + 1; j < files_read; j++) {
            double sim = 0.0;
            
            // Calcola somma delle similarità punto-punto
            for (int k = 0; k < N; k++) {
                sim += 1.0 - fabs(file_data[i][k] - file_data[j][k]);
            }
            
            similarities[sim_idx] = sim/((double)N);
            
            // Opzionale: stampa le prime 10 similarità come esempio
            if (sim_idx < 10) {
                printf("  sim[%d,%d] = %.6f\n", i, j, sim);
            }
            
            sim_idx++;
        }
    }
    
    printf("\nCalcolate %d similarità\n", sim_idx);
    
    if (sim_idx == 0) {
        printf("Nessuna similarità calcolata\n");
        free(similarities);
        for (int i = 0; i < files_read; i++) {
            free(file_data[i]);
            free(filenames[i]);
        }
        free(file_data);
        free(filenames);
        return 1;
    }
    
    // Calcola statistiche di base
    double mean = 0.0;
    double min = similarities[0];
    double max = similarities[0];
    
    for (int i = 0; i < sim_idx; i++) {
        mean += similarities[i];
        if (similarities[i] < min) min = similarities[i];
        if (similarities[i] > max) max = similarities[i];
    }
    mean /= sim_idx;
    
    double variance = 0.0;
    for (int i = 0; i < sim_idx; i++) {
        variance += (similarities[i] - mean) * (similarities[i] - mean);
    }
    variance /= sim_idx;
    double stddev = sqrt(variance);
    
    printf("\nStatistiche delle similarità tra file:\n");
    printf("  Media: %.6f\n", mean);
    printf("  Deviazione standard: %.6f\n", stddev);
    printf("  Minimo: %.6f\n", min);
    printf("  Massimo: %.6f\n", max);
    printf("  Range: [%.6f, %.6f]\n", min, max);
    
    // Crea istogramma
    int histogram[HISTOGRAM_BINS] = {0};
    double bin_width = (max - min) / HISTOGRAM_BINS;
    
    // Evita divisione per zero se max == min
    if (bin_width == 0) {
        bin_width = 1.0 / HISTOGRAM_BINS;
    }
    
    for (int i = 0; i < sim_idx; i++) {
        int bin = (int)((similarities[i] - min) / bin_width);
        if (bin < 0) bin = 0;
        if (bin >= HISTOGRAM_BINS) bin = HISTOGRAM_BINS - 1;
        histogram[bin]++;
    }
    
    // Salva istogramma su file
    char hist_filename[MAX_FILENAME];
    snprintf(hist_filename, sizeof(hist_filename),
             "file_similarity_histogram_eps_%.3f_mu_%.3f_sigma_%.3f_N_%d_c_%d_seedgraph_%d_"
             "av0_%.3f_dn_%.3f_fixedpoints_%d.txt",
             eps, mu, sigma, N, c, seedgraph, av0, dn,
             files_read);
    
    FILE* hist_file = fopen(hist_filename, "w");
    if (!hist_file) {
        printf("Errore: impossibile creare il file %s\n", hist_filename);
        free(similarities);
        for (int i = 0; i < files_read; i++) {
            free(file_data[i]);
            free(filenames[i]);
        }
        free(file_data);
        free(filenames);
        return 1;
    }
    
    fprintf(hist_file, "# Istogramma delle similarità tra file\n");
    fprintf(hist_file, "# Similarità = sum_i (1 - |x_i^a - x_i^b|)\n");
    fprintf(hist_file, "# Parametri: eps=%.3f, mu=%.3f, sigma=%.3f, N=%d, c=%d, seedgraph=%d\n", 
            eps, mu, sigma, N, c, seedgraph);
    fprintf(hist_file, "# av0=%.3f, dn=%.3f, tol=%.1e, maxiter=%d, damping=%.2f\n", 
            av0, dn, tol, maxiter, damping);
    fprintf(hist_file, "# Totale file: %d, File letti: %d\n", total_files_expected, files_read);
    fprintf(hist_file, "# Totale coppie: %d\n", sim_idx);
    fprintf(hist_file, "# Min similarità: %.6f, Max similarità: %.6f\n", min, max);
    fprintf(hist_file, "# bin_center\tcount\tfrequency\n");
    
    for (int i = 0; i < HISTOGRAM_BINS; i++) {
        double bin_center = min + (i + 0.5) * bin_width;
        double frequency = (double)histogram[i] / sim_idx;
        fprintf(hist_file, "%.6f\t%d\t%.6f\n", bin_center, histogram[i], frequency);
    }
    
    fclose(hist_file);
    printf("\nIstogramma salvato in: %s\n", hist_filename);
    
    // Opzionale: stampa la matrice di similarità (solo se il numero di file è ragionevole)
    if (files_read <= 20) {
        printf("\nMatrice di similarità (primi %d file):\n", files_read);
        printf("File\\File");
        for (int j = 0; j < files_read; j++) {
            printf("\tF%d", j);
        }
        printf("\n");
        
        for (int i = 0; i < files_read; i++) {
            printf("F%d\t", i);
            for (int j = 0; j < files_read; j++) {
                if (i == j) {
                    printf("\t-");
                } else if (j > i) {
                    // Calcola l'indice nella lista delle similarità
                    int idx = (i * (2 * files_read - i - 1)) / 2 + (j - i - 1);
                    printf("\t%.3f", similarities[idx]);
                } else {
                    // Per j < i, usa simmetria
                    int idx = (j * (2 * files_read - j - 1)) / 2 + (i - j - 1);
                    printf("\t%.3f", similarities[idx]);
                }
            }
            printf("\n");
        }
    }
    
    // Libera memoria
    free(similarities);
    for (int i = 0; i < files_read; i++) {
        free(file_data[i]);
        free(filenames[i]);
    }
    free(file_data);
    free(filenames);
    
    printf("\nMemoria liberata correttamente\n");
    
    return 0;
}
