#include "spmv_benchmarks.h"


static double wtime() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}


static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(double *arr, int n) {
    double *tmp = (double *)malloc(n * sizeof(double));
    memcpy(tmp, arr, n * sizeof(double));
    qsort(tmp, n, sizeof(double), cmp_double);
    double m = (n % 2 == 0) ? (tmp[n/2-1] + tmp[n/2]) / 2.0 : tmp[n/2];
    free(tmp);
    return m;
}

void evaluate_bc_matvecs(
    int nx,
    int ny,
    int nz,
    int K,
    FILE *csv,
    double *gs_mean,
    double *gs_median,
    int    *gs_count
) {
    int N = nx * ny * nz;

    printf("=============================================================\n");
    printf("Avaliação: malha %dx%dx%d (N=%d, 3N=%d), K=%d iterações\n",
           nx, ny, nz, N, 3*N, K);
    printf("=============================================================\n");

    BlockedCSR *A = generate_blocked27_3x3(nx, ny, nz);

    double *x      = (double *)malloc((size_t)3 * N * sizeof(double));
    double *y_ref  = (double *)malloc((size_t)3 * N * sizeof(double));
    double *y_test = (double *)malloc((size_t)3 * N * sizeof(double));

    for (int i = 0; i < 3 * N; i++) {
        x[i] = (double)i;
    }

    MatvecVariant variants[] = {
        {"Escalar",   bc_matvec},
        {"AVX256",    bc_matvec_avx256},
        {"AVX512",    bc_matvec_avx512},
        {"OpenMP_v1", bc_matvec_omp_v1},
        {"OpenMP_v2", bc_matvec_omp_v2},
        {"OpenMP_v3", bc_matvec_omp_v3}
    };

    bc_matvec(A, x, y_ref); // obter y_ref para comparação

    double *sample = (double *)malloc(K * sizeof(double));  // tempos individuais
    double  means[num_variants], medians[num_variants], errors[num_variants];

    int B = 10;

    for (int v = 0; v < num_variants; v++) {
        double sum = 0.0;

        for (int i = 0; i < 5; i++) {
            variants[v].func(A, x, y_test); // "aquecimento" para cache
        }
 
        for (int k = 0; k < K; k++) {
            double t0 = wtime();
            variants[v].func(A, x, y_test);
            sample[k] = wtime() - t0;
            sum += sample[k];
        }

        means[v]   = sum / K;
        medians[v] = median(sample, K);

        // std_deviation = ...;

        double max_err = 0.0;
        for (int i = 0; i < 3 * N; i++) {
            double diff = fabs(y_ref[i] - y_test[i]) / fabs(y_ref[i]);
            if (diff > max_err) max_err = diff;
        }
        errors[v] = max_err;
    }

    double mean_ref   = means[0];
    double median_ref = medians[0];

    printf("%-12s %12s %9s %12s %9s %12s\n",
           "Variante", "Media(s)", "Speedup (Mean)", "Mediana(s)", "Speedup (Median)", "Erro Max");
    printf("--------------------------------------------------------------------------\n");
    for (int v = 0; v < num_variants; v++) {
        double speedup_mean   = mean_ref   / means[v];
        double speedup_median = median_ref / medians[v];

        gs_mean[v]   += 1.0 / speedup_mean;
        gs_median[v] += 1.0 / speedup_median;

        printf("%-12s %12.6f %8.2fx %12.6f %8.2fx     %12.2e\n",
               variants[v].name,
               means[v],   speedup_mean,
               medians[v], speedup_median,
               errors[v]);

        fprintf(csv, "%d,%d,%d,%d,%s,%.6f,%.4f,%.6f,%.4f,%.2e\n",
            nx, ny, nz, N,
            variants[v].name,
            means[v],          // tempo absoluto em segundos
            speedup_mean,
            medians[v],
            speedup_median,
            errors[v]);
    }

    (*gs_count)++;
    printf("\n");

    free(sample);
    free(x);
    free(y_ref);
    free(y_test);
    bc_free(A);
}

static void ensure_dir(const char *path) {
    if (access(path, F_OK) == -1) {
        mkdir(path, 0755);
    }
}

static void ensure_experiment_dirs(const char *compiler, char *compiler_dir) {
    ensure_dir("experiments");
    ensure_dir("experiments/spmv");
    snprintf(compiler_dir, PATH_MAX, "experiments/spmv/%s", compiler);
    ensure_dir(compiler_dir);
}

static void build_path(char *out, const char *dir, const char *file) {
    snprintf(out, PATH_MAX, "%s/%s", dir, file);
}

int run_spmv_benchmarks(int argc, char **argv) {
    if (argc != 6) {
        printf("%s <inicio> <fim> <incremento> <iterações>\n", argv[0]);
        return 1;
    }

    double gs_mean[num_variants]   = {0};
    double gs_median[num_variants] = {0};
    int gs_count = 0;

    int ini = atoi(argv[1]);
    int fim = atoi(argv[2]);
    int inc = atoi(argv[3]);
    int K   = atoi(argv[4]);
    char *compiler = argv[5];

    if (ini <= 0 || fim < ini || inc <= 0 || K <= 0) {
        printf("Parâmetros inválidos.\n");
        return 1;
    }

    const char *names[] = {
        "Escalar",
        "AVX256",
        "AVX512",
        "OpenMP_v1",
        "OpenMP_v2",
        "OpenMP_v3"
    };

    char compiler_dir[PATH_MAX];
    ensure_experiment_dirs(compiler, compiler_dir);

    char spmv_runs[PATH_MAX];
    build_path(spmv_runs, compiler_dir, "spmv_runs.csv");

    FILE *csv = fopen(spmv_runs, "w");
    fprintf(csv, "nx,ny,nz,N,variante,media_s,speedup_mean,mediana_s,speedup_median,erro_max\n");

    for (int nx = ini; nx <= fim; nx += inc) {
        evaluate_bc_matvecs(nx, nx, nx, K, csv, gs_mean, gs_median, &gs_count);
    }

    fclose(csv);

    printf("=============================================================\n");
    printf("Speedup Geral (média harmônica sobre %d malhas)\n", gs_count);
    printf("=============================================================\n");
    printf("%-12s %16s %18s\n", "Variante", "Speedup (Mean)", "Speedup (Median)");
    printf("--------------------------------------------------\n");

    char spmv_general[PATH_MAX];
    build_path(spmv_general, compiler_dir, "spmv_general.csv");

    FILE *speedup_csv = fopen(spmv_general, "w");
    fprintf(speedup_csv, "variante,speedup_geral_mean,speedup_geral_median\n");

    for (int v = 0; v < num_variants; v++) {
        double speedup_mean   = gs_count / gs_mean[v];
        double speedup_median = gs_count / gs_median[v];

        printf("%-12s %15.2fx %17.2fx\n",
               names[v],
               speedup_mean,
               speedup_median);

        fprintf(speedup_csv, "%s,%.4f,%.4f\n",
                names[v],
                speedup_mean,
                speedup_median);
    }

    printf("\n");
    fclose(speedup_csv);

    return 0;
}