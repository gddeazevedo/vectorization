#include <spmv_benchmarks.h>


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

static void ensure_dir(const char *path) {
    if (access(path, F_OK) == -1) {
        mkdir(path, 0755);
    }
}

static void ensure_experiment_dirs(const std::string &compiler, std::string &compiler_dir) {
    ensure_dir("experiments");
    ensure_dir("experiments/spmv");
    compiler_dir = "experiments/spmv/" + compiler;
    ensure_dir(compiler_dir.c_str());
}

static std::string build_path(const std::string &dir, const std::string &file) {
    return dir + "/" + file;
}

static void print_separator(char ch, int width) {
    for (int i = 0; i < width; i++) putchar(ch);
    putchar('\n');
}

SpmvBenchmark::SpmvBenchmark(int ini, int fim, int inc, int K, const std::string &compiler) {
    this->ini = ini;
    this->fim = fim;
    this->inc = inc;
    this->K   = K;
    this->compiler = compiler;

    this->gs_mean   = std::vector<double>(variants.size(), 0.0);
    this->gs_median = std::vector<double>(variants.size(), 0.0);
}

void SpmvBenchmark::evaluate_bc_matvecs(int nx, int ny, int nz, FILE *runs_csv) {
    int N = nx * ny * nz;

    constexpr int TABLE_WIDTH = 92;

    printf("\n");
    print_separator('=', TABLE_WIDTH);
    printf("  Malha: %d x %d x %d   |   N = %d   |   3N = %d   |   K = %d iterações\n",
           nx, ny, nz, N, 3*N, K);
    print_separator('=', TABLE_WIDTH);

    BlockedCSR *A = generate_blocked27_3x3(nx, ny, nz);

    double *x      = (double *)malloc((size_t)3 * N * sizeof(double));
    double *y_ref  = (double *)malloc((size_t)3 * N * sizeof(double));
    double *y_test = (double *)malloc((size_t)3 * N * sizeof(double));

    for (int i = 0; i < 3 * N; i++) {
        x[i] = (double)i;
    }

    bc_matvec(A, x, y_ref);

    double *sample = (double *)malloc(K * sizeof(double));

    std::vector<double> means(variants.size());
    std::vector<double> medians(variants.size());
    std::vector<double> errors(variants.size());

    for (int v = 0; v < (int)variants.size(); v++) {
        double sum = 0.0;

        for (int i = 0; i < 5; i++) {
            variants[v].func(A, x, y_test); // "aquecimento" para minimizar impacto de variações na primeira execução
        }

        for (int k = 0; k < K; k++) {
            double t0 = wtime();
            variants[v].func(A, x, y_test);
            sample[k] = wtime() - t0;
            sum += sample[k];
        }

        means[v]   = sum / K;
        medians[v] = median(sample, K);

        double max_err = 0.0;
        for (int i = 0; i < 3 * N; i++) {
            double diff = fabs(y_ref[i] - y_test[i]) / fabs(y_ref[i]);
            if (diff > max_err) max_err = diff;
        }
        errors[v] = max_err;
    }

    double mean_ref   = means[0];
    double median_ref = medians[0];

    printf("  %-18s %14s %12s %14s %12s %12s\n",
           "Variante", "Média (s)", "Speedup", "Mediana (s)", "Speedup", "Erro Máx");
    print_separator('-', TABLE_WIDTH);

    for (int v = 0; v < (int)variants.size(); v++) {
        double speedup_mean   = mean_ref   / means[v];
        double speedup_median = median_ref / medians[v];

        gs_mean[v]   += 1.0 / speedup_mean;
        gs_median[v] += 1.0 / speedup_median;

        printf("  %-18s %14.6f %11.2fx %14.6f %11.2fx %12.2e\n",
               variants[v].name.c_str(),
               means[v],   speedup_mean,
               medians[v], speedup_median,
               errors[v]);

        fprintf(runs_csv, "%d,%d,%d,%d,%s,%.6f,%.4f,%.6f,%.4f,%.2e\n",
            N, nx, ny, nz,
            variants[v].name.c_str(),
            means[v],
            speedup_mean,
            medians[v],
            speedup_median,
            errors[v]);
    }

    print_separator('=', TABLE_WIDTH);

    gs_count++;

    free(sample);
    free(x);
    free(y_ref);
    free(y_test);
    bc_free(A);
}


int SpmvBenchmark::run() {
    if (ini <= 0 || fim < ini || inc <= 0 || K <= 0) {
        printf("Parâmetros inválidos.\n");
        return 1;
    }

    constexpr int SUMMARY_WIDTH = 64;

    std::string compiler_dir;

    ensure_experiment_dirs(compiler, compiler_dir);

    std::string spmv_runs = build_path(compiler_dir, "spmv_runs.csv");
    FILE *runs_csv = fopen(spmv_runs.c_str(), "w");
    fprintf(runs_csv, "N,nx,ny,nz,variante,media_s,speedup_mean,mediana_s,speedup_median,erro_max\n");

    printf("\n");
    print_separator('#', SUMMARY_WIDTH);
    printf("#%*s%*s#\n",
           (SUMMARY_WIDTH - 2) / 2 + 13, "SpMV Benchmark",
           (SUMMARY_WIDTH - 2) / 2 - 13, "");
    print_separator('#', SUMMARY_WIDTH);
    printf("  Compilador : %s\n", compiler.c_str());
    printf("  Malhas     : %d → %d (passo %d)\n", ini, fim, inc);
    printf("  Iterações  : %d por variante\n", K);
    printf("  Variantes  : %zu\n", variants.size());
    print_separator('#', SUMMARY_WIDTH);

    for (int nx = ini; nx <= fim; nx += inc) {
        evaluate_bc_matvecs(nx, nx, nx, runs_csv);
    }

    fclose(runs_csv);

    printf("\n");
    print_separator('=', SUMMARY_WIDTH);
    printf("  Speedup Geral (média harmônica sobre %d malhas)\n", gs_count);
    print_separator('=', SUMMARY_WIDTH);
    printf("  %-18s %18s %18s\n", "Variante", "Speedup (Mean)", "Speedup (Median)");
    print_separator('-', SUMMARY_WIDTH);

    ensure_experiment_dirs(compiler, compiler_dir);
    std::string spmv_general = build_path(compiler_dir, "spmv_general.csv");

    FILE *speedup_csv = fopen(spmv_general.c_str(), "w");
    fprintf(speedup_csv, "variante,speedup_geral_mean,speedup_geral_median\n");

    for (int v = 0; v < (int)variants.size(); v++) {
        double speedup_mean   = gs_count / gs_mean[v];
        double speedup_median = gs_count / gs_median[v];

        printf("  %-18s %17.2fx %17.2fx\n",
               variants[v].name.c_str(),
               speedup_mean,
               speedup_median);

        fprintf(speedup_csv, "%s,%.4f,%.4f\n",
                variants[v].name.c_str(),
                speedup_mean,
                speedup_median);
    }

    print_separator('=', SUMMARY_WIDTH);
    printf("\n");

    fclose(speedup_csv);

    return 0;
}