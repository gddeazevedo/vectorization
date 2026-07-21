#include <ilu0_benchmarks.h>

Ilu0Benchmark::Ilu0Benchmark(int ini, int fim, int inc, int K, const std::string &compiler) {
    this->ini = ini;
    this->fim = fim;
    this->inc = inc;
    this->K   = K;
    this->compiler = compiler;

    this->gs_mean   = std::vector<double>(variants.size(), 0.0);
    this->gs_median = std::vector<double>(variants.size(), 0.0);
}

void Ilu0Benchmark::evaluate_ilu0(int nx, int ny, int nz, FILE *runs_csv) {
    int N = nx * ny * nz;

    constexpr int TABLE_WIDTH = 92;

    printf("\n");
    print_separator('=', TABLE_WIDTH);
    printf("  Malha: %d x %d x %d   |   N = %d   |   3N = %d   |   K = %d iterações\n",
           nx, ny, nz, N, 3*N, K);
    print_separator('=', TABLE_WIDTH);

    BlockedCSR A = BlockedCSR::generate_blocked27_3x3(nx, ny, nz);

    size_t vals_size = (size_t)A.nnzb * A.bs * A.bs * sizeof(double);

    double *orig_vals = (double *)malloc(vals_size);
    double *ref_vals  = (double *)malloc(vals_size);
    memcpy(orig_vals, A.vals, vals_size);

    // Resultado de referência (variante base)
    ilu0_decomposition(A);
    memcpy(ref_vals, A.vals, vals_size);

    double *sample = (double *)malloc(K * sizeof(double));

    std::vector<double> means(variants.size());
    std::vector<double> medians(variants.size());
    std::vector<double> errors(variants.size());

    for (int v = 0; v < (int)variants.size(); v++) {
        double sum = 0.0;

        for (int i = 0; i < 5; i++) {
            memcpy(A.vals, orig_vals, vals_size);
            variants[v].func(A);
        }

        for (int k = 0; k < K; k++) {
            memcpy(A.vals, orig_vals, vals_size);
            double t0 = wtime();
            variants[v].func(A);
            sample[k] = wtime() - t0;
            sum += sample[k];
        }

        means[v]   = sum / K;
        medians[v] = median(sample, K);

        // Erro relativo máximo contra a referência
        double max_err = 0.0;
        int total_vals = A.nnzb * A.bs * A.bs;
        for (int i = 0; i < total_vals; i++) {
            double ref = fabs(ref_vals[i]);
            if (ref > 0.0) {
                double diff = fabs(ref_vals[i] - A.vals[i]) / ref;
                if (diff > max_err) max_err = diff;
            }
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
    free(orig_vals);
    free(ref_vals);
}


int Ilu0Benchmark::run() {
    if (ini <= 0 || fim < ini || inc <= 0 || K <= 0) {
        printf("Parâmetros inválidos.\n");
        return 1;
    }

    constexpr int SUMMARY_WIDTH = 64;

    std::string compiler_dir;

    ensure_experiment_dirs(compiler, compiler_dir);

    std::string ilu0_runs = build_path(compiler_dir, "ilu0_runs.csv");
    FILE *runs_csv = fopen(ilu0_runs.c_str(), "w");
    fprintf(runs_csv, "N,nx,ny,nz,variante,media_s,speedup_mean,mediana_s,speedup_median,erro_max\n");

    printf("\n");
    print_separator('#', SUMMARY_WIDTH);
    printf("#%*s%*s#\n",
           (SUMMARY_WIDTH - 2) / 2 + 13, "ILU0 Benchmark",
           (SUMMARY_WIDTH - 2) / 2 - 13, "");
    print_separator('#', SUMMARY_WIDTH);
    printf("  Compilador : %s\n", compiler.c_str());
    printf("  Malhas     : %d → %d (passo %d)\n", ini, fim, inc);
    printf("  Iterações  : %d por variante\n", K);
    printf("  Variantes  : %zu\n", variants.size());
    print_separator('#', SUMMARY_WIDTH);

    for (int nx = ini; nx <= fim; nx += inc) {
        evaluate_ilu0(nx, nx, nx, runs_csv);
    }

    fclose(runs_csv);

    printf("\n");
    print_separator('=', SUMMARY_WIDTH);
    printf("  Speedup Geral (média harmônica sobre %d malhas)\n", gs_count);
    print_separator('=', SUMMARY_WIDTH);
    printf("  %-18s %18s %18s\n", "Variante", "Speedup (Mean)", "Speedup (Median)");
    print_separator('-', SUMMARY_WIDTH);

    ensure_experiment_dirs(compiler, compiler_dir);
    std::string ilu0_general = build_path(compiler_dir, "ilu0_general.csv");

    FILE *speedup_csv = fopen(ilu0_general.c_str(), "w");
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