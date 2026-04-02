#pragma once

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <limits.h>
#include <unistd.h>
#include "bcsr.h"
#include "bc_matvec.h"


using matvec_func_t = void (*)(const BlockedCSR * __restrict__, const double * __restrict__, double * __restrict__);

struct MatvecVariant {
    const std::string name;
    matvec_func_t func;
};

class SpmvBenchmark {
    public:
        SpmvBenchmark(int ini, int fim, int inc, int K, const std::string &compiler);
        int run();

    private:
        FILE *output_file;
        int ini, fim, inc, K;
        std::string compiler;
        std::vector<double> gs_mean;
        std::vector<double> gs_median;
        int gs_count = 0;
        const std::vector<MatvecVariant> variants = {
            {"Escalar",   bc_matvec},
            {"AVX256",    bc_matvec_avx256},
            {"AVX512",    bc_matvec_avx512},
            {"AVX512_v2", bc_matvec_avx512_v2},
            {"OpenMP_v1", bc_matvec_omp_v1},
            {"OpenMP_v2", bc_matvec_omp_v2},
            {"OpenMP_v3", bc_matvec_omp_v3}
        };

        void evaluate_bc_matvecs(int nx, int ny, int nz, FILE *runs_csv);
};


static double wtime(); 
static int cmp_double(const void *a, const void *b);
static double median(std::vector<double> &arr);
static void ensure_experiment_dirs(const std::string &compiler, std::string &compiler_dir);
static std::string build_path(const std::string &dir, const std::string &file);
