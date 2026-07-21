#pragma once

#include <utils.h>
#include <bcsr.h>
#include <spmv.h>

struct MatvecVariant {
    const std::string name;
    spmv_func_t func;
};

class SpmvBenchmark {
    private:
        FILE *output_file;
        int ini, fim, inc, K;
        std::string compiler;
        std::vector<double> gs_mean;
        std::vector<double> gs_median;
        int gs_count = 0;
        const std::vector<MatvecVariant> variants = {
            {"Base",        spmv},
            {"OpenMP",      spmv_omp},
            {"AVX256",      spmv_avx256},
            {"AVX512",      spmv_avx512},
            {"Highway256",  spmv_hwy256},
            {"Highway512",  spmv_hwy512}
        };

        void evaluate_spmvs(int nx, int ny, int nz, FILE *runs_csv);

    public:
        SpmvBenchmark(int ini, int fim, int inc, int K, const std::string &compiler);
        int run();
};
