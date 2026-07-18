#pragma once

#include <utils.h>
#include <bcsr.h>
#include <bc_matvec.h>

struct MatvecVariant {
    const std::string name;
    bc_matvec_func_t func;
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
            {"Base",        bc_matvec},
            {"OpenMP",      bc_matvec_omp},
            {"AVX256",      bc_matvec_avx256},
            {"AVX512",      bc_matvec_avx512},
            {"Highway256",  bc_matvec_hwy_256},
            {"Highway512",  bc_matvec_hwy_512},
            {"HighwayScal", bc_matvec_hwy_scalable}
        };

        void evaluate_bc_matvecs(int nx, int ny, int nz, FILE *runs_csv);

    public:
        SpmvBenchmark(int ini, int fim, int inc, int K, const std::string &compiler);
        int run();
};
