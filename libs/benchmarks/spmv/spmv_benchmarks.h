#pragma once

#include <utils.h>
#include <bcsr.h>
#include <bc_matvec.h>


using matvec_func_t = void (*)(const BlockedCSR * __restrict__, const double * __restrict__, double * __restrict__);

struct MatvecVariant {
    const std::string name;
    matvec_func_t func;
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
            {"Base",       bc_matvec},
            {"AVX256",     bc_matvec_avx256},
            {"AVX512",     bc_matvec_avx512_masked_reduce},
            {"AVX512_v2",  bc_matvec_avx512_scalar_reduce},
            {"OpenMP_v1",  bc_matvec_omp_v1},
            {"OpenMP_v2",  bc_matvec_omp_v2},
            {"OpenMP_v3",  bc_matvec_omp_v3},
            {"Highway",    bc_matvec_hwy},
            {"Highway_v2", bc_matvec_hwy_v2},
        };

        void evaluate_bc_matvecs(int nx, int ny, int nz, FILE *runs_csv);

    public:
        SpmvBenchmark(int ini, int fim, int inc, int K, const std::string &compiler);
        int run();
};
