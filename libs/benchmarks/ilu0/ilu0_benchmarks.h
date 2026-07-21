#pragma once

#include <utils.h>
#include <bcsr.h>
#include <ilu0.h>


struct Ilu0Variant {
    const std::string name;
    ilu0_func_t func;
};

class Ilu0Benchmark {
    private:
        int ini, fim, inc, K;
        std::string compiler;
        std::vector<double> gs_mean;
        std::vector<double> gs_median;
        int gs_count = 0;
        const std::vector<Ilu0Variant> variants = {
            {"Base",        ilu0_decomposition},
            {"OpenMP",      ilu0_decomposition_omp},
            {"AVX256",      ilu0_decomposition_avx256},
            {"AVX512",      ilu0_decomposition_avx512},
            {"Highway256",  ilu0_decomposition_hwy256},
            {"Highway512",  ilu0_decomposition_hwy512}
        };

        void evaluate_ilu0(int nx, int ny, int nz, FILE *runs_csv);

    public:
        Ilu0Benchmark(int ini, int fim, int inc, int K, const std::string &compiler);
        int run();
};