#include <iostream>
#include "bcsr.h"
#include "spmv_benchmarks.h"

int main(int argc, char **argv) {
    if (argc != 6) {
        printf("%s <inicio> <fim> <incremento> <iterações>\n", argv[0]);
        return 1;
    }

    int ini = atoi(argv[1]);
    int fim = atoi(argv[2]);
    int inc = atoi(argv[3]);
    int K   = atoi(argv[4]);

    std::string compiler = argv[5];

    SpmvBenchmark benchmark(ini, fim, inc, K, compiler);

    return benchmark.run();
}
