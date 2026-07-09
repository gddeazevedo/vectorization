#pragma once

#include <bcsr.h>

void ilu0_decomposition(BlockedCSR *A);
void ilu0_decomposition_omp(BlockedCSR *A);
void ilu0_decomposition_avx256(BlockedCSR *A);
void ilu0_decomposition_avx512(BlockedCSR *A);
void ilu0_decomposition_hwy(BlockedCSR *A);