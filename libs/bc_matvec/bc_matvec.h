#pragma once

#include <stdlib.h>
#include <immintrin.h>
#include "bcsr.h"

#define MASK_ZERO_SLOT_3 0x8 // 0b1000 (big endian) | 0b0001 (little endian)
#define MASK_SUM_FIRST_3 0x07 // 0b00000111
#define MASK_SUM_MID_3   0x38 // 0b00111000
#define MASK_SUM_LAST_2  0xC0 // 0b11000000

void bc_matvec(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);
void bc_matvec_omp_v1(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);
void bc_matvec_omp_v2(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);
void bc_matvec_omp_v3(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);
static double hadd_256(__m256d v);
void bc_matvec_avx256(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);
void bc_matvec_avx512(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);
void bc_matvec_avx512_v2(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);
void bc_matvec_avx512_v3(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);
void bc_matvec_avx512_v4(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);