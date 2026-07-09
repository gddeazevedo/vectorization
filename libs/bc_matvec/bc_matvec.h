#pragma once

#include <stdlib.h>
#include <immintrin.h>
#include <hwy/highway.h>
#include <bcsr.h>

#define MASK_ZERO_SLOT_3 0x8  // 0b1000
#define MASK_SUM_FIRST_3 0x07 // 0b00000111
#define MASK_SUM_MID_3   0x38 // 0b00111000
#define MASK_SUM_LAST_2  0xC0 // 0b11000000

namespace hn = hwy::HWY_NAMESPACE;

using bc_matvec_func_t = void (*)(const BlockedCSR * __restrict__, const double * __restrict__, double * __restrict__);

static double hadd_256(__m256d v);

void bc_matvec(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);
void bc_matvec_omp(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);
void bc_matvec_avx256(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);
void bc_matvec_avx512(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);
void bc_matvec_hwy_256(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);
void bc_matvec_hwy_512(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y);