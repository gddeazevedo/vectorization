#pragma once


#include <stdlib.h>
#include <immintrin.h>
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

void invert_matrix(const double *M, double *M_inv);

void matmat(const double *A, const double *B, double *C);
void matmat_omp(const double *A, const double *B, double *C);
void matmat_avx256(const double *A, const double *B, double *C);
void matmat_avx512(const double *A, const double *B, double *C);
void matmat_hwy(const double *A, const double *B, double *C);

void matsub(const double *A, const double *B, double *C);
void matsub_omp(const double *A, const double *B, double *C);
void matsub_avx256(const double *A, const double *B, double *C);
void matsub_avx512(const double *A, const double *B, double *C);
void matsub_hwy(const double *A, const double *B, double *C);
