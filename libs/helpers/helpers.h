#pragma once

#include <stdlib.h>
#include <immintrin.h>
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;

void invert_matrix(const double *M, double *M_inv);

void matmat(double *dst_matrix, const double *A, const double *B);
void matmat_omp(double *dst_matrix, const double *A, const double *B);
void matmat_avx256(double *dst_matrix, const double *A, const double *B);
void matmat_avx512(double *dst_matrix, const double *A, const double *B);
void matmat_hwy(double *dst_matrix, const double *A, const double *B);

void matsub(double *dst_matrix, const double *A, const double *B);
void matsub_omp(double *dst_matrix, const double *A, const double *B);
void matsub_avx256(double *dst_matrix, const double *A, const double *B);
void matsub_avx512(double *dst_matrix, const double *A, const double *B);
void matsub_hwy(double *dst_matrix, const double *A, const double *B);
