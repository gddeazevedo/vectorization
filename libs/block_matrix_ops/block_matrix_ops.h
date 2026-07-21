#pragma once

#include <stdlib.h>
#include <immintrin.h>
#include <hwy/highway.h>

namespace hn = hwy::HWY_NAMESPACE;


#define BS 3
#define idx(i,j) ((i)*BS + (j))

void invert_3x3_matrix(double *dst_matrix, const double *M);

void matmat(double *dst_matrix, const double *A, const double *B);
void matmat_omp(double *dst_matrix, const double *A, const double *B);
void matmat_avx256(double *dst_matrix, const double *A, const double *B);
void matmat_avx512(double *dst_matrix, const double *A, const double *B);
void matmat_hwy256(double *dst_matrix, const double *A, const double *B);
void matmat_hwy512(double *dst_matrix, const double *A, const double *B);

void matsub(double *dst_matrix, const double *A, const double *B);
void matsub_omp(double *dst_matrix, const double *A, const double *B);
void matsub_avx256(double *dst_matrix, const double *A, const double *B);
void matsub_avx512(double *dst_matrix, const double *A, const double *B);
void matsub_hwy256(double *dst_matrix, const double *A, const double *B);
void matsub_hwy512(double *dst_matrix, const double *A, const double *B);
