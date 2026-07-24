#include <block_matrix_ops.h>

static void invert_common(double *dst_matrix, double &det, const double *M) {
    double C00 = M[idx(1,1)] * M[idx(2,2)] - M[idx(1,2)] * M[idx(2,1)];
    double C01 = M[idx(1,2)] * M[idx(2,0)] - M[idx(1,0)] * M[idx(2,2)];
    double C02 = M[idx(1,0)] * M[idx(2,1)] - M[idx(1,1)] * M[idx(2,0)];

    det = M[idx(0,0)] * C00 + M[idx(0,1)] * C01 + M[idx(0,2)] * C02;

    dst_matrix[idx(0,0)] = C00;
    dst_matrix[idx(1,0)] = C01;
    dst_matrix[idx(2,0)] = C02;
    dst_matrix[idx(0,1)] = M[idx(0,2)] * M[idx(2,1)] - M[idx(0,1)] * M[idx(2,2)];
    dst_matrix[idx(1,1)] = M[idx(0,0)] * M[idx(2,2)] - M[idx(0,2)] * M[idx(2,0)];
    dst_matrix[idx(2,1)] = M[idx(0,1)] * M[idx(2,0)] - M[idx(0,0)] * M[idx(2,1)];
    dst_matrix[idx(0,2)] = M[idx(0,1)] * M[idx(1,2)] - M[idx(0,2)] * M[idx(1,1)];
    dst_matrix[idx(1,2)] = M[idx(0,2)] * M[idx(1,0)] - M[idx(0,0)] * M[idx(1,2)];
    dst_matrix[idx(2,2)] = M[idx(0,0)] * M[idx(1,1)] - M[idx(0,1)] * M[idx(1,0)];
}

void invert_3x3_matrix(double *dst_matrix, const double *M) {
    double det;
    invert_common(dst_matrix, det, M);

    for (int i = 0; i < BS * BS; i++) {
        dst_matrix[i] /= det;
    }
}

void invert_3x3_matrix_omp(double *dst_matrix, const double *M) {
    double det;
    invert_common(dst_matrix, det, M);

    #pragma omp simd
    for (int i = 0; i < BS * BS; i++) {
        dst_matrix[i] /= det;
    }
}

void invert_3x3_matrix_avx256(double *dst_matrix, const double *M) {
    double det;
    invert_common(dst_matrix, det, M);

    __m256d vdet = _mm256_set1_pd(det);
    __m256d vdst = _mm256_loadu_pd(dst_matrix);
    vdst = _mm256_div_pd(vdst, vdet);
    _mm256_storeu_pd(dst_matrix, vdst);

    vdst = _mm256_loadu_pd(&dst_matrix[4]);
    vdst = _mm256_div_pd(vdst, vdet);
    _mm256_storeu_pd(&dst_matrix[4], vdst);

    dst_matrix[8] = dst_matrix[8] / det;
}

void invert_3x3_matrix_avx512(double *dst_matrix, const double *M) {
    double det;
    invert_common(dst_matrix, det, M);

    __m512d vdet = _mm512_set1_pd(det);
    __m512d vdst = _mm512_loadu_pd(dst_matrix);
    vdst = _mm512_div_pd(vdst, vdet);
    _mm512_storeu_pd(dst_matrix, vdst);

    dst_matrix[8] = dst_matrix[8] / det;
}

void invert_3x3_matrix_hwy256(double *dst_matrix, const double *M) {
    double det;
    invert_common(dst_matrix, det, M);

    const hn::FixedTag<double, 4> d;

    auto vdet = hn::Set(d, det);
    auto vdst = hn::LoadU(d, dst_matrix);
    vdst = hn::Div(vdst, vdet);
    hn::StoreU(vdst, d, dst_matrix);

    vdst = hn::LoadU(d, dst_matrix + 4);
    vdst = hn::Div(vdst, vdet);
    hn::StoreU(vdst, d, dst_matrix + 4);

    dst_matrix[8] = dst_matrix[8] / det;
}

void invert_3x3_matrix_hwy512(double *dst_matrix, const double *M) {
    double det;
    invert_common(dst_matrix, det, M);

    const hn::FixedTag<double, 8> d;

    auto vdet = hn::Set(d, det);
    auto vdst = hn::LoadU(d, dst_matrix);
    vdst = hn::Div(vdst, vdet);
    hn::StoreU(vdst, d, dst_matrix);

    dst_matrix[8] = dst_matrix[8] / det;
}

void matmat(double *dst_matrix, const double *A, const double *B) {
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            for (int k = 0; k < BS; k++) {
                dst_matrix[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
            }
        }
    }
}

void matmat_omp(double *dst_matrix, const double *A, const double *B) {
    for (int i = 0; i < BS; i++) {
        for (int k = 0; k < BS; k++) {
            #pragma omp simd simdlen(3)
            for (int j = 0; j < BS; j++) {
                dst_matrix[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
            }
        }
    }
}

void matmat_avx256(double *dst_matrix, const double *A, const double *B) {
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            for (int k = 0; k < BS; k++) {
                dst_matrix[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
            }
        }
    }
}

void matmat_avx512(double *dst_matrix, const double *A, const double *B) {
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            for (int k = 0; k < BS; k++) {
                dst_matrix[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
            }
        }
    }
}

void matmat_hwy256(double *dst_matrix, const double *A, const double *B) {
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            for (int k = 0; k < BS; k++) {
                dst_matrix[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
            }
        }
    }
}

void matmat_hwy512(double *dst_matrix, const double *A, const double *B) {
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            for (int k = 0; k < BS; k++) {
                dst_matrix[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
            }
        }
    }
}

void matsub(double *dst_matrix, const double *A, const double *B) {
    for (int i = 0; i < BS * BS; i++) {
        dst_matrix[i] = A[i] - B[i];
    }
}

void matsub_omp(double *dst_matrix, const double *A, const double *B) {
    #pragma omp simd
    for (int i = 0; i < BS * BS; i++) {
        dst_matrix[i] = A[i] - B[i];
    }
}

void matsub_avx256(double *dst_matrix, const double *A, const double *B) {
    __m256d va   = _mm256_loadu_pd(&A[0]);
    __m256d vb   = _mm256_loadu_pd(&B[0]);
    __m256d vdst = _mm256_sub_pd(va, vb);
    _mm256_storeu_pd(&dst_matrix[0], vdst);    
    
    va   = _mm256_loadu_pd(&A[4]);
    vb   = _mm256_loadu_pd(&B[4]);
    vdst = _mm256_sub_pd(va, vb);
    _mm256_storeu_pd(&dst_matrix[4], vdst);    

    dst_matrix[8] = A[8] - B[8];
}

void matsub_avx512(double *dst_matrix, const double *A, const double *B) {   
    __m512d va   = _mm512_loadu_pd(A);
    __m512d vb   = _mm512_loadu_pd(B);
    __m512d vdst = _mm512_sub_pd(va, vb);
    _mm512_storeu_pd(dst_matrix, vdst);
    dst_matrix[8] = A[8] - B[8];    
}

void matsub_hwy256(double *dst_matrix, const double *A, const double *B) {
    const hn::FixedTag<double, 4> d;

    auto va   = hn::LoadU(d, A);
    auto vb   = hn::LoadU(d, B);
    auto vdst = hn::Sub(va, vb);
    hn::StoreU(vdst, d, dst_matrix);

    va = hn::LoadU(d, A + 4);
    vb = hn::LoadU(d, B + 4);
    vdst = hn::Sub(va, vb);
    hn::StoreU(vdst, d, dst_matrix + 4);

    dst_matrix[8] = A[8] - B[8];
}

void matsub_hwy512(double *dst_matrix, const double *A, const double *B) {
    const hn::FixedTag<double, 8> d;

    auto va   = hn::LoadU(d, A);
    auto vb   = hn::LoadU(d, B);
    auto vdst = hn::Sub(va, vb);
    hn::StoreU(vdst, d, dst_matrix);

    dst_matrix[8] = A[8] - B[8];
}
