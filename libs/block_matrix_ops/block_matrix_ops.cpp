#include <block_matrix_ops.h>

void invert_3x3_matrix(double *dst_matrix, const double *M) {
    double C00 = M[idx(1,1)] * M[idx(2,2)] - M[idx(1,2)] * M[idx(2,1)];
    double C01 = M[idx(1,2)] * M[idx(2,0)] - M[idx(1,0)] * M[idx(2,2)];
    double C02 = M[idx(1,0)] * M[idx(2,1)] - M[idx(1,1)] * M[idx(2,0)];

    double det = M[idx(0,0)] * C00 + M[idx(0,1)] * C01 + M[idx(0,2)] * C02;

    dst_matrix[idx(0,0)] = C00;
    dst_matrix[idx(1,0)] = C01;
    dst_matrix[idx(2,0)] = C02;
    dst_matrix[idx(0,1)] = M[idx(0,2)] * M[idx(2,1)] - M[idx(0,1)] * M[idx(2,2)];
    dst_matrix[idx(1,1)] = M[idx(0,0)] * M[idx(2,2)] - M[idx(0,2)] * M[idx(2,0)];
    dst_matrix[idx(2,1)] = M[idx(0,1)] * M[idx(2,0)] - M[idx(0,0)] * M[idx(2,1)];
    dst_matrix[idx(0,2)] = M[idx(0,1)] * M[idx(1,2)] - M[idx(0,2)] * M[idx(1,1)];
    dst_matrix[idx(1,2)] = M[idx(0,2)] * M[idx(1,0)] - M[idx(0,0)] * M[idx(1,2)];
    dst_matrix[idx(2,2)] = M[idx(0,0)] * M[idx(1,1)] - M[idx(0,1)] * M[idx(1,0)];

    for (int i = 0; i < BS * BS; i++) {
        dst_matrix[i] /= det;
    }
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

}

void matmat_avx256(double *dst_matrix, const double *A, const double *B) {

}

void matmat_avx512(double *dst_matrix, const double *A, const double *B) {

}

void matmat_hwy256(double *dst_matrix, const double *A, const double *B) {

}

void matmat_hwy512(double *dst_matrix, const double *A, const double *B) {

}


void matsub(double *dst_matrix, const double *A, const double *B) {
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            dst_matrix[idx(i,j)] = A[idx(i,j)] - B[idx(i,j)];
        }
    }
}

void matsub_omp(double *dst_matrix, const double *A, const double *B) {

}

void matsub_avx256(double *dst_matrix, const double *A, const double *B) {

}

void matsub_avx512(double *dst_matrix, const double *A, const double *B) {

}

void matsub_hwy256(double *dst_matrix, const double *A, const double *B) {

}

void matsub_hwy512(double *dst_matrix, const double *A, const double *B) {

}

