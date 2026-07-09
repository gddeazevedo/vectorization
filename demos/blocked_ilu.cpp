#include "bcsr.h"
#include <cstdlib>
#include <cstdio>

#define BS 3
#define idx(i, j) i * BS + j


double *invert_matrix(const double *M) {
    double *inv = (double *) malloc(sizeof(double) * BS * BS);

    double C00 = M[idx(1,1)] * M[idx(2,2)] - M[idx(1,2)] * M[idx(2,1)];
    double C01 = M[idx(1,2)] * M[idx(2,0)] - M[idx(1,0)] * M[idx(2,2)];
    double C02 = M[idx(1,0)] * M[idx(2,1)] - M[idx(1,1)] * M[idx(2,0)];

    double det = M[idx(0,0)] * C00 + M[idx(0,1)] * C01 + M[idx(0,2)] * C02;

    inv[idx(0,0)] = C00;
    inv[idx(1,0)] = C01;
    inv[idx(2,0)] = C02;
    inv[idx(0,1)] = M[idx(0,2)] * M[idx(2,1)] - M[idx(0,1)] * M[idx(2,2)];
    inv[idx(1,1)] = M[idx(0,0)] * M[idx(2,2)] - M[idx(0,2)] * M[idx(2,0)];
    inv[idx(2,1)] = M[idx(0,1)] * M[idx(2,0)] - M[idx(0,0)] * M[idx(2,1)];
    inv[idx(0,2)] = M[idx(0,1)] * M[idx(1,2)] - M[idx(0,2)] * M[idx(1,1)];
    inv[idx(1,2)] = M[idx(0,2)] * M[idx(1,0)] - M[idx(0,0)] * M[idx(1,2)];
    inv[idx(2,2)] = M[idx(0,0)] * M[idx(1,1)] - M[idx(0,1)] * M[idx(1,0)];

    for (int i = 0; i < BS * BS; i++) {
        inv[i] /= det;
    }

    return inv;
}

double *matmat(const double *A, const double *B) {
    double *C = (double *) calloc(BS * BS, sizeof(double));

    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            for (int k = 0; k < BS; k++) {
                C[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
            }
        }
    }

    return C;
}


void ilu0_decomposition(const BlockedCSR *A) {
    for (int i = 0; i < A->nb; i++) {
        int row_start = A->ia[i];
        int row_end   = A->ia[i + 1];


        for (int p = row_start; p < row_end; p++) {

        }

    }
}



int main() {
    BlockedCSR *A = generate_blocked27_3x3(3, 3, 3);

    bc_draw(const_cast<const BlockedCSR *>(A));

    return 0;
}
