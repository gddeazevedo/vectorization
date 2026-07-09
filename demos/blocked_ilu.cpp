#include "bcsr.h"
#include <cstdlib>
#include <cstdio>

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

double *matsub(const double *A, const double *B) {
    double *C = (double *) calloc(BS * BS, sizeof(double));

    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            C[idx(i,j)] = A[idx(i,j)] - B[idx(i,j)];
        }
    }

    return C;
}

double *get_block(const BlockedCSR *A, int row, int col) {
    int row_start = A->ia[row];
    int row_end   = A->ia[row + 1];

    for (int p = row_start; p < row_end; p++) {
        if (A->ja[p] == row) {
            return &A->vals[(size_t)p * A->bs * A->bs];
        }
    }

    return nullptr;
}


void ilu0_decomposition(BlockedCSR *A) {
    for (int i = 0; i < A->nb; i++) {
        int row_start = A->ia[i];
        int row_end   = A->ia[i + 1];

        for (int p = row_start; p < row_end; p++) {
            int k = A->ja[p];

            if (k >= i) {
                break;
            }

            double *block_ik = &A->vals[(size_t)p * A->bs * A->bs];
            double *diag_kk  = get_block(A, k, k);
            bool allocated_diag = false;

            if (diag_kk == nullptr) {
                diag_kk = (double *) calloc(A->bs * A->bs, sizeof(double));
                diag_kk[0] = 1.0;
                diag_kk[4] = 1.0;
                diag_kk[8] = 1.0;
                allocated_diag = true;
            }

            double *inv   = invert_matrix(diag_kk);
            double *prod  = matmat(block_ik, inv);
            memcpy(block_ik, prod, sizeof(double) * A->bs * A->bs);

            free(inv);
            free(prod);
            if (allocated_diag) {
                free(diag_kk);
            }

            for (int q = p + 1; q < row_end; q++) {
                int j = A->ja[q];

                double *block_kj = get_block(A, k, j);

                if (block_kj == nullptr) {
                    continue;
                }

                double *block_ij = &A->vals[(size_t)q * A->bs * A->bs];
                double *prod     = matmat(block_ik, block_kj);
                double *diff     = matsub(block_ij, prod);
                memcpy(block_ij, diff, sizeof(double) * A->bs * A->bs);
                free(prod);
                free(diff);
            }
        }
    }
}


int main() {
    BlockedCSR *A = generate_blocked27_3x3(3, 3, 3);

    bc_draw(const_cast<const BlockedCSR *>(A));

    ilu0_decomposition(A);

    printf("\n\n");

    draw_global_matrix(const_cast<const BlockedCSR *>(A));

    return 0;
}
