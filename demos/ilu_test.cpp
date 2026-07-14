#include "bcsr.h"
#include <cstdlib>
#include <cstdio>
#include <math.h>

void invert_matrix(const double *M, double *M_inv) {
    double C00 = M[idx(1,1)] * M[idx(2,2)] - M[idx(1,2)] * M[idx(2,1)];
    double C01 = M[idx(1,2)] * M[idx(2,0)] - M[idx(1,0)] * M[idx(2,2)];
    double C02 = M[idx(1,0)] * M[idx(2,1)] - M[idx(1,1)] * M[idx(2,0)];

    double det = M[idx(0,0)] * C00 + M[idx(0,1)] * C01 + M[idx(0,2)] * C02;

    M_inv[idx(0,0)] = C00;
    M_inv[idx(1,0)] = C01;
    M_inv[idx(2,0)] = C02;
    M_inv[idx(0,1)] = M[idx(0,2)] * M[idx(2,1)] - M[idx(0,1)] * M[idx(2,2)];
    M_inv[idx(1,1)] = M[idx(0,0)] * M[idx(2,2)] - M[idx(0,2)] * M[idx(2,0)];
    M_inv[idx(2,1)] = M[idx(0,1)] * M[idx(2,0)] - M[idx(0,0)] * M[idx(2,1)];
    M_inv[idx(0,2)] = M[idx(0,1)] * M[idx(1,2)] - M[idx(0,2)] * M[idx(1,1)];
    M_inv[idx(1,2)] = M[idx(0,2)] * M[idx(1,0)] - M[idx(0,0)] * M[idx(1,2)];
    M_inv[idx(2,2)] = M[idx(0,0)] * M[idx(1,1)] - M[idx(0,1)] * M[idx(1,0)];

    for (int i = 0; i < BS * BS; i++) {
        M_inv[i] /= det;
    }
}

void matmat(const double *A, const double *B, double *C) {
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            for (int k = 0; k < BS; k++) {
                C[idx(i,j)] += A[idx(i,k)] * B[idx(k,j)];
            }
        }
    }
}

void matsub(const double *A, const double *B, double *C) {
    for (int i = 0; i < BS; i++) {
        for (int j = 0; j < BS; j++) {
            C[idx(i,j)] = A[idx(i,j)] - B[idx(i,j)];
        }
    }
}


void ilu0_decomposition(BlockedCSR *A) {
    double *prod = (double *) calloc(BS * BS, sizeof(double));
    double *diff = (double *) calloc(BS * BS, sizeof(double));
    double *inv  = (double *) calloc(BS * BS, sizeof(double));

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

            invert_matrix(diag_kk, inv);
            matmat(block_ik, inv, prod);
            memcpy(block_ik, prod, sizeof(double) * A->bs * A->bs);
            memset(prod, 0, sizeof(double) * BS * BS);

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
                matmat(block_ik, block_kj, prod);
                matsub(block_ij, prod, diff);
                memset(prod, 0, sizeof(double) * BS * BS);
                memcpy(block_ij, diff, sizeof(double) * A->bs * A->bs);
            }
        }
    }

    free(prod);
    free(diff);
    free(inv);
}



// Expande BlockedCSR para matriz densa n x n (n = nb * bs)
double *bcsr_to_dense(const BlockedCSR *A) {
    int n = A->nb * A->bs;
    double *dense = (double *) calloc(n * n, sizeof(double));

    for (int i = 0; i < A->nb; i++) {
        for (int p = A->ia[i]; p < A->ia[i + 1]; p++) {
            int j = A->ja[p];
            double *block = &A->vals[(size_t)p * A->bs * A->bs];

            for (int r = 0; r < A->bs; r++) {
                for (int c = 0; c < A->bs; c++) {
                    int gi = i * A->bs + r;
                    int gj = j * A->bs + c;
                    dense[gi * n + gj] = block[r * A->bs + c];
                }
            }
        }
    }

    return dense;
}

// ILU0 escalar in-place sobre matriz densa n x n
void ilu0_scalar(double *A, int n) {
    for (int i = 1; i < n; i++) {
        for (int k = 0; k < i; k++) {
            if (A[i * n + k] == 0.0) continue;

            A[i * n + k] = A[i * n + k] / A[k * n + k];

            for (int j = k + 1; j < n; j++) {
                if (A[i * n + j] == 0.0) continue;
                A[i * n + j] = A[i * n + j] - A[i * n + k] * A[k * n + j];
            }
        }
    }
}

void test_ilu0() {
    BlockedCSR *A = generate_blocked27_3x3(2, 2, 2);

    // copia densa antes do ILU0
    double *dense = bcsr_to_dense(A);
    int n = A->nb * A->bs;

    // ILU0 escalar na densa
    ilu0_scalar(dense, n);

    // ILU0 blocked na BlockedCSR
    ilu0_decomposition(A);

    // expande o resultado blocked para densa
    double *blocked_result = bcsr_to_dense(A);

    // compara
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double err = fabs(dense[i * n + j] - blocked_result[i * n + j]);
            if (err > max_err) max_err = err;
        }
    }

    printf("Erro maximo: %e\n", max_err);

    if (max_err < 1e-10) {
        printf("TESTE OK\n");
    } else {
        printf("TESTE FALHOU\n");

        printf("\nEscalar:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%8.4f ", dense[i * n + j]);
            }
            printf("\n");
        }

        printf("\nBlocked:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%8.4f ", blocked_result[i * n + j]);
            }
            printf("\n");
        }
    }

    free(dense);
    free(blocked_result);
}

int main() {
    test_ilu0();
    return 0;
}