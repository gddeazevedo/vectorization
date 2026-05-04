#include <bcsr.h>

BlockedCSR *bc_alloc(int nb, int bs, int max_nblocks) {
    BlockedCSR *A = (BlockedCSR *)malloc(sizeof(BlockedCSR));
    if (!A) { perror("malloc A"); exit(1); }
    A->nb = nb;
    A->bs = bs;
    A->nnzb = 0;
    A->ia = (int *)malloc((nb + 1) * sizeof(int));
    A->ja = (int *)malloc(max_nblocks * sizeof(int));
    A->vals = (double *)malloc((size_t)max_nblocks * bs * bs * sizeof(double));
    if (!A->ia || !A->ja || !A->vals) { perror("malloc arrays"); exit(1); }
    A->ia[0] = 0;
    return A;
}

void bc_shrink_to_fit(BlockedCSR *A) {
    A->ja = (int *)realloc(A->ja, (size_t)A->nnzb * sizeof(int));
    A->vals = (double *)realloc(A->vals, (size_t)A->nnzb * A->bs * A->bs * sizeof(double));
}

void bc_free(BlockedCSR *A) {
    if (!A) {
        return;
    }
    free(A->ia);
    free(A->ja);
    free(A->vals);
    free(A);
}

void bc_push_block(BlockedCSR *A, int brow, int bcol, const double *block) {
    int bs = A->bs;
    int pos = A->nnzb;
    A->ja[pos] = bcol;
    memcpy(&A->vals[(size_t)pos * bs * bs], block, (size_t)bs * bs * sizeof(double));
    A->nnzb++;
    A->ia[brow + 1] = A->nnzb;
}


BlockedCSR *generate_blocked27_3x3(int nx, int ny, int nz) {
    int N = nx * ny * nz;
    int bs = 3;
    int max_blocks = N * 27;
    BlockedCSR *A = bc_alloc(N, bs, max_blocks);

    int nnz_count = 0;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int id = i + j * nx + k * nx * ny;
                A->ia[id] = nnz_count;

                for (int dk = -1; dk <= 1; dk++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        for (int di = -1; di <= 1; di++) {
                            int ni = i + di;
                            int nj = j + dj;
                            int nk = k + dk;
                            if (ni < 0 || nj < 0 || nk < 0 || ni >= nx || nj >= ny || nk >= nz) continue;
                            int nid = ni + nj * nx + nk * nx * ny;
                            double blk[9];
                            if (nid == id) {
                                for (int r = 0; r < 9; r++) blk[r] = 0.0;
                                blk[0] = 2.0; blk[4] = 2.0; blk[8] = 2.5;
                                blk[1] = blk[2] = blk[3] = blk[5] = blk[6] = blk[7] = 0.1;
                            } else {
                                for (int r = 0; r < 9; r++) blk[r] = 0.05;
                            }
                            bc_push_block(A, id, nid, blk);
                            nnz_count++;
                        }
                    }
                }
                A->ia[id + 1] = nnz_count;
            }
        }
    }
    A->nnzb = nnz_count;
    bc_shrink_to_fit(A);
    return A;
}

void bc_draw(const BlockedCSR *A) {
    int nb = A->nb;

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < nb; j++) {
            unsigned char isBlock = 0;
            for(int k = A->ia[i]; k < A->ia[i + 1]; k++) {
                if(j == A->ja[k]) {
                    printf("[X]");
                    isBlock = 1;
                    break;
                }           
            }

            if(isBlock == 0) {
                printf("   ");
            } 
        }

        printf("\n");
    }
}