#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#define BLOCK_SIZE      3
#define EXPECTED_INPUTS 4

typedef struct {
    int    n_block_rows;        // Number of block rows
    int    block_size;          // Number of elements per block row/column
    int    n_nonzero_blocks;    // Number of non-zero blocks
    int    *row_blocks_ptrs;    // Stores the indices of the blocks that start each block row, indices of col_blocks_indices and block_values
    int    *col_blocks_indices; // Stores the nonzero blocks columns indices
    double *block_values;       // Stores the nonzero blocks elements
} BlockedCSR;


void bc_draw(BlockedCSR *A);
void bc_shrink_to_fit(BlockedCSR *A);
void bc_free(BlockedCSR *A);
void bc_push_block(BlockedCSR *A, int block_row, int block_col, const double *block);
void bc_matvec(const BlockedCSR *A, const double *input, double *output);
BlockedCSR *bc_alloc(int n_block_rows, int block_size, int max_n_blocks);
BlockedCSR *create_blocked_27pt_3x3(int nx, int ny, int nz);
unsigned char is_not_neighbour(int x, int y, int z, int nx, int ny, int nz);


int main() {
    return 0;
}

void bc_draw(BlockedCSR *A) {}

void bc_shrink_to_fit(BlockedCSR *A) {}

void bc_free(BlockedCSR *A) {}

void bc_push_block(BlockedCSR *A, int block_row, int block_col, const double *block) {}

void bc_matvec(const BlockedCSR *A, const double *input, double *output) {}

BlockedCSR *bc_alloc(int n_block_rows, int block_size, int max_n_blocks) {
    BlockedCSR * A = (BlockedCSR*) malloc(sizeof(BlockedCSR));

    if (A == NULL) {
        perror("Error allocating the Blocked CSR");
        exit(1);
    }

    A->n_block_rows     = n_block_rows;
    A->block_size       = block_size;
    A->n_nonzero_blocks = 0;
}

BlockedCSR *create_blocked_27pt_3x3(int nx, int ny, int nz) {}

unsigned char is_not_neighbour(int x, int y, int z, int nx, int ny, int nz) {}
