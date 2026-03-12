#pragma once

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

struct BlockedCSR {
   int nb;      // número de "block rows" (nós)
   int bs;      // block size vamos usar 3 
   int nnzb;    // número de blocos não nulos
   int *ia;     // tamanho nb+1, índice inicial de cada block-row em ja/vals
   int *ja;     // tamanho nnzb, coluna (block index) de cada bloco
   double *vals;// tamanho nnzb * bs * bs, blocos armazenados consecutivamente em row-major dentro do bloco
};

BlockedCSR *bc_alloc(int nb, int bs, int max_nblocks);
BlockedCSR *generate_blocked27_3x3(int nx, int ny, int nz);
void bc_shrink_to_fit(BlockedCSR *A);
void bc_free(BlockedCSR *A);
void bc_push_block(BlockedCSR *A, int brow, int bcol, const double *block);
void bc_draw(const BlockedCSR *A);
