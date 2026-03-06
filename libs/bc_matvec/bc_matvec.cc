#include "bc_matvec.h"

void bc_matvec(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y) {
    int bs = A->bs;
    int Nblocks = A->nb;
    int total_len = Nblocks * bs;
    for (int i = 0; i < total_len; i++) y[i] = 0.0;
    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow + 1];
        double *yrow  = &y[brow * bs];
        for (int p = row_start; p < row_end; p++) { // percorre os blocos de uma linha
            int bcol = A->ja[p];
            const double *blk = &A->vals[p * bs * bs];
            const double *xcol = &x[bcol * bs];
            yrow[0] += blk[0] * xcol[0];
            yrow[0] += blk[1] * xcol[1];
            yrow[0] += blk[2] * xcol[2];
            
            yrow[1] += blk[3] * xcol[0];
            yrow[1] += blk[4] * xcol[1];
            yrow[1] += blk[5] * xcol[2];

            yrow[2] += blk[6] * xcol[0];
            yrow[2] += blk[7] * xcol[1];
            yrow[2] += blk[8] * xcol[2];
        }
    }
}

void bc_matvec_omp_v1(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y) {
    int bs        = A->bs;
    int total_len = A->nb * bs;

    for (int i = 0; i < total_len; i++) {
        y[i] = 0.0;
    }

    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow + 1];
        double *yrow  = &y[brow * bs];

        double s0 = 0.0, s1 = 0.0, s2 = 0.0;

        #pragma omp simd reduction(+:s0,s1,s2)
        for (int p = row_start; p < row_end; p++) {
            int bcol           = A->ja[p];
            const double *blk  = &A->vals[p * bs * bs];
            const double *xcol = &x[bcol * bs];

            s0 += blk[0]*xcol[0] + blk[1]*xcol[1] + blk[2]*xcol[2];
            s1 += blk[3]*xcol[0] + blk[4]*xcol[1] + blk[5]*xcol[2];
            s2 += blk[6]*xcol[0] + blk[7]*xcol[1] + blk[8]*xcol[2];
        }

        yrow[0] += s0;
        yrow[1] += s1;
        yrow[2] += s2;
    }
}

void bc_matvec_omp_v2(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y) {
    int bs        = A->bs;
    int total_len = A->nb * bs;

    for (int i = 0; i < total_len; i++) {
        y[i] = 0.0;
    }

    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow + 1];
        double *yrow  = &y[brow * bs];

        for (int p = row_start; p < row_end; p++) {
            int bcol           = A->ja[p];
            const double *blk  = &A->vals[p * bs * bs];
            const double *xcol = &x[bcol * bs];

            for (int i = 0; i < bs; i++) {
                double acc = 0.0;

                #pragma omp simd reduction(+:acc)
                for (int j = 0; j < bs; j++) {
                    acc += blk[i*bs + j] * xcol[j];
                }

                yrow[i] += acc;
            }
        }
    }
}

void bc_matvec_omp_v3(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y) {
    int total_len = A->nb * A->bs;

    for (int i = 0; i < total_len; i++) y[i] = 0.0;

    #pragma omp parallel for schedule(static)
    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow + 1];
        double *yrow  = &y[brow * 3];

        double y0 = 0.0, y1 = 0.0, y2 = 0.0;

        for (int p = row_start; p < row_end; p++) {
            const double *blk  = &A->vals[p * 9];
            const double *xcol = &x[A->ja[p] * 3];

            y0 += blk[0]*xcol[0] + blk[1]*xcol[1] + blk[2]*xcol[2];
            y1 += blk[3]*xcol[0] + blk[4]*xcol[1] + blk[5]*xcol[2];
            y2 += blk[6]*xcol[0] + blk[7]*xcol[1] + blk[8]*xcol[2];
        }

        yrow[0] = y0;
        yrow[1] = y1;
        yrow[2] = y2;
    }
}


static double hadd_256(__m256d v) {
    __m256d zero = _mm256_setzero_pd();

    // v_mod = [v0, v1, v2, 0.0]
    __m256d v_mod = _mm256_blend_pd(v, zero, MASK_ZERO_SLOT_3); 
    __m256d sum_half = _mm256_add_pd(
        v_mod, 
        _mm256_permute2f128_pd(v_mod, v_mod, 0x01) 
    );

    __m128d sum_128 = _mm256_castpd256_pd128(sum_half);
    __m128d final_sum = _mm_hadd_pd(sum_128, sum_128); 
    return _mm_cvtsd_f64(final_sum);
}

void bc_matvec_avx256(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y) {
    int bs = A->bs;
   
    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow+1];

        double *yrow = &y[(size_t)brow * bs];
        int p = row_start;
        int bcol = A->ja[p];
        const double *blk  = &A->vals[(size_t)p * bs * bs];
        const double *xcol = &x[(size_t)bcol * bs];

        __m256d vx = _mm256_loadu_pd(xcol);

        __m256d vb0 = _mm256_loadu_pd(blk);
        __m256d y0 = _mm256_mul_pd(vb0, vx);

        __m256d vb1 = _mm256_loadu_pd(&blk[3]);
        __m256d y1 = _mm256_mul_pd(vb1, vx);

        __m256d vb2  = _mm256_loadu_pd(&blk[6]);
        __m256d y2 = _mm256_mul_pd(vb2, vx);

        for (int p = row_start + 1; p < row_end; p++) {
            bcol = A->ja[p];
            blk  = &A->vals[(size_t)p * bs * bs];
            xcol = &x[(size_t)bcol * bs];
            __m256d vx = _mm256_loadu_pd(xcol);
            
            vb0 = _mm256_loadu_pd(blk);
            y0  = _mm256_fmadd_pd(vb0, vx, y0);
            
            vb1 = _mm256_loadu_pd(&blk[3]);
            y1 = _mm256_fmadd_pd(vb1, vx, y1);
            
            vb2  = _mm256_loadu_pd(&blk[6]);
            y2 = _mm256_fmadd_pd(vb2, vx, y2);
        }
        yrow[0] = hadd_256(y0);
        yrow[1] = hadd_256(y1);
        yrow[2] = hadd_256(y2);   
    }
}

void bc_matvec_avx256_parallel(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y) {
    int bs = A->bs;
   
    #pragma omp parallel for schedule(static)
    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow+1];

        double *yrow = &y[(size_t)brow * bs];
        int p = row_start;
        int bcol = A->ja[p];
        const double *blk  = &A->vals[(size_t)p * bs * bs];
        const double *xcol = &x[(size_t)bcol * bs];

        __m256d vx = _mm256_loadu_pd(xcol);

        __m256d vb0 = _mm256_loadu_pd(blk);
        __m256d y0 = _mm256_mul_pd(vb0, vx);

        __m256d vb1 = _mm256_loadu_pd(&blk[3]);
        __m256d y1 = _mm256_mul_pd(vb1, vx);

        __m256d vb2  = _mm256_loadu_pd(&blk[6]);
        __m256d y2 = _mm256_mul_pd(vb2, vx);

        for (int p = row_start + 1; p < row_end; p++) {
            bcol = A->ja[p];
            blk  = &A->vals[(size_t)p * bs * bs];
            xcol = &x[(size_t)bcol * bs];
            __m256d vx = _mm256_loadu_pd(xcol);
            
            vb0 = _mm256_loadu_pd(blk);
            y0  = _mm256_fmadd_pd(vb0, vx, y0);
            
            vb1 = _mm256_loadu_pd(&blk[3]);
            y1 = _mm256_fmadd_pd(vb1, vx, y1);
            
            vb2  = _mm256_loadu_pd(&blk[6]);
            y2 = _mm256_fmadd_pd(vb2, vx, y2);
        }
        yrow[0] = hadd_256(y0);
        yrow[1] = hadd_256(y1);
        yrow[2] = hadd_256(y2);   
    }
}

void bc_matvec_avx512(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y) {
    int bs = A->bs; // = 3 neste caso

    // Índices de permutação para replicar os elementos de x
    // [x0, x1, x2, x0, x1, x2, x0, x1]
    __m512i perm_idx = _mm512_set_epi64(1, 0, 2, 1, 0, 2, 1, 0);

    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow+1];

        double *yrow = &y[(size_t)brow * bs];

        __m512d acc = _mm512_setzero_pd();
        double y2_extra = 0.0;

        for (int p = row_start; p < row_end; p++) {
            int bcol = A->ja[p];
            const double *blk = &A->vals[(size_t)p * bs * bs];
            const double *xcol = &x[(size_t)bcol * bs];

            // Carrega 8 elementos do bloco
            __m512d vb = _mm512_loadu_pd(blk);

            // Carrega xcol em um vetor de 256 bits
            __m256d vx_256 = _mm256_loadu_pd(xcol);
            // Replica os elementos de xcol usando permutação
            __m512d vx_512 = _mm512_broadcast_f64x4(vx_256);
            vx_512 = _mm512_permutexvar_pd(perm_idx, vx_512);

            acc = _mm512_fmadd_pd(vb, vx_512, acc);
            y2_extra += blk[8] * xcol[2];
        }

        yrow[0] = _mm512_mask_reduce_add_pd(MASK_SUM_FIRST_3, acc);
        yrow[1] = _mm512_mask_reduce_add_pd(MASK_SUM_MID_3, acc);
        yrow[2] = _mm512_mask_reduce_add_pd(MASK_SUM_LAST_2, acc) + y2_extra;
    }
}

void bc_matvec_avx512_parallel(const BlockedCSR * __restrict__ A, const double * __restrict__ x, double * __restrict__ y) {
    int bs = A->bs; // = 3 neste caso

    // Índices de permutação para replicar os elementos de x
    // [x0, x1, x2, x0, x1, x2, x0, x1]
    __m512i perm_idx = _mm512_set_epi64(1, 0, 2, 1, 0, 2, 1, 0);

    #pragma omp parallel for schedule(static)
    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow+1];

        double *yrow = &y[(size_t)brow * bs];

        __m512d acc = _mm512_setzero_pd();
        double y2_extra = 0.0;

        for (int p = row_start; p < row_end; p++) {
            int bcol = A->ja[p];
            const double *blk = &A->vals[(size_t)p * bs * bs];
            const double *xcol = &x[(size_t)bcol * bs];

            // Carrega 8 elementos do bloco
            __m512d vb = _mm512_loadu_pd(blk);

            // Carrega xcol em um vetor de 256 bits
            __m256d vx_256 = _mm256_loadu_pd(xcol);
            // Replica os elementos de xcol usando permutação
            __m512d vx_512 = _mm512_broadcast_f64x4(vx_256);
            vx_512 = _mm512_permutexvar_pd(perm_idx, vx_512);

            acc = _mm512_fmadd_pd(vb, vx_512, acc);
            y2_extra += blk[8] * xcol[2];
        }

        yrow[0] = _mm512_mask_reduce_add_pd(MASK_SUM_FIRST_3, acc);
        yrow[1] = _mm512_mask_reduce_add_pd(MASK_SUM_MID_3, acc);
        yrow[2] = _mm512_mask_reduce_add_pd(MASK_SUM_LAST_2, acc) + y2_extra;
    }
}