#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include "bcsr.h"
#include "bc_matvec.h"

#define num_variants 6

typedef void (*matvec_func)(const BlockedCSR * __restrict__, const double * __restrict__, double * __restrict__);

struct MatvecVariant {
    const char *name;
    matvec_func func;
};

typedef struct MatvecVariant MatvecVariant;

static double wtime(); 
static int cmp_double(const void *a, const void *b);
static double median(double *arr, int n);
void evaluate_bc_matvecs(int nx, int ny, int nz, int K, FILE *csv);
int run_spmv_benchmarks(int argc, char **argv);
