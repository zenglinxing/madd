/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include"madd.h"

double tolerance = 1e-7;

void print_matrix(int m, int n, double *arr)
{
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++){
            printf("%f\t", arr[i*n+j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    uint64_t seed = 10;
    int n = 3, n_vec = 2, i, j;
    if (argc >= 2){
        n = atoi(argv[1]);
    }
    if (argc >= 3){
        n_vec = atoi(argv[2]);
    }
    if (argc >= 4){
        seed = strtoull(argv[3], NULL, 0);
    }
    printf("n = %d\n", n);
    printf("n_vec = %d\n", n_vec);
    printf("seed = %llu\n", seed);
    RNG_Param rng = RNG_Init(seed, 0);

    double *A = (double*)malloc((uint64_t)n * n * sizeof(double));
    double *AA = (double*)malloc((uint64_t)n * n * sizeof(double));
    double *B = (double*)malloc((uint64_t)n * n_vec * sizeof(double));
    double *BB = (double*)malloc((uint64_t)n * n_vec *sizeof(double));
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            A[i*n+j] = Rand(&rng);
        }
    }
    for (i=0; i<n; i++){
        for (j=0; j<n_vec; j++){
            B[i*n_vec+j] = Rand(&rng);
        }
    }
    memcpy(AA, A, (uint64_t)n*n*sizeof(double));
    memcpy(BB, B, (uint64_t)n*n_vec*sizeof(double));

    printf("matrix A:\n");
    print_matrix(n, n, A);
    printf("matrix B:\n");
    print_matrix(n, n_vec, BB);

    Linear_Equations(n, AA, n_vec, BB);

    printf("matrix A:\n");
    print_matrix(n, n, AA);
    printf("matrix B:\n");
    print_matrix(n, n_vec, BB);

    /* check by matrix multiply */
    bool flag_same = true;
    double *bb = (double*)malloc((uint64_t)n * n_vec *sizeof(double));
    Matrix_Multiply(n, n_vec, n, A, BB, bb);
    for (i=0; i<n; i++){
        for (j=0; j<n_vec; j++){
            double dev = bb[i*n_vec+j] - B[i*n_vec+j];
            double rate = fabs(dev / B[i*n_vec+j]);
            if (rate > tolerance){
                flag_same = false;
            }
        }
    }

    free(A);
    free(B);
    free(BB);

    if (!flag_same){
        printf("*** Test Failed! ***\n");
        exit(EXIT_FAILURE);
    }else{
        exit(EXIT_SUCCESS);
    }

    return 0;
}