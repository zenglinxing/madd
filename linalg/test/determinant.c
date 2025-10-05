/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>
#include"madd.h"

bool flag_success = true;
RNG_Param rng;
double tolerance = 1e-6;
uint64_t n_random_test = 4, seed = 10;

bool compare_matrix(uint64_t n, double *m1, double *m2)
{
    uint64_t i, j;
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            double a = m1[i*n+j];
            double b = m2[i*n+j];
            double dev = a - b;
            if (a != 0 && fabs(dev / a) > tolerance){
                printf("matrix differed at index (%d, %d) -> %f & %f\n", i, j, a, b);
                return false;
            }
        }
    }
    return true;
}

bool compare_cnum_matrix(uint64_t n, Cnum *m1, Cnum *m2)
{
    uint64_t i, j;
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            Cnum a = m1[i*n+j];
            Cnum b = m2[i*n+j];
            Cnum dev = Cnum_Sub(a, b);
            if (a.real !=0 && a.imag != 0 && fabs(dev.real / a.real) > tolerance && fabs(dev.imag / a.imag) > tolerance){
                printf("matrix differed at index (%d, %d) -> %f + %f*I & %f + %f*I\n", i, j, a.real, a.imag, b.real, b.imag);
                return false;
            }
        }
    }
    return true;
}

void _3x3_det0_matrix_test()
{
    printf("=== 3x3 Matrix, Det = 0 Test ===\n");
    double mat1[9], mat2[9], res1, res2;
    uint64_t i;
    printf("Matrix:\n");
    for (i=0; i<9; i++){
        mat1[i] = mat2[i] = i + 1;
        printf("%f\t", mat1[i]);
    }
    printf("\n");
    bool flag_blas = Determinant(3, mat1, &res1);
    bool flag_Bareiss = Determinant_Bareiss(3, mat2, &res2);
    if (!flag_blas){
        flag_success = false;
        printf("*** Determinant doesn't work ***\n");
    }
    if (!flag_blas){
        flag_success = false;
        printf("*** Determinant_Bareiss doesn't work ***\n");
    }

    if (fabs(res1) > tolerance){
        flag_success = false;
        printf("*** Determinant is not 0 -> %f ***\n", res1);
    }
    if (fabs(res2) > tolerance){
        flag_success = false;
        printf("*** Determinant Bareiss is not 0 -> %f ***\n", res2);
    }
    if (flag_blas && flag_Bareiss && fabs(res1) <= tolerance && fabs(res2) <= tolerance){
        printf("Test Passed\n");
    }
    printf("\n\n");
}

void random_matrix_test()
{
    printf("=== Random Matrix Test ===\n");
    double *mat1 = (double*)malloc(n_random_test*n_random_test*sizeof(double));
    double *mat2 = (double*)malloc(n_random_test*n_random_test*sizeof(double));
    uint64_t i, j, n=n_random_test;

    printf("matrix\n");
    for (i=0; i<n_random_test; i++){
        for (j=0; j<n_random_test; j++){
            mat1[i*n+j] = mat2[i*n+j] = Rand(&rng);
            printf("%f\t", mat1[i*n+j]);
        }
        printf("\n");
    }

    double res1, res2;
    bool flag_blas = Determinant(n_random_test, mat1, &res1);
    bool flag_Bareiss = Determinant_Bareiss(n_random_test, mat2, &res2);

    free(mat1);
    free(mat2);

    printf("res from Determinant:\t%f\n", res1);
    printf("res from Determinant_Bareiss:\t%f\n", res2);

    if (fabs((res1 - res2) / res1) > tolerance){
        flag_success = false;
        printf("*** Determinants are different ***\n");
    }else{
        printf("Test Passed\n");
    }
    printf("\n\n");
}

void random_complex_matrix_test()
{
    printf("=== Random Complex Matrix Test ===\n");
    Cnum *mat1 = (Cnum*)malloc(n_random_test*n_random_test*sizeof(Cnum));
    Cnum *mat2 = (Cnum*)malloc(n_random_test*n_random_test*sizeof(Cnum));
    uint64_t i, j, n=n_random_test;

    printf("matrix\n");
    for (i=0; i<n_random_test; i++){
        for (j=0; j<n_random_test; j++){
            mat1[i*n+j].real = mat2[i*n+j].real = Rand(&rng);
            mat1[i*n+j].imag = mat2[i*n+j].imag = Rand(&rng);
            printf("%f + %f*I\t", mat1[i*n+j].real, mat1[i*n+j].imag);
        }
        printf("\n");
    }

    Cnum res1, res2;
    bool flag_blas = Determinant_c64(n_random_test, mat1, &res1);
    bool flag_Bareiss = Determinant_Bareiss_c64(n_random_test, mat2, &res2);

    free(mat1);
    free(mat2);

    printf("res from Determinant:\t%f + %f*I\n", res1.real, res1.imag);
    printf("res from Determinant_Bareiss:\t%f*I\n", res2.real, res2.imag);

    if (fabs((res1.real - res2.real) / res1.real) > tolerance || fabs((res1.imag - res2.imag) / res1.imag) > tolerance){
        flag_success = false;
        printf("*** Determinants are different ***\n");
    }else{
        printf("Test Passed\n");
    }
    printf("\n\n");
}

int main(int argc, char *argv[])
{
    if (argc >= 2){
        n_random_test = atoi(argv[1]);
    }
    if (argc >= 3){
        seed = atoi(argv[2]);
    }
    printf("n for random matrix test: %llu\n", n_random_test);
    printf("random seed: %llu\n", seed);

    rng = RNG_Init(seed, 0);

    _3x3_det0_matrix_test();
    random_matrix_test();
    random_complex_matrix_test();

    if (!flag_success){
        printf("*** Some Test(s) Failed ***\n");
        exit(EXIT_FAILURE);
    }else{
        printf("All Tests Passed\n");
    }
    return 0;
}