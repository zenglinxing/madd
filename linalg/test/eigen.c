/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>
#include<string.h>
#include<math.h>
#include"madd.h"

double tolerance = 1e-5;
int n = 4;
bool flag_real_left_verified = true,
     flag_real_right_verified = true,
     flag_2_real_verified = true,
     flag_complex_left_verified = true,
     flag_complex_right_verified = true;
RNG_Param rng;

void print_real_matrix(uint64_t m, uint64_t n, double *mat)
{
    for (uint64_t i=0; i<m; i++){
        for (uint64_t j=0; j<n; j++){
            printf("%f\t", mat[i*n+j]);
        }
        printf("\n");
    }
}

void print_cnum_matrix(uint64_t m, uint64_t n, Cnum *mat)
{
    for (uint64_t i=0; i<m; i++){
        for (uint64_t j=0; j<n; j++){
            printf("%f + %f*I\t", mat[i*n+j].real, mat[i*n+j].imag);
        }
        printf("\n");
    }
}


void NxN_Real_Matrix_Test()
{
    printf("\n===\n%d X %d Real Matrix (Random) Test\n===\n", n, n);
    // allocate memory
    uint64_t nn = (uint64_t)n * n, i, j;
    size_t nn_size = nn * sizeof(double), nn_cnum_size = nn * sizeof(Cnum);
    double *matrix = (double*)malloc(nn_size);
    double *mm = (double*)malloc(nn_size);
    Cnum *matrix_cnum = (Cnum*)malloc(2*nn_size);
    Cnum *eigenvalue = (Cnum*)malloc((uint64_t)n * sizeof(Cnum));
    Cnum *vr = (Cnum*)malloc(nn * sizeof(Cnum));
    Cnum *vl = (Cnum*)malloc(nn * sizeof(Cnum));
    
    // temporary space for verification
    Cnum *temp_result = (Cnum*)malloc(n * sizeof(Cnum));
    Cnum *expected_result = (Cnum*)malloc(n * sizeof(Cnum));
    
    if (!matrix || !mm || !eigenvalue || !vr || !vl || !temp_result || !expected_result) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // generate random matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i*n+j] = matrix_cnum[i*n+j].real = Rand(&rng); // random number
            matrix_cnum[i*n+j].imag = 0;
        }
    }
    memcpy(mm, matrix, nn_size);
    
    // print original matrix
    printf("Original matrix:\n");
    print_real_matrix(n, n, matrix);

    // cal eigenvalue & eigenvector
    Eigen(n, matrix, eigenvalue, true, vl, true, vr);

    printf("Eigenvalues:\n");
    print_cnum_matrix(n, 1, eigenvalue);
    printf("Eigenvectors Left:\n");
    print_cnum_matrix(n, n, vl);
    printf("Eigenvectors Right:\n");
    print_cnum_matrix(n, n, vr);

    Cnum *eigenvalue_matrix = (Cnum*)malloc(nn_cnum_size);
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            eigenvalue_matrix[i*n+j].real = (i == j) ? eigenvalue[i].real : 0;
            eigenvalue_matrix[i*n+j].imag = (i == j) ? eigenvalue[i].imag : 0;
        }
    }
    Cnum *mm_res = (Cnum*)malloc(nn_cnum_size), *mv_res = (Cnum*)malloc(nn_cnum_size);

    // left eigen check
    printf("---\nLeft Eigen Check\n---\n");
    Cnum *vlT = (Cnum*)malloc(nn_cnum_size);
    memcpy(vlT, vl, nn_cnum_size);
    Matrix_Transpose_c64(n, n, vlT);
    Matrix_Multiply_c64(n, n, n, vlT, matrix_cnum, mm_res);
    Matrix_Multiply_c64(n, n, n, eigenvalue_matrix, vlT, mv_res);
    printf("Eigenvectors Left (T) x Original Matrix:\n");
    print_cnum_matrix(n, n, mm_res);
    printf("Eigenvalues Matrix x Eigenvectors Left (T):\n");
    print_cnum_matrix(n, n, mv_res);
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            double dev_real = mm_res[i*n+j].real - mv_res[i*n+j].real;
            double dev_imag = mm_res[i*n+j].imag - mv_res[i*n+j].imag;
            double rate_real = fabs(dev_real / mv_res[i*n+j].real), rate_imag = fabs(dev_imag / mv_res[i*n+j].imag);
            if (rate_real > tolerance || rate_imag > tolerance){
                flag_real_left_verified = false;
            }
        }
    }

    // right eigen check
    printf("---\nRight Eigen Check\n---\n");
    Matrix_Multiply_c64(n, n, n, matrix_cnum, vr, mm_res);
    Matrix_Multiply_c64(n, n, n, vr, eigenvalue_matrix, mv_res);
    printf("Original Matrix x Eigenvectors Right:\n");
    print_cnum_matrix(n, n, mm_res);
    printf("Eigenvectors Right x Eigenvalues Matrix:\n");
    print_cnum_matrix(n, n, mv_res);
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            double dev_real = mm_res[i*n+j].real - mv_res[i*n+j].real;
            double dev_imag = mm_res[i*n+j].imag - mv_res[i*n+j].imag;
            double rate_real = fabs(dev_real / mv_res[i*n+j].real), rate_imag = fabs(dev_imag / mv_res[i*n+j].imag);
            if (rate_real > tolerance || rate_imag > tolerance){
                flag_real_right_verified = false;
            }
        }
    }

    // free
    free(matrix);
    free(mm);
    free(eigenvalue);
    free(vr);
    free(vl);
    free(eigenvalue_matrix);
    free(mm_res);
    free(mv_res);
    free(vlT);

    printf("4 X 4 Real Matrix Tests:\n");
    if (!flag_real_left_verified){
        printf("*** Test Real Matrix (Left) Failed ***\n");
    }else{
        printf("Test Real Matrix (Left) Passed\n");
    }
    if (!flag_real_right_verified){
        printf("*** Test Real Matrix (Right) Failed ***\n");
    }else{
        printf("Test Real Matrix (Right) Passed\n");
    }
    printf("\n\n");
}

void _2x2_Real_Matrix_Complex_Eigen_Test()
{
    uint64_t i, j;
    // complex eigenvalue test
    printf("\n===\n2 X 2 Real Matrix (Complex eigenvalues) Test\n===\n");
    double mat2[4] = {1, -1, 1, 1};
    //double mat2[4] = {1, 0, 0, -1};
    Cnum eig2[2];
    Cnum vl2[4], vr2[4];
    print_real_matrix(2, 2, mat2);
    Eigen(2, mat2, eig2, true, vl2, true, vr2);
    printf("Eigenvalues\n");
    print_cnum_matrix(2, 1, eig2);
    printf("Eigenvectors Left\n");
    print_cnum_matrix(2, 2, vl2);
    printf("Eigenvectors Right\n");
    print_cnum_matrix(2, 2, vr2);

    double sqrt2 = sqrt(0.5);
    if (fabs(vl2[0].real + sqrt2)/sqrt2 > tolerance ||
        fabs(vl2[1].real + sqrt2)/sqrt2 > tolerance ||
        fabs(vl2[2].imag + vl2[3].imag) > tolerance ||
        fabs(fabs(vl2[2].imag) - sqrt2)/sqrt2 > tolerance){
        printf("%f\n", fabs(vl2[0].real - sqrt2)/sqrt2);
        printf("%f\n", fabs(vl2[1].real - sqrt2)/sqrt2);
        printf("%f\n", fabs(vl2[2].imag + vl2[3].imag));
        printf("%f\n", fabs(fabs(vl2[2].imag) + sqrt2)/sqrt2);
        flag_2_real_verified = false;
    }
    if (fabs(vr2[0].real - sqrt2)/sqrt2 > tolerance ||
        fabs(vr2[1].real - sqrt2)/sqrt2 > tolerance ||
        fabs(vr2[2].imag + vr2[3].imag) > tolerance ||
        fabs(fabs(vr2[2].imag) - sqrt2)/sqrt2 > tolerance){
        printf("%f\n", fabs(vr2[0].real - sqrt2)/sqrt2);
        printf("%f\n", fabs(vr2[1].real - sqrt2)/sqrt2);
        printf("%f\n", fabs(vr2[2].imag + vr2[3].imag));
        printf("%f\n", fabs(fabs(vr2[2].imag) + sqrt2)/sqrt2);
        flag_2_real_verified = false;
    }

    if (!flag_2_real_verified){
        printf("*** Test Complex Eigenvalues Failed ***\n");
    }else{
        printf("Test Complex Eigenvalues Passed\n");
    }
    printf("\n\n");
}

void NxN_Complex_Matrix_Eigen_Test()
{
    printf("\n===\n%d x %d Complex Matrix Test\n===\n", n, n);
    uint64_t i, j;
    size_t size_nn = (uint64_t)n*n*sizeof(Cnum), size_n = (uint64_t)n*sizeof(Cnum);
    Cnum *matrix = (Cnum*)malloc(size_nn);
    Cnum *mm = (Cnum*)malloc(size_nn);
    Cnum *eigenvalue = (Cnum*)malloc(size_n);
    Cnum *vl = (Cnum*)malloc(size_nn);
    Cnum *vr = (Cnum*)malloc(size_nn);
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            matrix[i*n+j].real = Rand(&rng);
            matrix[i*n+j].imag = Rand(&rng);
        }
    }
    print_cnum_matrix(n, n, matrix);
    memcpy(mm, matrix, size_nn);
    
    Eigen_c64(n, matrix, eigenvalue, true, vl, true, vr);

    printf("Eigenvalues:\n");
    print_cnum_matrix(n, 1, eigenvalue);
    printf("Eigenvector Left:\n");
    print_cnum_matrix(n, n, vl);
    printf("Eigenvector Right:\n");
    print_cnum_matrix(n, n, vr);
    
    Cnum cnum_zero = {.real = 0, .imag = 0};
    Cnum *eigenvalue_matrix = (Cnum*)malloc(size_nn);
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            eigenvalue_matrix[i*n+j] = (i == j) ? eigenvalue[i] : cnum_zero;
        }
    }
    Cnum *mm_res = (Cnum*)malloc(size_nn), *mv_res = (Cnum*)malloc(size_nn);

    // left eigen check
    printf("---\nLeft Eigen Check\n---\n");
    Cnum *vlT = (Cnum*)malloc(size_nn);
    memcpy(vlT, vl, size_nn);
    Matrix_Hermitian_Transpose_c64(n, n, vlT);
    Matrix_Multiply_c64(n, n, n, vlT, mm, mm_res);
    Matrix_Multiply_c64(n, n, n, eigenvalue_matrix, vlT, mv_res);
    printf("Eigenvectors Left (T) x Original Matrix:\n");
    print_cnum_matrix(n, n, mm_res);
    printf("Eigenvalues Matrix x Eigenvectors Left (T):\n");
    print_cnum_matrix(n, n, mv_res);
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            double dev_real = mm_res[i*n+j].real - mv_res[i*n+j].real;
            double dev_imag = mm_res[i*n+j].imag - mv_res[i*n+j].imag;
            double rate_real = fabs(dev_real / mv_res[i*n+j].real), rate_imag = fabs(dev_imag / mv_res[i*n+j].imag);
            if (rate_real > tolerance || rate_imag > tolerance){
                flag_complex_left_verified = false;
            }
        }
    }

    // right eigen check
    printf("---\nRight Eigen Check\n---\n");
    Matrix_Multiply_c64(n, n, n, mm, vr, mm_res);
    Matrix_Multiply_c64(n, n, n, vr, eigenvalue_matrix, mv_res);
    printf("Original Matrix x Eigenvectors Right:\n");
    print_cnum_matrix(n, n, mm_res);
    printf("Eigenvectors Right x Eigenvalues Matrix:\n");
    print_cnum_matrix(n, n, mv_res);
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            double dev_real = mm_res[i*n+j].real - mv_res[i*n+j].real;
            double dev_imag = mm_res[i*n+j].imag - mv_res[i*n+j].imag;
            double rate_real = fabs(dev_real / mv_res[i*n+j].real), rate_imag = fabs(dev_imag / mv_res[i*n+j].imag);
            if (rate_real > tolerance || rate_imag > tolerance){
                flag_complex_right_verified = false;
            }
        }
    }

    free(matrix);
    free(mm);
    free(eigenvalue);
    free(vl);
    free(vr);
    free(eigenvalue_matrix);

    if (!flag_complex_left_verified){
        printf("*** Test Complex Matrix (Left) Failed ***\n");
    }
    if (!flag_complex_right_verified){
        printf("*** Test Complex Matrix (Right) Failed ***\n");
    }
    if (flag_complex_left_verified && flag_complex_right_verified){
        printf("Complex Matrix Test Passed\n");
    }
}

int main(int argc, char *argv[])
{
    madd_error_keep_print = true;
    uint64_t seed = 10;
    if (argc >= 2){
        n = atoi(argv[1]);
    }
    if (argc >= 3){
        seed = strtoull(argv[2], NULL, 10);
    }
    printf("n = %d\n", n);
    printf("seed = %llu\n", seed);
    rng = RNG_Init(seed, 0);

    NxN_Real_Matrix_Test();
    _2x2_Real_Matrix_Complex_Eigen_Test();
    NxN_Complex_Matrix_Eigen_Test();

    // All tests reports
    printf("\n\n===\nAll Tests Reports:\n");
    if (!flag_real_left_verified || !flag_real_right_verified || !flag_2_real_verified || !flag_complex_left_verified || !flag_complex_right_verified){
        printf("*** Some Test(s) Failed ***\n");
        exit(EXIT_FAILURE);
    }else{
        printf("All Tests Passed \\o/\n");
    }
    
    return 0;
}