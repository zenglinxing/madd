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

bool compare_matrix(uint64_t n, double *m1, double *m2)
{
    uint64_t i, j;
    bool flag_ret = true;
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            double a = m1[i*n+j];
            double b = m2[i*n+j];
            double dev = a - b;
            if (a != 0 && fabs(dev / a) > tolerance){
                printf("matrix differed at index (%d, %d) -> %f & %f\n", i, j, a, b);
                flag_ret = false;
            }
        }
    }
    return flag_ret;
}

bool compare_cnum_matrix(uint64_t n, Cnum *m1, Cnum *m2)
{
    uint64_t i, j;
    bool flag_ret = true;
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            Cnum a = m1[i*n+j];
            Cnum b = m2[i*n+j];
            Cnum dev = Cnum_Sub(a, b);
            if (a.real !=0 && a.imag != 0 && fabs(dev.real / a.real) > tolerance && fabs(dev.imag / a.imag) > tolerance){
                printf("matrix differed at index (%d, %d) -> %f + %f*I & %f + %f*I\n", i, j, a.real, a.imag, b.real, b.imag);
                flag_ret = false;
            }
        }
    }
    return flag_ret;
}

bool check_eigen_left(uint64_t n, double *A, double *B, Cnum *eigenvalue,
                      Cnum *vleft)
{
    Cnum *cA = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *cB = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *vH = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *diag = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *res1 = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *res2 = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *res3 = (Cnum*)malloc(n*n*sizeof(Cnum));
    memcpy(vH, vleft, n*n*sizeof(Cnum));
    Matrix_Hermitian_Transpose_c64(n, n, vH);
    Madd_Set0_c64(n*n, diag);
    uint64_t i, j;
    for (i=0; i<n; i++){
        diag[i*n+i] = eigenvalue[i];
        for (j=0; j<n; j++){
            uint64_t id = i*n + j;
            cA[id].real = A[id];
            cA[id].imag = 0;
            cA[id].real = B[id];
            cB[id].imag = 0;
        }
    }
    Matrix_Multiply_c64(n, n, n, vH, cA, res1);
    Matrix_Multiply_c64(n, n, n, vleft, diag, res2);
    Matrix_Multiply_c64(n, n, n, res2, cB, res3);

    bool flag_same = compare_cnum_matrix(n, res1, res3);

    free(cA);
    free(cB);
    free(vH);
    free(diag);
    free(res1);
    free(res2);
    free(res3);
    return flag_same;
}

bool check_eigen_right(uint64_t n, double *A, double *B, Cnum *eigenvalue,
                       Cnum *vright)
{
    Cnum *cA = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *cB = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *diag = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *res1 = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *res2 = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *res3 = (Cnum*)malloc(n*n*sizeof(Cnum));
    Madd_Set0_c64(n*n, diag);
    uint64_t i, j;
    for (i=0; i<n; i++){
        diag[i*n+i] = eigenvalue[i];
        for (j=0; j<n; j++){
            uint64_t id = i*n + j;
            cA[id].real = A[id];
            cA[id].imag = 0;
            cA[id].real = B[id];
            cB[id].imag = 0;
        }
    }
    Matrix_Multiply_c64(n, n, n, cA, vright, res1);
    Matrix_Multiply_c64(n, n, n, diag, cB, res2);
    Matrix_Multiply_c64(n, n, n, res2, vright, res3);

    bool flag_same = compare_cnum_matrix(n, res1, res3);

    free(cA);
    free(cB);
    free(diag);
    free(res1);
    free(res2);
    free(res3);
    return flag_same;
}

bool check_eigen_cnum_left(uint64_t n, Cnum *A, Cnum *B, Cnum *eigenvalue,
                           Cnum *vleft)
{
    Cnum *vH = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *diag = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *res1 = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *res2 = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *res3 = (Cnum*)malloc(n*n*sizeof(Cnum));
    memcpy(vH, vleft, n*n*sizeof(Cnum));
    Matrix_Hermitian_Transpose_c64(n, n, vH);
    Madd_Set0_c64(n*n, diag);
    uint64_t i, j;
    for (i=0; i<n; i++){
        diag[i*n+i] = eigenvalue[i];
    }
    Matrix_Multiply_c64(n, n, n, vH, A, res1);
    Matrix_Multiply_c64(n, n, n, diag, vH, res2);
    Matrix_Multiply_c64(n, n, n, res2, B, res3);

    bool flag_same = compare_cnum_matrix(n, res1, res3);

    free(vH);
    free(diag);
    free(res1);
    free(res2);
    free(res3);
    return flag_same;
}

bool check_eigen_cnum_right(uint64_t n, Cnum *A, Cnum *B, Cnum *eigenvalue,
                            Cnum *vright)
{
    Cnum *diag = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *res1 = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *res2 = (Cnum*)malloc(n*n*sizeof(Cnum));
    Cnum *res3 = (Cnum*)malloc(n*n*sizeof(Cnum));
    Madd_Set0_c64(n*n, diag);
    uint64_t i, j;
    for (i=0; i<n; i++){
        diag[i*n+i] = eigenvalue[i];
    }
    Matrix_Multiply_c64(n, n, n, A, vright, res1);
    Matrix_Multiply_c64(n, n, n, B, vright, res2);
    Matrix_Multiply_c64(n, n, n, res2, diag, res3);

    bool flag_same = compare_cnum_matrix(n, res1, res3);

    free(diag);
    free(res1);
    free(res2);
    free(res3);
    return flag_same;
}

void NxN_Real_Matrix_Test()
{
    printf("\n===\n%d X %d Real Matrix (Random) Test\n===\n", n, n);
    // allocate memory
    uint64_t nn = (uint64_t)n * n, i, j;
    size_t nn_size = nn * sizeof(double), nn_cnum_size = nn * sizeof(Cnum);
    double *matrix = (double*)malloc(nn_size);
    double *B = (double*)malloc(nn_size);
    double *BB = (double*)malloc(nn_size);
    double *mm = (double*)malloc(nn_size);
    Madd_Set0(n*n, B);
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
        B[i*n + i] = 1;
        for (int j = 0; j < n; j++) {
            matrix[i*n+j] = matrix_cnum[i*n+j].real = Rand(&rng); // random number
            matrix_cnum[i*n+j].imag = 0;
        }
    }
    memcpy(mm, matrix, nn_size);
    memcpy(BB, B, nn_size);
    
    // print original matrix
    printf("Original matrix:\n");
    print_real_matrix(n, n, matrix);

    // cal eigenvalue & eigenvector
    Generalized_Eigen(n, matrix, B, eigenvalue, true, vl, true, vr);

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
    double mm2[4];
    memcpy(mm2, mat2, 4*sizeof(double));
    double diag2[4] = {1, 0, 0, 1};
    double dd2[4];
    memcpy(dd2, diag2, 4*sizeof(double));
    //double mat2[4] = {1, 0, 0, -1};
    Cnum eig2[2];
    Cnum vl2[4], vr2[4];
    print_real_matrix(2, 2, mat2);
    Generalized_Eigen(2, mat2, diag2, eig2, true, vl2, true, vr2);
    printf("Eigenvalues\n");
    print_cnum_matrix(2, 1, eig2);
    printf("Eigenvectors Left\n");
    print_cnum_matrix(2, 2, vl2);
    printf("Eigenvectors Right\n");
    print_cnum_matrix(2, 2, vr2);

    bool flag_left = check_eigen_left(2, mm2, dd2, eig2, vl2);
    if (!flag_left) flag_2_real_verified = false;
    bool flag_right = check_eigen_right(2, mm2, dd2, eig2, vl2);
    if (!flag_right) flag_2_real_verified = false;

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
    Cnum *diag = (Cnum*)malloc(size_nn);
    Cnum *dd = (Cnum*)malloc(size_nn);
    Madd_Set0_c64(n*n, dd);
    Madd_Set0_c64(n*n, diag);
    for (i=0; i<n; i++){
        dd[i*n+i].real = diag[i*n+i].real = 1;
        for (j=0; j<n; j++){
            matrix[i*n+j].real = Rand(&rng);
            matrix[i*n+j].imag = Rand(&rng);
        }
    }
    print_cnum_matrix(n, n, matrix);
    memcpy(mm, matrix, size_nn);
    
    Generalized_Eigen_c64(n, matrix, diag, eigenvalue, true, vl, true, vr);

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
    flag_complex_left_verified = check_eigen_cnum_left(n, mm, dd, eigenvalue, vl);

    // right eigen check
    printf("---\nRight Eigen Check\n---\n");
    //print_cnum_matrix(n, n, dd);
    flag_complex_right_verified = check_eigen_cnum_right(n, mm, dd, eigenvalue, vr);

    free(matrix);
    free(mm);
    free(eigenvalue);
    free(vl);
    free(vr);
    free(diag);
    free(dd);
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