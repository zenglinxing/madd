/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/linalg.h
*/
#ifndef MADD_LINALG_H
#define MADD_LINALG_H

#include<stdint.h>
#include<stdbool.h>
#include"../basic/cnum.h"
#include"linalg.cuh"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

/*
===============================================================================
sparse matrix
===============================================================================
*/
#define SPARSE_MATRIX_COO(num_type, Sparse_Matrix_COO_Unit, Sparse_Matrix_COO) \
typedef struct{ \
    uint64_t x, y; \
    num_type value; \
} Sparse_Matrix_COO_Unit; \
 \
typedef struct{ \
    uint64_t dim, n_unit; \
    Sparse_Matrix_COO_Unit *unit; \
} Sparse_Matrix_COO; \

SPARSE_MATRIX_COO(double, Sparse_Matrix_COO_Unit, Sparse_Matrix_COO)
SPARSE_MATRIX_COO(float, Sparse_Matrix_COO_Unit_f32, Sparse_Matrix_COO_f32)
SPARSE_MATRIX_COO(long double, Sparse_Matrix_COO_Unit_fl, Sparse_Matrix_COO_fl)
#ifdef ENABLE_QUADPRECISION
SPARSE_MATRIX_COO(__float128, Sparse_Matrix_COO_Unit_f128, Sparse_Matrix_COO_f128)
#endif /* ENABLE_QUADPRECISION */

/*
===============================================================================
matrix transpose
===============================================================================
*/
bool Matrix_Transpose(uint64_t m, uint64_t n, double *matrix);
bool Matrix_Transpose_Inplace(uint64_t m, uint64_t n, double *matrix);
bool Matrix_Transpose_Outplace(uint64_t m, uint64_t n, double *matrix, double *newmat);

bool Matrix_Transpose_f32(uint32_t m, uint32_t n, float *matrix);
bool Matrix_Transpose_Inplace_f32(uint32_t m, uint32_t n, float *matrix);
bool Matrix_Transpose_Outplace_f32(uint32_t m, uint32_t n, float *matrix, float *newmat);

bool Matrix_Transpose_fl(uint64_t m, uint64_t n, long double *matrix);
bool Matrix_Transpose_Inplace_fl(uint64_t m, uint64_t n, long double *matrix);
bool Matrix_Transpose_Outplace_fl(uint64_t m, uint64_t n, long double *matrix, long double *newmat);

#ifdef ENABLE_QUADPRECISION
bool Matrix_Transpose_f128(uint64_t m, uint64_t n, __float128 *matrix);
bool Matrix_Transpose_Inplace_f128(uint64_t m, uint64_t n, __float128 *matrix);
bool Matrix_Transpose_Outplace_f128(uint64_t m, uint64_t n, __float128 *matrix, __float128 *newmat);
#endif /* ENABLE_QUADPRECISION */

bool Matrix_Transpose_c64(uint64_t m, uint64_t n, Cnum *matrix);
bool Matrix_Transpose_Inplace_c64(uint64_t m, uint64_t n, Cnum *matrix);
bool Matrix_Transpose_Outplace_c64(uint64_t m, uint64_t n, Cnum *matrix, Cnum *newmat);

bool Matrix_Transpose_c32(uint64_t m, uint64_t n, Cnum32 *matrix);
bool Matrix_Transpose_Inplace_c32(uint64_t m, uint64_t n, Cnum32 *matrix);
bool Matrix_Transpose_Outplace_c32(uint64_t m, uint64_t n, Cnum32 *matrix, Cnum32 *newmat);

bool Matrix_Transpose_cl(uint64_t m, uint64_t n, Cnuml *matrix);
bool Matrix_Transpose_Inplace_cl(uint64_t m, uint64_t n, Cnuml *matrix);
bool Matrix_Transpose_Outplace_cl(uint64_t m, uint64_t n, Cnuml *matrix, Cnuml *newmat);

#ifdef ENABLE_QUADPRECISION
bool Matrix_Transpose_c128(uint64_t m, uint64_t n, Cnum128 *matrix);
bool Matrix_Transpose_Inplace_c128(uint64_t m, uint64_t n, Cnum128 *matrix);
bool Matrix_Transpose_Outplace_c128(uint64_t m, uint64_t n, Cnum128 *matrix, Cnum128 *newmat);
#endif /* ENABLE_QUADPRECISION */

/* Hermitian Transpose */
bool Matrix_Hermitian_Transpose_c64(uint64_t m, uint64_t n, Cnum *matrix);
bool Matrix_Hermitian_Transpose_c32(uint64_t m, uint64_t n, Cnum32 *matrix);
bool Matrix_Hermitian_Transpose_cl(uint64_t m, uint64_t n, Cnuml *matrix);
#ifdef ENABLE_QUADPRECISION
bool Matrix_Hermitian_Transpose_c128(uint64_t m, uint64_t n, Cnum128 *matrix);
#endif /* ENABLE_QUADPRECISION */

/*
===============================================================================
matrix multiply
===============================================================================
*/
bool Matrix_Multiply(int32_t m, int32_t n, int32_t l,
                     double *a, double *b, double *res);
bool Matrix_Multiply_f32(int32_t m, int32_t n, int32_t l,
                         float *a, float *b, float *res);
bool Matrix_Multiply_c64(int32_t m, int32_t n, int32_t l,
                         Cnum *a, Cnum *b, Cnum *res);
bool Matrix_Multiply_c32(int32_t m, int32_t n, int32_t l,
                         Cnum32 *a, Cnum32 *b, Cnum32 *res);

bool Matrix_Multiply_Naive(uint64_t m, uint64_t n, uint64_t l,
                           double *a, double *b, double *res);

/*
===============================================================================
matrix inverse
===============================================================================
*/
bool Matrix_Inverse(int32_t n, double *matrix);
bool Matrix_Inverse_f32(int32_t n, float *matrix);
bool Matrix_Inverse_c64(int32_t n, Cnum *matrix);
bool Matrix_Inverse_c32(int32_t n, Cnum32 *matrix);

/*
===============================================================================
linear equations
===============================================================================
eq: n x n
vector: n x n_vector
*/
bool Linear_Equations(int32_t n, double *eq, int32_t n_vector, double *vector);
bool Linear_Equations_f32(int32_t n, float *eq, int32_t n_vector, float *vector);
bool Linear_Equations_c64(int32_t n, Cnum *eq, int32_t n_vector, Cnum *vector);
bool Linear_Equations_c32(int32_t n, Cnum32 *eq, int32_t n_vector, Cnum32 *vector);

bool Linear_Equations_Tridiagonal(int32_t n, double *lower, double *diag, double *upper,
                                  int32_t n_vector, double *vector);
bool Linear_Equations_Tridiagonal_f32(int32_t n, float *lower, float *diag, float *upper,
                                      int32_t n_vector, float *vector);
bool Linear_Equations_Tridiagonal_c64(int32_t n, Cnum *lower, Cnum *diag, Cnum *upper,
                                      int32_t n_vector, Cnum *vector);
bool Linear_Equations_Tridiagonal_c32(int32_t n, Cnum32 *lower, Cnum32 *diag, Cnum32 *upper,
                                      int32_t n_vector, Cnum32 *vector);

/*
===============================================================================
eigenvalue & eigenvector
===============================================================================
*/
bool Eigen(int32_t n, double *matrix, Cnum *eigenvalue,
           bool flag_left, Cnum *eigenvector_left,
           bool flag_right, Cnum *eigenvector_right);
bool Eigen_f32(int32_t n, float *matrix, Cnum32 *eigenvalue,
               bool flag_left, Cnum32 *eigenvector_left,
               bool flag_right, Cnum32 *eigenvector_right);
bool Eigen_c64(int32_t n, Cnum *matrix, Cnum *eigenvalue,
               bool flag_left, Cnum *eigenvector_left,
               bool flag_right, Cnum *eigenvector_right);
bool Eigen_c32(int32_t n, Cnum32 *matrix, Cnum32 *eigenvalue,
               bool flag_left, Cnum32 *eigenvector_left,
               bool flag_right, Cnum32 *eigenvector_right);

bool Generalized_Eigen(int32_t n, double *matrix_A, double *matrix_B,
                       Cnum *eigenvalue,
                       bool flag_left, Cnum *eigenvector_left,
                       bool flag_right, Cnum *eigenvector_right);
bool Generalized_Eigen_f32(int32_t n, float *matrix_A, float *matrix_B,
                           Cnum32 *eigenvalue,
                           bool flag_left, Cnum32 *eigenvector_left,
                           bool flag_right, Cnum32 *eigenvector_right);
bool Generalized_Eigen_c64(int32_t n, Cnum *matrix_A, Cnum *matrix_B,
                           Cnum *eigenvalue,
                           bool flag_left, Cnum *eigenvector_left,
                           bool flag_right, Cnum *eigenvector_right);
bool Generalized_Eigen_c32(int32_t n, Cnum32 *matrix_A, Cnum32 *matrix_B,
                           Cnum32 *eigenvalue,
                           bool flag_left, Cnum32 *eigenvector_left,
                           bool flag_right, Cnum32 *eigenvector_right);

/*
===============================================================================
determinant
===============================================================================
*/
bool Determinant(int32_t n, double *matrix, double *res);
bool Determinant_f32(int32_t n, float *matrix, float *res);
bool Determinant_c64(int32_t n, Cnum *matrix, Cnum *res);
bool Determinant_c32(int32_t n, Cnum32 *matrix, Cnum32 *res);

bool Determinant_Bareiss(uint64_t n, double *mat, double *res);
bool Determinant_Bareiss_f32(uint64_t n, float *mat, float *res);
bool Determinant_Bareiss_fl(uint64_t n, long double *mat, long double *res);
#ifdef ENABLE_QUADPRECISION
bool Determinant_Bareiss_f128(uint64_t n, __float128 *mat, __float128 *res);
#endif /* ENABLE_QUADPRECISION */
bool Determinant_Bareiss_c64(uint64_t n, Cnum *mat, Cnum *res);
bool Determinant_Bareiss_c32(uint64_t n, Cnum32 *mat, Cnum32 *res);
bool Determinant_Bareiss_cl(uint64_t n, Cnuml *mat, Cnuml *res);
#ifdef ENABLE_QUADPRECISION
bool Determinant_Bareiss_c128(uint64_t n, Cnum128 *mat, Cnum128 *res);
#endif /* ENABLE_QUADPRECISION */

#endif /* MADD_LINALG_H */