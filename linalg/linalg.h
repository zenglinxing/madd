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

/*
===============================================================================
matrix multiply
===============================================================================
*/
bool Matrix_Multiply(int m, int n, int l,
                     double *a, double *b, double *res);

bool Matrix_Multiply_Naive(uint64_t m, uint64_t n, uint64_t l,
                           double *a, double *b, double *res);

/*
===============================================================================
linear equations
===============================================================================
eq: n x n
vector: n x n_vector
*/
bool Linear_Equations(int n, double *eq, int n_vector, double *vector);
bool Linear_Equations_f32(int n, float *eq, int n_vector, float *vector);
bool Linear_Equations_c64(int n, Cnum *eq, int n_vector, Cnum *vector);
bool Linear_Equations_c32(int n, Cnum32 *eq, int n_vector, Cnum32 *vector);

#endif /* MADD_LINALG_H */