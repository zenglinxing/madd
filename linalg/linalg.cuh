/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/linalg.cuh
*/
#ifndef MADD_LINALG_CUH
#define MADD_LINALG_CUH

#include<stdint.h>
#include<stdbool.h>
#include"../basic/cnum.h"

/*
===============================================================================
matrix multiply
===============================================================================
*/
bool Matrix_Multiply_cuda(int m, int n, int l,
                          double *a, double *b, double *res);
bool Matrix_Multiply_cuda_f32(int m, int n, int l,
                              float *a, float *b, float *res);
bool Matrix_Multiply_cuda_c64(int m, int n, int l,
                              Cnum *a, Cnum *b, Cnum *res);
bool Matrix_Multiply_cuda_c32(int m, int n, int l,
                              Cnum32 *a, Cnum32 *b, Cnum32 *res);

bool Matrix_Multiply_cuda64(int64_t m, int64_t n, int64_t l,
                          double *a, double *b, double *res);
bool Matrix_Multiply_cuda64_f32(int64_t m, int64_t n, int64_t l,
                              float *a, float *b, float *res);
bool Matrix_Multiply_cuda64_c64(int64_t m, int64_t n, int64_t l,
                              Cnum *a, Cnum *b, Cnum *res);
bool Matrix_Multiply_cuda64_c32(int64_t m, int64_t n, int64_t l,
                              Cnum32 *a, Cnum32 *b, Cnum32 *res);

/*
===============================================================================
linear equations
===============================================================================
eq: n x n
vector: n x n_vector
*/
bool Linear_Equations_cuda(int n, double *eq, int n_vector, double *vector);
bool Linear_Equations_cuda_f32(int n, float *eq, int n_vector, float *vector);
bool Linear_Equations_cuda_c64(int n, Cnum *eq, int n_vector, Cnum *vector);
bool Linear_Equations_cuda_c32(int n, Cnum32 *eq, int n_vector, Cnum32 *vector);

/*
===============================================================================
eigenvalue & eigenvector
===============================================================================
*/
bool Eigen_cuda64(int64_t n, double *matrix,
                  Cnum *eigenvalue,
                  bool flag_left, Cnum *eigenvector_left,
                  bool flag_right, Cnum *eigenvector_right);

/*
===============================================================================
cublas error
===============================================================================
*/
void Madd_cublasCreate_error(int ret, const char *func_name);
void Madd_cublasSetStream_error(int ret, const char *func_name);

/*
===============================================================================
cusolver error
===============================================================================
*/
void Madd_cusolverDnCreate_error(int ret, const char *func_name);
void Madd_cusolverDnSetStream_error(int ret, const char *func_name);
void Madd_cusolverDnCreateParams_error(int ret, const char *func_name);

#endif /* MADD_LINALG_CUH */