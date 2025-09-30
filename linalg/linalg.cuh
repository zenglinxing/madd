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
bool Matrix_Multiply_cuda(int64_t m, int64_t n, int64_t l,
                          double *a, double *b, double *res);
bool Matrix_Multiply_cuda_f32(int64_t m, int64_t n, int64_t l,
                              float *a, float *b, float *res);
bool Matrix_Multiply_cuda_c64(int64_t m, int64_t n, int64_t l,
                              Cnum *a, Cnum *b, Cnum *res);
bool Matrix_Multiply_cuda_c32(int64_t m, int64_t n, int64_t l,
                              Cnum32 *a, Cnum32 *b, Cnum32 *res);


/*
===============================================================================
cublas error
===============================================================================
*/
void Madd_cublasCreate_error(int ret, const char *func_name);
void Madd_cublasSetStream_error(int ret, const char *func_name);

#endif /* MADD_LINALG_CUH */