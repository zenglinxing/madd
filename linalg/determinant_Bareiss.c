/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/determinant_Bareiss.c
*/
#include<stdint.h>
#include<stdbool.h>
#include"../basic/basic.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define DET_BAREISS__ALGORITHM(num_type) \
{ \
    if (res == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: res is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (n == 0){ \
        *res = 0; \
        return true; \
    } \
 \
    uint64_t i, j, k, n_1=n-1; \
    num_type pivot=1, *p_kk, *p_ij, *p_ik, *p_kj; \
    for (k=0;k<n_1;k++){ \
        p_kk=matrix + k * (n + 1); \
        for (i=k+1;i<n;i++){ \
            p_ij=matrix + i * n + k + 1; \
            p_ik=matrix + i * n + k; \
            p_kj=matrix + k * (n + 1) + 1; \
            for (j=k+1;j<n;j++,p_ij++,p_kj++){ \
                /*matrix[i][j]=matrix[k][k]*matrix[i][j]-matrix[i][k]*matrix[k][j];*/ \
                /*matrix[i][j]/=pivot;*/ \
                *p_ij=*p_kk * *p_ij - *p_ik * *p_kj; \
                *p_ij /= pivot; \
            } \
        } \
        pivot = *p_kk; \
        if (pivot == 0){ \
            *res = 0; \
            return true; \
        } \
    } \
    *res = matrix[n * n - 1]; \
    return true; \
} \

bool Determinant_Bareiss(uint64_t n, double *matrix, double *res)
DET_BAREISS__ALGORITHM(double)

bool Determinant_Bareiss_f32(uint64_t n, float *matrix, float *res)
DET_BAREISS__ALGORITHM(float)

bool Determinant_Bareiss_fl(uint64_t n, long double *matrix, long double *res)
DET_BAREISS__ALGORITHM(long double)

#ifdef ENABLE_QUADPRECISION
bool Determinant_Bareiss_f128(uint64_t n, __float128 *matrix, __float128 *res)
DET_BAREISS__ALGORITHM(__float128)
#endif /* ENABLE_QUADPRECISION */

#define DET_BAREISS_CNUM__ALGORITHM(Cnum, Cnum_Mul, Cnum_Sub, Cnum_Div) \
{ \
    if (res == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: res is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (n == 0){ \
        res->real = res->imag = 0; \
        return true; \
    } \
 \
    uint64_t i, j, k, n_1=n-1; \
    Cnum pivot={.real=1, .imag=0}, *p_kk, *p_ij, *p_ik, *p_kj; \
    for (k=0; k<n_1; k++){ \
        p_kk=matrix + k * (n + 1); \
        for (i=k+1;i<n;i++){ \
            p_ij=matrix + i * n + k + 1; \
            p_ik=matrix + i * n + k; \
            p_kj=matrix + k * (n + 1) + 1; \
            for (j=k+1; j<n; j++,p_ij++,p_kj++){ \
                /*matrix[i][j]=matrix[k][k]*matrix[i][j]-matrix[i][k]*matrix[k][j];*/ \
                /*matrix[i][j]/=pivot;*/ \
                Cnum a = Cnum_Mul(*p_kk, *p_ij); \
                Cnum b = Cnum_Mul(*p_ik, *p_kj); \
                *p_ij = Cnum_Sub(a, b); \
                *p_ij = Cnum_Div(*p_ij, pivot); \
            } \
        } \
        pivot = *p_kk; \
        if (pivot.real == 0 && pivot.imag == 0){ \
            res->real = res->imag = 0; \
            return true; \
        } \
    } \
    *res = matrix[n * n - 1]; \
    return true; \
} \

bool Determinant_Bareiss_c64(uint64_t n, Cnum *matrix, Cnum *res)
DET_BAREISS_CNUM__ALGORITHM(Cnum, Cnum_Mul, Cnum_Sub, Cnum_Div)

bool Determinant_Bareiss_c32(uint64_t n, Cnum32 *matrix, Cnum32 *res)
DET_BAREISS_CNUM__ALGORITHM(Cnum32, Cnum_Mul_c32, Cnum_Sub_c32, Cnum_Div_c32)

bool Determinant_Bareiss_cl(uint64_t n, Cnuml *matrix, Cnuml *res)
DET_BAREISS_CNUM__ALGORITHM(Cnuml, Cnum_Mul_cl, Cnum_Sub_cl, Cnum_Div_cl)

#ifdef ENABLE_QUADPRECISION
bool Determinant_Bareiss_c128(uint64_t n, Cnum128 *matrix, Cnum128 *res)
DET_BAREISS_CNUM__ALGORITHM(Cnum128, Cnum_Mul_c128, Cnum_Sub_c128, Cnum_Div_c128)
#endif /* ENABLE_QUADPRECISION */