/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fmin/Jacobi-Iteration.c
*/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>
#include"fmin.h"
#include"../basic/basic.h"

#define FMIN_JACOBI_ITERATION__ALGORITHM(num_type) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n = 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (b == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: b is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (solution == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: solution is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    size_t malloc_size = (uint64_t)n*2*sizeof(num_type); \
    num_type *diag = (num_type*)malloc(malloc_size), *x_new = diag + n; \
    if (diag == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for diag & x_new.", __func__, malloc_size); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    uint64_t n1 = n + 1; \
    num_type *p = matrix; \
    for (uint64_t ix=0; ix<n; ix++,p+=n1){ \
        if (*p == 0 || !isnormal(*p)){ \
            free(diag); \
            wchar_t error_info[MADD_ERROR_INFO_LEN]; \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: value of diag %llu (value %f) should be non-zero.", __func__, ix, *p); \
            Madd_Error_Add(MADD_ERROR, error_info); \
            return false; \
        } \
        diag[ix] = 1 / *p; \
    } \
 \
    size_t size_cpy = (uint64_t)n * sizeof(num_type); \
    for (uint64_t i_step=0; i_step<n_step; i_step++){ \
        num_type *prow = matrix; \
        for (uint64_t ix=0; ix<n; ix++, prow+=n){ \
            num_type sum = 0; \
            for (uint64_t iy=0; iy<n; iy++){ \
                if (ix == iy) continue; \
                sum += prow[iy] * solution[iy]; \
            } \
            x_new[ix] = diag[ix] * (b[ix] - sum); \
        } \
        memcpy(solution, x_new, size_cpy); \
    } \
 \
    free(diag); \
    return true; \
} \

bool Fmin_Jacobi_Iteration(uint64_t n, double *matrix, const double *b, double *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION__ALGORITHM(double)

bool Fmin_Jacobi_Iteration_f32(uint64_t n, float *matrix, const float *b, float *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION__ALGORITHM(float)

bool Fmin_Jacobi_Iteration_fl(uint64_t n, long double *matrix, const long double *b, long double *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION__ALGORITHM(long double)

#ifdef ENABLE_QUADPRECISION
bool Fmin_Jacobi_Iteration_f128(uint64_t n, __float128 *matrix, const __float128 *b, __float128 *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION__ALGORITHM(__float128)
#endif /* ENABLE_QUADPRECISION */

#define FMIN_JACOBI_ITERATION_CNUM__ALGORITHM(Cnum, Real_Div_Cnum, Cnum_Add, Cnum_Sub, Cnum_Mul) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n = 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (b == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: b is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (solution == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: solution is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    size_t malloc_size = (uint64_t)n*2*sizeof(Cnum); \
    Cnum *diag = (Cnum*)malloc(malloc_size), *x_new = diag + n; \
    if (diag == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for diag & x_new.", __func__, malloc_size); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    uint64_t n1 = n + 1; \
    Cnum *p = matrix; \
    for (uint64_t ix=0; ix<n; ix++,p+=n1){ \
        if (p->real == 0 && p->imag == 0 || !isnormal(p->real) || !isnormal(p->imag)){ \
            free(diag); \
            wchar_t error_info[MADD_ERROR_INFO_LEN]; \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: value of diag %llu (value %f+%f*I) should be non-zero.", __func__, ix, p->real, p->imag); \
            Madd_Error_Add(MADD_ERROR, error_info); \
            return false; \
        } \
        diag[ix] = Real_Div_Cnum(1, *p); \
    } \
 \
    size_t size_cpy = (uint64_t)n * sizeof(Cnum); \
    for (uint64_t i_step=0; i_step<n_step; i_step++){ \
        Cnum *prow = matrix; \
        for (uint64_t ix=0; ix<n; ix++, prow+=n){ \
            Cnum sum = {.real=0, .imag=0}; \
            for (uint64_t iy=0; iy<n; iy++){ \
                if (ix == iy) continue; \
                Cnum temp = Cnum_Mul(prow[iy], solution[iy]); \
                sum = Cnum_Add(sum, temp); \
            } \
            Cnum temp = Cnum_Sub(b[ix], sum); \
            x_new[ix] = Cnum_Mul(diag[ix], temp); \
        } \
        memcpy(solution, x_new, size_cpy); \
    } \
 \
    free(diag); \
    return true; \
} \

bool Fmin_Jacobi_Iteration_c64(uint64_t n, Cnum *matrix, const Cnum *b, Cnum *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION_CNUM__ALGORITHM(Cnum, Real_Div_Cnum, Cnum_Add, Cnum_Sub, Cnum_Mul)

bool Fmin_Jacobi_Iteration_c32(uint64_t n, Cnum32 *matrix, const Cnum32 *b, Cnum32 *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION_CNUM__ALGORITHM(Cnum32, Real_Div_Cnum_c32, Cnum_Add_c32, Cnum_Sub_c32, Cnum_Mul_c32)

bool Fmin_Jacobi_Iteration_cl(uint64_t n, Cnuml *matrix, const Cnuml *b, Cnuml *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION_CNUM__ALGORITHM(Cnuml, Real_Div_Cnum_cl, Cnum_Add_cl, Cnum_Sub_cl, Cnum_Mul_cl)

#ifdef ENABLE_QUADPRECISION
bool Fmin_Jacobi_Iteration_c128(uint64_t n, Cnum128 *matrix, const Cnum128 *b, Cnum128 *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION_CNUM__ALGORITHM(Cnum128, Real_Div_Cnum_c128, Cnum_Add_c128, Cnum_Sub_c128, Cnum_Mul_c128)
#endif /* ENABLE_QUADPRECISION */