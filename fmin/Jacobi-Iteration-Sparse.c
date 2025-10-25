/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fmin/Jacobi-Iteration-Sparse.c
*/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>
#include"fmin.h"
#include"../basic/basic.h"
#include"../linalg/linalg.h"

#define FMIN_JACOBI_ITERATION_SPARSE__ALGORITHM(num_type, Sparse_Matrix_COO_Unit, isnormal, Madd_Set0) \
{ \
    if (matrix == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix->dim == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix->dim = 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix->n_unit == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix->n_unit = 0.", __func__); \
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
    for (uint64_t i=0; i<matrix->n_unit; i++){ \
        Sparse_Matrix_COO_Unit *p = matrix->unit + i; \
        if (p->x >= matrix->dim || p->y >= matrix->dim){ \
            wchar_t error_info[MADD_ERROR_INFO_LEN]; \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the matrix element (%llu,%llu) exceeds the dimension (%llu) of matrix.", __func__, p->x, p->y, matrix->dim); \
            Madd_Error_Add(MADD_ERROR, error_info); \
            return false; \
        } \
    } \
 \
    size_t malloc_size = (uint64_t)matrix->dim*2*sizeof(num_type); \
    num_type *diag = (num_type*)malloc(malloc_size), *x_new = diag + matrix->dim; \
    if (diag == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for diag & x_new.", __func__, malloc_size); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    for (uint64_t i=0; i<matrix->dim; i++){ \
        diag[i] = INFINITY; \
    } \
    for (uint64_t i=0; i<matrix->n_unit; i++){ \
        Sparse_Matrix_COO_Unit *p = matrix->unit + i; \
        if (p->x == p->y){ \
            if (p->value == 0){ \
                free(diag); \
                wchar_t error_info[MADD_ERROR_INFO_LEN]; \
                swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: got 0 at diag %llu.", __func__); \
                Madd_Error_Add(MADD_ERROR, error_info); \
                return false; \
            } \
            diag[p->x] = 1 / p->value; \
        } \
    } \
    for (uint64_t ix=0; ix<matrix->dim; ix++){ \
        if (!isnormal(diag[ix])){ \
            free(diag); \
            wchar_t error_info[MADD_ERROR_INFO_LEN]; \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: got %f at diag %llu.", __func__, (double)diag[ix]); \
            Madd_Error_Add(MADD_ERROR, error_info); \
            return false; \
        } \
    } \
 \
    size_t size_cpy = (uint64_t)matrix->dim * sizeof(num_type); \
    for (uint64_t i_step=0; i_step < n_step; i_step++){ \
        Madd_Set0(matrix->dim, x_new); \
        Sparse_Matrix_COO_Unit *p = matrix->unit; \
        for (uint64_t i_unit=0; i_unit<matrix->n_unit; i_unit++, p++){ \
            if (p->x == p->y) continue; \
            x_new[p->x] += p->value * solution[p->y]; \
        } \
        for (uint64_t ix=0; ix<matrix->dim; ix++){ \
            x_new[ix] = diag[ix] * (b[ix] - x_new[ix]); \
        } \
        memcpy(solution, x_new, size_cpy); \
    } \
 \
    free(diag); \
    return true; \
} \

bool Fmin_Jacobi_Iteration_Sparse(Sparse_Matrix_COO *matrix, const double *b, double *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION_SPARSE__ALGORITHM(double, Sparse_Matrix_COO_Unit, isnormal, Madd_Set0)

bool Fmin_Jacobi_Iteration_Sparse_f32(Sparse_Matrix_COO_f32 *matrix, const float *b, float *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION_SPARSE__ALGORITHM(float, Sparse_Matrix_COO_Unit_f32, isnormal, Madd_Set0_f32)

bool Fmin_Jacobi_Iteration_Sparse_fl(Sparse_Matrix_COO_fl *matrix, const long double *b, long double *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION_SPARSE__ALGORITHM(long double, Sparse_Matrix_COO_Unit_fl, isnormal, Madd_Set0_fl)

#ifdef ENABLE_QUADPRECISION
bool Fmin_Jacobi_Iteration_Sparse_f128(Sparse_Matrix_COO_f128 *matrix, const __float128 *b, __float128 *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION_SPARSE__ALGORITHM(__float128, Sparse_Matrix_COO_Unit_f128, isnormal, Madd_Set0_f128)
#endif /* ENABLE_QUADPRECISION */

#define FMIN_JACOBI_ITERATION_SPARSE_CNUM__ALGORITHM(Cnum, real_type, Sparse_Matrix_COO_Unit_c64, isnormal, Madd_Set0_c64, \
                                                     Real_Div_Cnum, Cnum_Add, Cnum_Sub, Cnum_Mul) \
{ \
    if (matrix == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix->dim == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix->dim = 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix->n_unit == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix->n_unit = 0.", __func__); \
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
    for (uint64_t i=0; i<matrix->n_unit; i++){ \
        Sparse_Matrix_COO_Unit_c64 *p = matrix->unit + i; \
        if (p->x >= matrix->dim || p->y >= matrix->dim){ \
            wchar_t error_info[MADD_ERROR_INFO_LEN]; \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the matrix element (%llu,%llu) exceeds the dimension (%llu) of matrix.", __func__, p->x, p->y, matrix->dim); \
            Madd_Error_Add(MADD_ERROR, error_info); \
            return false; \
        } \
    } \
 \
    size_t malloc_size = (uint64_t)matrix->dim*2*sizeof(Cnum); \
    Cnum *diag = (Cnum*)malloc(malloc_size), *x_new = diag + matrix->dim; \
    if (diag == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for diag & x_new.", __func__, malloc_size); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    for (uint64_t i=0; i<matrix->dim; i++){ \
        diag[i].real = diag[i].imag = INFINITY; \
    } \
    for (uint64_t i=0; i<matrix->n_unit; i++){ \
        Sparse_Matrix_COO_Unit_c64 *p = matrix->unit + i; \
        if (p->x == p->y){ \
            if (p->value.real == 0 && p->value.imag == 0){ \
                free(diag); \
                wchar_t error_info[MADD_ERROR_INFO_LEN]; \
                swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: got 0 at diag %llu.", __func__); \
                Madd_Error_Add(MADD_ERROR, error_info); \
                return false; \
            } \
            diag[p->x] = Real_Div_Cnum(1, p->value); \
        } \
    } \
    for (uint64_t ix=0; ix<matrix->dim; ix++){ \
        if (!isnormal(diag[ix].real) || !isnormal(diag[ix].imag)){ \
            free(diag); \
            wchar_t error_info[MADD_ERROR_INFO_LEN]; \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: got %f+%f*I at diag %llu.", __func__, (real_type)diag[ix].real, (real_type)diag[ix].imag); \
            Madd_Error_Add(MADD_ERROR, error_info); \
            return false; \
        } \
    } \
 \
    size_t size_cpy = (uint64_t)matrix->dim * sizeof(Cnum); \
    for (uint64_t i_step=0; i_step < n_step; i_step++){ \
        Madd_Set0_c64(matrix->dim, x_new); \
        Sparse_Matrix_COO_Unit_c64 *p = matrix->unit; \
        for (uint64_t i_unit=0; i_unit<matrix->n_unit; i_unit++, p++){ \
            if (p->x == p->y) continue; \
            Cnum temp = Cnum_Mul(p->value, solution[p->y]); \
            x_new[p->x] = Cnum_Add(x_new[p->x], temp); \
        } \
        for (uint64_t ix=0; ix<matrix->dim; ix++){ \
            Cnum temp = Cnum_Sub(b[ix], x_new[ix]); \
            x_new[ix] = Cnum_Mul(diag[ix], temp); \
        } \
        memcpy(solution, x_new, size_cpy); \
    } \
 \
    free(diag); \
    return true; \
} \

bool Fmin_Jacobi_Iteration_Sparse_c64(Sparse_Matrix_COO_c64 *matrix, const Cnum *b, Cnum *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION_SPARSE_CNUM__ALGORITHM(Cnum, double, Sparse_Matrix_COO_Unit_c64, isnormal, Madd_Set0_c64, Real_Div_Cnum, Cnum_Add, Cnum_Sub, Cnum_Mul)

bool Fmin_Jacobi_Iteration_Sparse_c32(Sparse_Matrix_COO_c32 *matrix, const Cnum32 *b, Cnum32 *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION_SPARSE_CNUM__ALGORITHM(Cnum32, float, Sparse_Matrix_COO_Unit_c32, isnormal, Madd_Set0_c32, Real_Div_Cnum_c32, Cnum_Add_c32, Cnum_Sub_c32, Cnum_Mul_c32)

bool Fmin_Jacobi_Iteration_Sparse_cl(Sparse_Matrix_COO_cl *matrix, const Cnuml *b, Cnuml *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION_SPARSE_CNUM__ALGORITHM(Cnuml, long double, Sparse_Matrix_COO_Unit_cl, isnormal, Madd_Set0_cl, Real_Div_Cnum_cl, Cnum_Add_cl, Cnum_Sub_cl, Cnum_Mul_cl)

#ifdef ENABLE_QUADPRECISION
bool Fmin_Jacobi_Iteration_Sparse_c128(Sparse_Matrix_COO_c128 *matrix, const Cnum128 *b, Cnum128 *solution, uint64_t n_step)
FMIN_JACOBI_ITERATION_SPARSE_CNUM__ALGORITHM(Cnum128, __float128, Sparse_Matrix_COO_Unit_c128, isnormal, Madd_Set0_c128, Real_Div_Cnum_c128, Cnum_Add_c128, Cnum_Sub_c128, Cnum_Mul_c128)
#endif /* ENABLE_QUADPRECISION */