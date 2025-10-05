/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/eigen.c
*/
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include<stdbool.h>
#define HAVE_LAPACK_CONFIG_H
#include<lapacke.h>

#include"linalg.h"
#include"../basic/basic.h"

#define GENERALIZED_EIGEN_REAL__ALGORITHM(Cnum, real_num_type, LAPACKE_dggev, Matrix_Transpose, Cnum_Div_Real) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix_A == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix_A is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix_B == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix_B is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (eigenvalue == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: eigenvalue is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (flag_left && eigenvector_left == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: flag_left is true, but eigenvector_left is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (flag_right && eigenvector_right == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: flag_right is true, but eigenvector_right is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    real_num_type *wr = (real_num_type*)eigenvalue, *wi = (real_num_type*)wr + n; \
    char jobvl = (flag_left) ? 'V' : 'N', jobvr = (flag_right) ? 'V' : 'N'; \
    uint64_t nn = (uint64_t)n * n; \
    size_t size_nn = nn*sizeof(real_num_type), size_n = n*sizeof(real_num_type); \
    size_t size_alloc = size_n; \
    if (flag_left){ \
        size_alloc += size_nn; \
    } \
    if (flag_right){ \
        size_alloc += size_nn; \
    } \
    real_num_type *beta = (real_num_type*)malloc(size_alloc); \
    real_num_type *vl = beta + n; \
    real_num_type *vr = (flag_right) ? vl + nn : vl; \
    if (beta == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for beta, maybe vl & vr.", __func__, size_alloc); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    /* eigen */ \
    lapack_int info = LAPACKE_dggev(LAPACK_ROW_MAJOR, jobvl, jobvr, n, \
                                    matrix_A, n, \
                                    matrix_B, n, \
                                    wr, wi, \
                                    beta, \
                                    vl, n, \
                                    vr, n); \
    if (info != 0){ \
        free(beta); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        if (info < 0){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the %d-th argument had an illegal value.", __func__, -info); \
        }else if (info <= n){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: The QZ iteration failed.  No eigenvectors have been calculated, but ALPHAR(j), ALPHAI(j), and BETA(j) should be correct for j=%d,...,N.", __func__, info+1); \
        }else if (info == n+1){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: other than QZ iteration failed in %hs.", __func__, "dhgeqz"); \
        }else{ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: error return from %hs.", __func__, "dtgevc"); \
        } \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    Matrix_Transpose(2, n, (real_num_type*)eigenvalue); \
    for (uint64_t i=0; i<n; i++){ \
        if (beta[i] == 0){ \
            wchar_t error_info[MADD_ERROR_INFO_LEN]; \
            if (eigenvalue[i].real == 0 && eigenvalue[i].imag == 0){ \
                swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %llu-th eigenvalue is infinity.", __func__, i); \
            }else{ \
                swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %llu-th eigenvalue and %llu-th ratio are both 0. This may indicate that there are something wrong with your input.", __func__, i, i); \
            } \
            Madd_Error_Add(MADD_WARNING, error_info); \
        } \
        eigenvalue[i] = Cnum_Div_Real(eigenvalue[i], beta[i]); \
    } \
 \
    if (flag_left || flag_right){ \
        uint64_t i, j; \
        for (i=0; i<n; i++){ \
            Cnum *le = eigenvector_left + i, *re = eigenvector_right + i; \
            real_num_type *lv = vl + i, *rv = vr + i; \
            if (eigenvalue[i].imag == 0){ \
                for (j=0; j<n; j++,lv+=n,rv+=n,le+=n,re+=n){ \
                    if (flag_left){ \
                        le->real = *lv; \
                        le->imag = 0; \
                    } \
                    if (flag_right){ \
                        re->real = *rv; \
                        re->imag = 0; \
                    } \
                } \
            }else{ \
                for (j=0; j<n; j++,lv+=n,rv+=n,le+=n,re+=n){ \
                    if (flag_left){ \
                        le->real = le[1].real = *lv; \
                        le->imag = lv[1]; \
                        le[1].imag = -lv[1]; \
                    } \
                    if (flag_right){ \
                        re->real = re[1].real = *rv; \
                        re->imag = rv[1]; \
                        re[1].imag = -rv[1]; \
                    } \
                } \
                i ++; \
                continue; \
            } \
        } \
    } \
 \
    free(beta); \
    return true; \
} \

bool Generalized_Eigen(int n, double *matrix_A, double *matrix_B,
                       Cnum *eigenvalue,
                       bool flag_left, Cnum *eigenvector_left,
                       bool flag_right, Cnum *eigenvector_right)
GENERALIZED_EIGEN_REAL__ALGORITHM(Cnum, double, LAPACKE_dggev, Matrix_Transpose, Cnum_Div_Real)

bool Generalized_Eigen_f32(int n, float *matrix_A, float *matrix_B,
                           Cnum32 *eigenvalue,
                           bool flag_left, Cnum32 *eigenvector_left,
                           bool flag_right, Cnum32 *eigenvector_right)
GENERALIZED_EIGEN_REAL__ALGORITHM(Cnum32, float, LAPACKE_sggev, Matrix_Transpose_f32, Cnum_Div_Real_c32)

#define GENERALIZED_EIGEN_CNUM__ALGORITHM(Cnum, LAPACKE_zggev, lapack_complex_double, Cnum_Div) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix_A == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix_A is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix_B == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix_B is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (eigenvalue == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: eigenvalue is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (flag_left && eigenvector_left == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: flag_left is true, but eigenvector_left is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (flag_right && eigenvector_right == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: flag_right is true, but eigenvector_right is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    Cnum *beta = (Cnum*)malloc(n*sizeof(Cnum)); \
    char jobvl = (flag_left) ? 'V' : 'N', jobvr = (flag_right) ? 'V' : 'N'; \
    lapack_int info = LAPACKE_zggev(LAPACK_ROW_MAJOR, jobvl, jobvr, n, \
                                    (lapack_complex_double*)matrix_A, n, \
                                    (lapack_complex_double*)matrix_B, n, \
                                    (lapack_complex_double*)eigenvalue, \
                                    (lapack_complex_double*)beta, \
                                    (lapack_complex_double*)eigenvector_left, n, \
                                    (lapack_complex_double*)eigenvector_right, n); \
    if (info != 0){ \
        free(beta); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        if (info < 0){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the %d-th argument had an illegal value.", __func__, -info); \
        }else if (info <= n){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: The QZ iteration failed.  No eigenvectors have been calculated, but ALPHAR(j), ALPHAI(j), and BETA(j) should be correct for j=%d,...,N.", __func__, info+1); \
        }else if (info == n+1){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: other than QZ iteration failed in %hs.", __func__, "dhgeqz"); \
        }else{ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: error return from %hs.", __func__, "dtgevc"); \
        } \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    for (uint64_t i=0; i<n; i++){ \
        if (beta[i].real == 0 && beta[i].imag == 0){ \
            wchar_t error_info[MADD_ERROR_INFO_LEN]; \
            if (eigenvalue[i].real == 0 && eigenvalue[i].imag == 0){ \
                swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %llu-th eigenvalue is infinity.", __func__, i); \
            }else{ \
                swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %llu-th eigenvalue and %llu-th ratio are both 0. This may indicate that there are something wrong with your input.", __func__, i, i); \
            } \
            Madd_Error_Add(MADD_WARNING, error_info); \
        } \
        eigenvalue[i] = Cnum_Div(eigenvalue[i], beta[i]); \
    } \
 \
    free(beta); \
    return true; \
} \

bool Generalized_Eigen_c64(int n, Cnum *matrix_A, Cnum *matrix_B,
                           Cnum *eigenvalue,
                           bool flag_left, Cnum *eigenvector_left,
                           bool flag_right, Cnum *eigenvector_right)
GENERALIZED_EIGEN_CNUM__ALGORITHM(Cnum, LAPACKE_zggev, lapack_complex_double, Cnum_Div)

bool Generalized_Eigen_c32(int n, Cnum32 *matrix_A, Cnum32 *matrix_B,
                           Cnum32 *eigenvalue,
                           bool flag_left, Cnum32 *eigenvector_left,
                           bool flag_right, Cnum32 *eigenvector_right)
GENERALIZED_EIGEN_CNUM__ALGORITHM(Cnum32, LAPACKE_cggev, lapack_complex_float, Cnum_Div_c32)