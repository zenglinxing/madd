/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./integrate/int_gauss.c
*/
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<wchar.h>
#include<stdbool.h>
#define HAVE_LAPACK_CONFIG_H
#include<lapacke.h>
#include"integrate.h"
#include"../polynomial/poly1d.h"
#include"../special_func/special_func.h"
#include"../basic/basic.h"

#define INTEGRATE_GAUSS_LEGENDRE_X__ALGORITHM(num_type, integer_type, sqrt, LAPACKE_dsteqr) \
{ \
    if (n_int_ == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
    if (x_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given x_int is NULL.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
 \
    integer_type n_int = n_int_; \
    uint64_t /*nn=(uint64_t)n_int*n_int,*/ i; \
    size_t n1_size = (uint64_t)(n_int-1)*sizeof(num_type); \
    num_type *subdiag=(num_type*)malloc(n1_size), b; \
    if (subdiag == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for matrix.", __func__, n1_size); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    x_int[0] = 0; \
    for (i=1; i<n_int; i++){ \
        b = i/sqrt(4.*i*i-1); \
        subdiag[i-1] = b; \
        x_int[i] = 0; \
    } \
    /* eigen */ \
    char compz = 'N'; /* cal eigenvalues only, without eigenvectors */ \
    lapack_int info = LAPACKE_dsteqr(LAPACK_ROW_MAJOR, compz, n_int, x_int, subdiag, NULL, n_int); \
 \
    free(subdiag); \
 \
    if (info){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        if (info < 0){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: (source file: %hs)(line: %d) from LAPACKE_dsteqr: the %d-th argument had an illegal value.", __func__, __FILE__, __LINE__, -info); \
        }else{ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: (source file: %hs)(line: %d) from LAPACKE_dsteqr: the algorithm has failed to find all the eigenvalues in a total of 30*N iterations; %d elements of E have not converged to zero; on exit, D and E contain the elements of a symmetric tridiagonal matrix which is orthogonally similar to the original matrix.", __func__, __FILE__, __LINE__, info); \
        } \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    return true; \
} \

#define INTEGRATE_GAUSS_LEGENDRE_W__ALGORITHM(integer_type, Poly1d, Poly1d_Create, Special_Func_Legendre, Poly1d_Derivative, Poly1d_Value, Poly1d_Free) \
{ \
    if (n_int == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
    if (x_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given x_int is NULL.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
    if (w_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given w_int is NULL.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
 \
    integer_type i; \
    Poly1d poly = Poly1d_Create(n_int, 0), dpoly = Poly1d_Create(n_int-1, 0); \
    Special_Func_Legendre(&poly); \
    /*printf("legendre poly\n"); \
    for (i=0; i<=poly.n; i++){ \
        printf("a%u=\t%f\n", i, poly.a[i]); \
    }*/ \
    Poly1d_Derivative(&poly, &dpoly); \
    /*printf("legendre diff poly\n"); \
    for (i=0; i<=dpoly.n; i++){ \
        printf("a%u=\t%f\n", i, dpoly.a[i]); \
    }*/ \
    /* weight */ \
    double p_prime, x; /* derivative of Legendre polynomial */ \
    for (i=0; i<n_int; i++){ \
        x = x_int[i]; \
        p_prime = Poly1d_Value(x, &dpoly); \
        /*printf("i=%u\tx=\t%f\tp'=\t%f\n", i, x, p_prime);*/ \
        w_int[i] = 2./( (1-x*x)*p_prime*p_prime ); \
    } \
    /* free */ \
    Poly1d_Free(&poly); \
    Poly1d_Free(&dpoly); \
    return true; \
} \

#define INTEGRATE_GAUSS_LEGENDRE_VIA_XW__ALGORITHM(num_type, integer_type) \
{ \
    if (n_int == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
    if (x_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given x_int is NULL.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
    if (w_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given w_int is NULL.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
 \
    num_type x_range = x2 - x1, x_mod = x_range/2, x_mid = (x1 + x2)/2, x, res = 0; \
    integer_type i; \
    for (i=0; i<n_int; i++){ \
        x = x_mid + x_int[i] * x_mod; \
        res += func(x, other_param) * w_int[i] * x_mod; \
    } \
    return res; \
} \

#define INTEGRATE_GAUSS_LEGENDRE__ALGORITHM(num_type, Integrate_Gauss_Legendre_x, func_name_1, Integrate_Gauss_Legendre_w, func_name_2, Integrate_Gauss_Legendre_via_xw) \
{ \
    num_type *x_int=(num_type*)malloc(2*(uint64_t)n_int*sizeof(num_type)), *w_int=x_int+n_int; \
    if (x_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for x_int & w_int.", __func__, 2*n_int*sizeof(num_type)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        free(x_int); \
        return 0; \
    } \
    /* root */ \
    bool flag_x_int = Integrate_Gauss_Legendre_x(n_int, x_int); \
    if (!flag_x_int){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: error when preparing integrate points (x). See info from %hs.", __func__, func_name_1); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        free(x_int); \
        return 0; \
    } \
    bool flag_w_int = Integrate_Gauss_Legendre_w(n_int, x_int, w_int); \
    if (!flag_w_int){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: error when preparing integrate weights (w). See info from %hs.", __func__, func_name_2); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        free(x_int); \
        return 0; \
    } \
    num_type res = Integrate_Gauss_Legendre_via_xw(func, x1, x2, n_int, other_param, x_int, w_int); \
    /* free */ \
    free(x_int); \
    return res; \
} \

/* double */
bool Integrate_Gauss_Legendre_x(int32_t n_int_, double *x_int)
INTEGRATE_GAUSS_LEGENDRE_X__ALGORITHM(double, uint64_t, sqrt, LAPACKE_dsteqr)

bool Integrate_Gauss_Legendre_w(int32_t n_int, double *x_int, double *w_int)
INTEGRATE_GAUSS_LEGENDRE_W__ALGORITHM(uint64_t, Poly1d, Poly1d_Create, Special_Func_Legendre, Poly1d_Derivative, Poly1d_Value, Poly1d_Free)

double Integrate_Gauss_Legendre_via_xw(double func(double, void *), double x1, double x2, int32_t n_int, void *other_param, double *x_int, double *w_int)
INTEGRATE_GAUSS_LEGENDRE_VIA_XW__ALGORITHM(double, uint64_t)

double Integrate_Gauss_Legendre(double func(double, void *), double x1, double x2, int32_t n_int, void *other_param)
INTEGRATE_GAUSS_LEGENDRE__ALGORITHM(double, Integrate_Gauss_Legendre_x, "Integrate_Gauss_Legendre_x", Integrate_Gauss_Legendre_w, "Integrate_Gauss_Legendre_w", Integrate_Gauss_Legendre_via_xw)

/* float */
bool Integrate_Gauss_Legendre_x_f32(int32_t n_int_, float *x_int)
INTEGRATE_GAUSS_LEGENDRE_X__ALGORITHM(float, uint32_t, sqrtf, LAPACKE_ssteqr)

bool Integrate_Gauss_Legendre_w_f32(int32_t n_int, float *x_int, float *w_int)
INTEGRATE_GAUSS_LEGENDRE_W__ALGORITHM(uint32_t, Poly1d_f32, Poly1d_Create_f32, Special_Func_Legendre_f32, Poly1d_Derivative_f32, Poly1d_Value_f32, Poly1d_Free_f32)

float Integrate_Gauss_Legendre_via_xw_f32(float func(float, void *), float x1, float x2, int32_t n_int, void *other_param, float *x_int, float *w_int)
INTEGRATE_GAUSS_LEGENDRE_VIA_XW__ALGORITHM(float, uint32_t)

float Integrate_Gauss_Legendre_f32(float func(float, void *), float x1, float x2, int32_t n_int, void *other_param)
INTEGRATE_GAUSS_LEGENDRE__ALGORITHM(float, Integrate_Gauss_Legendre_x_f32, "Integrate_Gauss_Legendre_x_f32", Integrate_Gauss_Legendre_w_f32, "Integrate_Gauss_Legendre_w_f32", Integrate_Gauss_Legendre_via_xw_f32)