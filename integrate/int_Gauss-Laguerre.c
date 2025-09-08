/* coding: utf-8 */
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<stdint.h>
#include<stdbool.h>
#define HAVE_LAPACK_CONFIG_H
#include<lapacke.h>

#include"../basic/basic.h"

#define INTEGRATE_GAUSS_LAGUERRE_XW__ALGORITHM(num_type, integer_type, LAPACKE_dsteqr) \
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
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (w_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given w_int is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    num_type *subdiag = (num_type*)malloc((uint64_t)(n_int-1)*sizeof(num_type)); \
    if (subdiag == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for subdiag.", __func__, (uint64_t)(n_int-1)*sizeof(num_type)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    num_type *eigenvectors = (num_type*)malloc((uint64_t)n_int*n_int*sizeof(num_type)); \
    if (subdiag == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for eigenvectors.", __func__, (uint64_t)n_int*n_int*sizeof(num_type)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        free(subdiag); \
        return false; \
    } \
    integer_type i; \
    x_int[0] = 1; \
    for (i=1; i<n_int; i++){ \
        x_int[i] = 2.*i + 1; \
        subdiag[i-1] = sqrt(i); \
    } \
 \
    char compz = 'I'; /* cal eigenvalues & eigenvectors */ \
    lapack_int info = LAPACKE_dsteqr(LAPACK_ROW_MAJOR, compz, n_int, \
                                     x_int, subdiag, eigenvectors, n_int); \
    if (info){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        if (info < 0){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: (source file: %hs)(line: %d) from LAPACKE_dsteqr: the %d-th argument had an illegal value.", __func__, __FILE__, __LINE__, -info); \
        }else{ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: (source file: %hs)(line: %d) from LAPACKE_dsteqr: the algorithm has failed to find all the eigenvalues in a total of 30*N iterations; %d elements of E have not converged to zero; on exit, D and E contain the elements of a symmetric tridiagonal matrix which is orthogonally similar to the original matrix.", __func__, __FILE__, __LINE__, info); \
        } \
        Madd_Error_Add(MADD_ERROR, error_info); \
        free(subdiag); \
        free(eigenvectors); \
        return false; \
    } \
 \
    for (i=0; i<n_int; i++){ \
        num_type temp = eigenvectors[i*n_int]; \
        w_int[i] = temp * temp; \
    } \
 \
    free(subdiag); \
    free(eigenvectors); \
    return true; \
} \

#define INTEGRATE_GAUSS_LAGUERRE_VIA_XW__ALGORITHM(num_type, integer_type) \
{ \
    if (n_int == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return 0; \
    } \
    if (x_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given x_int is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
    if (w_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given w_int is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
 \
    num_type s = 0, x, scale = 1/exp_index, exp_factor = exp(-exp_index * xmin); \
    integer_type i; \
    for (i=0; i<n_int; i++){ \
        x = xmin + x_int[i] * scale; \
        s += func(x, other_param) * w_int[i]; \
    } \
    return s * scale * exp_factor; \
} \

#define INTEGRATE_GAUSS_LAGUERRE__ALGORITHM(num_type, Integrate_Gauss_Laguerre_xw, Integrate_Gauss_Laguerre_via_xw, func_name) \
{ \
    if (n_int == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return 0; \
    } \
    size_t n2_size = 2*(uint64_t)n_int*sizeof(num_type); \
    num_type *x_int = (num_type*)malloc(n2_size), *w_int = x_int + n_int; \
    if (x_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes.", __func__, n2_size); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
 \
    bool flag_xw = Integrate_Gauss_Laguerre_xw(n_int, x_int, w_int); \
    if (!flag_xw){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: see error info from %hs.", __func__, func_name); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
    num_type s = Integrate_Gauss_Laguerre_via_xw(func, xmin, exp_index, n_int, other_param, x_int, w_int); \
 \
    free(x_int); \
    return s; \
} \

/* uint64_t & double */
bool Integrate_Gauss_Laguerre_xw(uint64_t n_int, double *x_int, double *w_int)
INTEGRATE_GAUSS_LAGUERRE_XW__ALGORITHM(double, uint64_t, LAPACKE_dsteqr)

double Integrate_Gauss_Laguerre_via_xw(double func(double, void*), double xmin, double exp_index,
                                       uint64_t n_int, void *other_param,
                                       double *x_int, double *w_int)
INTEGRATE_GAUSS_LAGUERRE_VIA_XW__ALGORITHM(double, uint64_t)

/*
suppose h(x) = func(x) * exp(-exp_index * x)
return \int_{xmin}^{$\infty$} h(x) dx
*/
double Integrate_Gauss_Laguerre(double func(double, void*), double xmin, double exp_index,
                                uint64_t n_int, void *other_param)
INTEGRATE_GAUSS_LAGUERRE__ALGORITHM(double, Integrate_Gauss_Laguerre_xw, Integrate_Gauss_Laguerre_via_xw, "Integrate_Gauss_Laguerre_xw")

/* uint32_t & float */
bool Integrate_Gauss_Laguerre_xw_f32(uint32_t n_int, float *x_int, float *w_int)
INTEGRATE_GAUSS_LAGUERRE_XW__ALGORITHM(float, uint32_t, LAPACKE_ssteqr)

float Integrate_Gauss_Laguerre_via_xw_f32(float func(float, void*), float xmin, float exp_index,
                                          uint32_t n_int, void *other_param,
                                          float *x_int, float *w_int)
INTEGRATE_GAUSS_LAGUERRE_VIA_XW__ALGORITHM(float, uint32_t)

/*
suppose h(x) = func(x) * exp(-exp_index * x)
return \int_{xmin}^{$\infty$} h(x) dx
*/
float Integrate_Gauss_Laguerre_f32(float func(float, void*), float xmin, float exp_index,
                                   uint32_t n_int, void *other_param)
INTEGRATE_GAUSS_LAGUERRE__ALGORITHM(float, Integrate_Gauss_Laguerre_xw_f32, Integrate_Gauss_Laguerre_via_xw_f32, "Integrate_Gauss_Laguerre_xw_f32")