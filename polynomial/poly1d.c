/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./polynomial/poly1d.c
1-D polynomial
*/
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<stdint.h>
#include<wchar.h>

#include"poly1d.h"
#include"../basic/basic.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define POLY1D_CREATE__ALGORITHM(num_type, size_num_type, Poly1d) \
{ \
    size_t n_size = ((uint64_t)_n+n+1)*sizeof(num_type); \
    Poly1d poly = {.n=n, ._n=_n}; \
    poly.mem = (num_type*)malloc(n_size); \
    if (poly.mem == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes.", __func__, n_size); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return poly; \
    } \
    size_num_type i; \
    for (i=0; i<_n+n+1; i++){ \
        poly.mem[i] = 0; \
    } \
    poly.a = poly.mem + _n; \
    return poly; \
} \

Poly1d Poly1d_Create(uint64_t n,uint64_t _n)
POLY1D_CREATE__ALGORITHM(double, uint32_t, Poly1d)

Poly1d_f32 Poly1d_Create_f32(uint32_t n,uint32_t _n)
POLY1D_CREATE__ALGORITHM(float, uint32_t, Poly1d_f32)

Poly1d_fl Poly1d_Create_LD(uint64_t n,uint64_t _n)
POLY1D_CREATE__ALGORITHM(long double, uint64_t, Poly1d_fl)

#define POLY1D_INIT__ALGORITHM(num_type,Poly1d) \
{ \
    size_t n_size = ((uint64_t)_n+n+1)*sizeof(num_type); \
    Poly1d poly = {.n=n, ._n=_n}; \
    poly.mem=(num_type*)malloc(n_size); \
    if (poly.mem == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes.", __func__, n_size); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return poly; \
    } \
    poly.a=poly.mem+_n; \
    memcpy(poly.mem, a, n_size); \
    return poly; \
} \

Poly1d Poly1d_Init(uint64_t n, uint64_t _n, double *a)
POLY1D_INIT__ALGORITHM(double, Poly1d)

Poly1d_f32 Poly1d_Init_f32(uint32_t n, uint32_t _n, float *a)
POLY1D_INIT__ALGORITHM(float, Poly1d_f32)

Poly1d_fl Poly1d_Init_fl(uint64_t n, uint64_t _n, long double *a)
POLY1D_INIT__ALGORITHM(long double, Poly1d_fl)

#define POLY1D_FREE__ALGORITHM \
{ \
    free(poly->mem); \
} \

void Poly1d_Free(Poly1d *poly)
POLY1D_FREE__ALGORITHM

void Poly1d_Free_f32(Poly1d_f32 *poly)
POLY1D_FREE__ALGORITHM

void Poly1d_Free_fl(Poly1d_fl *poly)
POLY1D_FREE__ALGORITHM

#define POLY1D_VALUE__ALGORITHM(num_type, size_num_type, Poly1d) \
{ \
    register size_num_type i; \
    register num_type *a = poly->a+poly->n, sum = *a; \
    a--; \
    for (i=poly->n; i>0; i--,a--){ \
        sum = sum * x + *a; \
    } \
    if (poly->_n == 0) return sum; \
    a=poly->a-poly->_n; \
    register num_type _x=1/x, _sum = *a * _x; \
    a++; \
    for (i=poly->_n-1; i>0; i--,a++){ \
        _sum = (_sum + *a) * _x; \
    } \
    return sum+_sum; \
} \

double Poly1d_Value(double x, Poly1d *poly)
POLY1D_VALUE__ALGORITHM(double, uint64_t, Poly1d)

float Poly1d_Value_f32(float x, Poly1d_f32 *poly)
POLY1D_VALUE__ALGORITHM(float, uint32_t, Poly1d_f32)

long double Poly1d_Value_fl(long double x, Poly1d_fl *poly)
POLY1D_VALUE__ALGORITHM(long double, uint64_t, Poly1d_fl)

#define POLY1D_DERIVATIVE__ALGORITHM(num_type, size_num_type, Poly1d) \
{ \
    register size_num_type i,n,_n; \
    register num_type *a=poly->a+1, *da=dpoly->a; \
    n = (poly->n >= 1) ? poly->n-1 : 0; \
    for (i=0; i<=n; i++,a++,da++){ \
        *da = *a * (i+1); \
    } \
    if (poly->_n > 0){ \
        _n=poly->_n; \
        a=poly->a-1; \
        da=dpoly->a-1; \
        *da = 0; \
        da--; \
        for (i=0; i<_n; i++,a--,da--){ \
            *da = - *a * (i+1); \
        } \
    } \
} \

void Poly1d_Derivative(Poly1d *poly, Poly1d *dpoly)
POLY1D_DERIVATIVE__ALGORITHM(double, uint64_t, Poly1d)

void Poly1d_Derivative_f32(Poly1d_f32 *poly, Poly1d_f32 *dpoly)
POLY1D_DERIVATIVE__ALGORITHM(float, uint32_t, Poly1d_f32)

void Poly1d_Derivative_fl(Poly1d_fl *poly, Poly1d_fl *dpoly)
POLY1D_DERIVATIVE__ALGORITHM(long double, uint64_t, Poly1d_fl)

#define POLY1D_DERIVATIVE_N_ORDER_ALLOCATED__ALGORITHM(num_type, size_num_type, Poly1d) \
{ \
    size_num_type n_pos, n_neg; \
    int64_t i_order; \
    n_pos = (poly->n > n_order) ? poly->n - n_order : 0; \
    n_neg = (poly->_n) ? poly->_n + n_order : 0; \
    /* assign a, n, _n for dpoly */ \
    dpoly->a = dpoly->mem + n_neg; \
    dpoly->n = n_pos; \
    dpoly->_n = n_neg; \
    /* positive part */ \
    num_type rate=1; \
    for (i_order=1; i_order<n_order; i_order++){ \
        rate *= i_order + 1; \
    } \
    if (poly->n < n_order){ \
        dpoly->a[0] = 0; \
    }else{ \
        for (i_order=0; i_order<=poly->n; i_order++){ \
            dpoly->a[i_order] = poly->a[i_order+n_order] * rate; \
            rate = (rate * (i_order+n_order+1)) / (i_order + 1); \
        } \
    } \
    /* negative part */ \
    if (n_neg){ \
        for (i_order=-1, rate=1; -i_order<=n_order; i_order--){ \
            rate *= i_order; \
        } \
        for (i_order=-1; -i_order<=poly->_n; i_order--){ \
            dpoly->a[i_order-n_order] = poly->a[i_order] * rate; \
            rate = (rate * (i_order-n_order)) / i_order; \
        } \
    } \
} \

#define POLY1D_DERIVATIVE_N_ORDER__ALGORITHM(num_type, size_num_type, Poly1d, Poly1d_Derivative_N_order_Allocated) \
{ \
    size_num_type n_pos, n_neg; \
    int64_t i_order; \
    n_pos = (poly->n > n_order) ? poly->n - n_order : 0; \
    n_neg = (poly->_n) ? poly->_n + n_order : 0; \
    /* dpoly */ \
    Poly1d dpoly; \
    /*dpoly.n = n_pos; \
    dpoly._n = n_neg;*/ /* These will be assigned in Poly1d_Derivative_N_order_Allocated */ \
    size_t size_malloc = ((uint64_t)n_pos+n_neg+1)*sizeof(num_type); \
    if (poly->mem == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes.", __func__, size_malloc); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return dpoly; \
    } \
    dpoly.mem = (num_type*)malloc(size_malloc); \
    Poly1d_Derivative_N_order_Allocated(poly, n_order, &dpoly); \
    return dpoly; \
} \

/*
only need to allocate enough memory for dpoly.mem
dpoly.a will be automatically re-assigned here.
*/
void Poly1d_Derivative_N_order_Allocated(const Poly1d *poly, uint64_t n_order, Poly1d *dpoly)
POLY1D_DERIVATIVE_N_ORDER_ALLOCATED__ALGORITHM(double, uint64_t, Poly1d)

Poly1d Poly1d_Derivative_N_order(const Poly1d *poly, uint64_t n_order)
POLY1D_DERIVATIVE_N_ORDER__ALGORITHM(double, uint32_t, Poly1d, Poly1d_Derivative_N_order_Allocated)

void Poly1d_Derivative_N_order_f32_Allocated(const Poly1d_f32 *poly, uint32_t n_order, Poly1d_f32 *dpoly)
POLY1D_DERIVATIVE_N_ORDER_ALLOCATED__ALGORITHM(float, uint32_t, Poly1d_f32)

Poly1d_f32 Poly1d_Derivative_N_order_f32(const Poly1d_f32 *poly, uint32_t n_order)
POLY1D_DERIVATIVE_N_ORDER__ALGORITHM(float, uint32_t, Poly1d_f32, Poly1d_Derivative_N_order_f32_Allocated)

void Poly1d_Derivative_N_order_fl_Allocated(const Poly1d_fl *poly, uint64_t n_order, Poly1d_fl *dpoly)
POLY1D_DERIVATIVE_N_ORDER_ALLOCATED__ALGORITHM(long double, uint64_t, Poly1d_fl)

Poly1d_fl Poly1d_Derivative_N_order_fl(const Poly1d_fl *poly, uint64_t n_order)
POLY1D_DERIVATIVE_N_ORDER__ALGORITHM(long double, uint64_t, Poly1d_fl, Poly1d_Derivative_N_order_fl_Allocated)

#define POLY1D_INTEGRATE__ALGORITHM(num_type, size_num_type) \
{ \
    size_num_type i,n_1=poly->n+1; \
    num_type *a=poly->a+poly->n, *ia; \
    ipoly->a[0] = 0.; \
    ia=ipoly->a+poly->n+1; \
    for (i=poly->n+1; i>=1; i--,ia--,a--){ \
        *ia = *a / i; \
    } \
    a=poly->a-poly->_n; \
    ia=ipoly->a-poly->_n+1; \
    for (i=poly->_n-1; i>=1; i--,ia++,a++){ \
        *ia = - *a / i; \
    } \
    return (poly->_n>=1) ? poly->a[-1] : 0; /* log coefficient */ \
} \

double Poly1d_Integrate(Poly1d *poly, Poly1d *ipoly)
POLY1D_INTEGRATE__ALGORITHM(double, uint64_t)

float Poly1d_Integrate_f32(Poly1d_f32 *poly,Poly1d_f32 *ipoly)
POLY1D_INTEGRATE__ALGORITHM(float, uint32_t)

long double Poly1d_Integrate_fl(Poly1d_fl *poly, Poly1d_fl *ipoly)
POLY1D_INTEGRATE__ALGORITHM(long double, uint64_t)

Poly1d Poly1d_Integrate_N_order(Poly1d *poly, uint64_t n_order, double *log_coefficient)
{
    char has_pos = 1, has_neg = 1;
    uint64_t n_pos, n_neg;
    int64_t i_order, n_neg_order=n_order;
    n_neg_order = -n_neg_order;
    *log_coefficient = 0;
    /* positive order */
    if (poly->n == 0){
        if (poly->a[0] == 0){
            n_pos = 0;
            has_pos = 0;
        }else{
            n_pos = 1;
        }
    }else{
        n_pos = poly->n + n_order;
    }
    /* negative order */
    if (poly->_n < n_order){
        n_neg = 0;
        has_neg = 0;
    }else{
        n_neg = poly->_n - n_order;
    }
    /* integrated polynomial */
    Poly1d ipoly = Poly1d_Create(n_pos, n_neg);
    /* integrate positive order */
    double rate=1;
    if (has_pos){
        for (i_order=0; i_order<n_order; i_order++){
            rate /= i_order + 1;
        }
        for (i_order=0; i_order<=poly->n; i_order++){
            ipoly.a[i_order+n_order] = poly->a[i_order] * rate;
            rate = rate * (i_order + 1) / (double)(i_order + n_order + 1);
        }
    }
    if (has_neg){
        for (i_order=1, rate=1; i_order<n_order; i_order++){
            rate /= -i_order;
        }
        *log_coefficient = poly->a[n_neg_order] * rate;
        rate /= n_neg_order;
        for (i_order=1; i_order<poly->_n-n_order; i_order++){
            ipoly.a[-i_order] = poly->a[-i_order+n_neg_order] * rate;
            rate *= -i_order/(-i_order+n_neg_order);
        }
    }
    return ipoly;
}

#define POLY1D_NINTEGRATE__ALGORITHM(num_type,size_num_type,log,fabs) \
{ \
    size_num_type i,n_1=poly->n+1; \
    num_type *a, coefficient, sum_min=0, sum_max=0; \
    a=poly->a+poly->n; \
    for (i=poly->n+1; i>=1; i--,a--){ \
        coefficient = *a / i; \
        sum_min = (sum_min + coefficient) * xmin; \
        sum_max = (sum_max + coefficient) * xmax; \
    } \
    num_type _sum_min=0, _sum_max=0, _xmin=1/xmin, _xmax=1/xmax; \
    if (poly->_n){ \
        a=poly->a-poly->_n; \
        for (i=poly->_n-1; i>=1; i--,a++){ \
            coefficient = - *a / i; \
            _sum_min = (_sum_min + coefficient) * _xmin; \
            _sum_max = (_sum_max + coefficient) * _xmax; \
        } \
    } \
    num_type log_int, log_coefficient=(poly->_n>=1) ? poly->a[-1] : 0; /* log coefficient */ \
    log_int = (log_coefficient==0) ? 0 : log_coefficient*( log(fabs(xmax)) - log(fabs(xmin)) ); \
    return (sum_max-sum_min) + (_sum_max-_sum_min) + log_int; \
} \

double Poly1d_NIntegrate(Poly1d *poly, double xmin, double xmax)
POLY1D_NINTEGRATE__ALGORITHM(double, uint64_t, log, fabs)

float Poly1d_NIntegrate_f32(Poly1d_f32 *poly, float xmin, float xmax)
POLY1D_NINTEGRATE__ALGORITHM(float, uint32_t, logf, fabsf)

long double Poly1d_NIntegrate_fl(Poly1d_fl *poly, long double xmin, long double xmax)
POLY1D_NINTEGRATE__ALGORITHM(long double, uint64_t, logl, fabsl)

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
Poly1d_f128 Poly1d_Create_f128(uint64_t n, uint64_t _n)
POLY1D_CREATE__ALGORITHM(__float128, uint64_t, Poly1d_f128)

Poly1d_f128 Poly1d_Init_f128(uint64_t n, uint64_t _n, __float128 *a)
POLY1D_INIT__ALGORITHM(__float128, Poly1d_f128)

void Poly1d_Free_f128(Poly1d_f128 *poly)
POLY1D_FREE__ALGORITHM

__float128 Poly1d_Value_f128(__float128 x, Poly1d_f128 *poly)
POLY1D_VALUE__ALGORITHM(__float128, uint64_t, Poly1d_f128)

void Poly1d_Derivative_f128(Poly1d_f128 *poly, Poly1d_f128 *dpoly)
POLY1D_DERIVATIVE__ALGORITHM(__float128, uint64_t, Poly1d_f128)

void Poly1d_Derivative_N_order_f128_Allocated(const Poly1d_f128 *poly, uint64_t n_order, Poly1d_f128 *dpoly)
POLY1D_DERIVATIVE_N_ORDER_ALLOCATED__ALGORITHM(__float128, uint64_t, Poly1d_f128)

Poly1d_f128 Poly1d_Derivative_N_order_f128(const Poly1d_f128 *poly, uint64_t n_order)
POLY1D_DERIVATIVE_N_ORDER__ALGORITHM(__float128, uint64_t, Poly1d_f128, Poly1d_Derivative_N_order_f128_Allocated)

__float128 Poly1d_Integrate_f128(Poly1d_f128 *poly, Poly1d_f128 *ipoly)
POLY1D_INTEGRATE__ALGORITHM(__float128, uint64_t)

__float128 Poly1d_NIntegrate_f128(Poly1d_f128 *poly, __float128 xmin, __float128 xmax)
POLY1D_NINTEGRATE__ALGORITHM(__float128, uint64_t, logq, fabsq)
#endif /* ENABLE_QUADPRECISION */

/*
=======================================================================================
complex number
*/
#define POLY1D_CREATE_CNUM__ALGORITHM(num_type, size_num_type, Poly1d) \
{ \
    size_t n_size = ((uint64_t)_n+n+1)*sizeof(num_type); \
    num_type zero = {.real=0, .imag=0}; \
    Poly1d poly; \
    poly.n = n; \
    poly._n = _n; \
    poly.mem = (num_type*)malloc(n_size); \
    if (poly.mem == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes.", __func__, n_size); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return poly; \
    } \
    size_num_type i; \
    for (i=0; i<_n+n+1; i++){ \
        poly.mem[i] = zero; \
    } \
    poly.a = poly.mem + _n; \
    return poly; \
} \

Poly1d_c Poly1d_Create_c(uint64_t n, uint64_t _n)
POLY1D_CREATE_CNUM__ALGORITHM(Cnum, uint64_t, Poly1d_c)

Poly1d_c32 Poly1d_Create_c32(uint64_t n, uint64_t _n)
POLY1D_CREATE_CNUM__ALGORITHM(Cnum_f32, uint64_t, Poly1d_c32)

Poly1d_cl Poly1d_Create_LDC(uint64_t n, uint64_t _n)
POLY1D_CREATE_CNUM__ALGORITHM(Cnum_fl, uint64_t, Poly1d_cl)

Poly1d_c Poly1d_Init_c(uint64_t n, uint64_t _n, Cnum *a)
POLY1D_INIT__ALGORITHM(Cnum, Poly1d_c)

Poly1d_c32 Poly1d_Init_c32(uint64_t n, uint64_t _n, Cnum_f32 *a)
POLY1D_INIT__ALGORITHM(Cnum_f32, Poly1d_c32)

Poly1d_cl Poly1d_Init_cl(uint64_t n, uint64_t _n, Cnum_fl *a)
POLY1D_INIT__ALGORITHM(Cnum_fl, Poly1d_cl)

void Poly1d_Free_c(Poly1d_c *poly)
POLY1D_FREE__ALGORITHM

void Poly1d_Free_c32(Poly1d_c32 *poly)
POLY1D_FREE__ALGORITHM

void Poly1d_Free_cl(Poly1d_cl *poly)
POLY1D_FREE__ALGORITHM

#define POLY1D_VALUE_CNUM__ALGORITHM(num_type, size_num_type, Cnum_Add, Real_Div_Cnum, Cnum_Div, Cnum_Mul) \
{ \
    size_num_type i; \
    num_type *a = poly->a+poly->n, sum = *a; \
    a--; \
    for (i=poly->n; i>0; i--,a--){ \
        sum = Cnum_Add(Cnum_Mul(sum, x), *a); \
        /* sum = sum * x + *a; */ \
    } \
    if (poly->_n == 0) return sum; \
    a = poly->a - poly->_n; \
    num_type _x, _sum; \
    _x = Real_Div_Cnum(1, x); \
    _sum = Cnum_Div(*a, _x); \
    a++; \
    for (i=poly->_n-1; i>0; i--,a++){ \
        _sum = Cnum_Mul(Cnum_Add(_sum, *a), _x); \
        /* _sum = (_sum + *a) * _x; */ \
    } \
    sum = Cnum_Add(_sum, sum); \
    return sum; \
} \

Cnum Poly1d_Value_c(Cnum x, Poly1d_c *poly)
POLY1D_VALUE_CNUM__ALGORITHM(Cnum, uint64_t, Cnum_Add, Real_Div_Cnum, Cnum_Div, Cnum_Mul)

Cnum_f32 Poly1d_Value_c32(Cnum_f32 x, Poly1d_c32 *poly)
POLY1D_VALUE_CNUM__ALGORITHM(Cnum_f32, uint64_t, Cnum_Add_f32, Real_Div_Cnum_f32, Cnum_Div_f32, Cnum_Mul_f32)

Cnum_fl Poly1d_Value_cl(Cnum_fl x, Poly1d_cl *poly)
POLY1D_VALUE_CNUM__ALGORITHM(Cnum_fl, uint64_t, Cnum_Add_fl, Real_Div_Cnum_fl, Cnum_Div_fl, Cnum_Mul_fl)

#define POLY1D_DERIVATIVE_CNUM__ALGORITHM(num_type, size_num_type, Cnum_Mul_Real, Cnum_Value) \
{ \
    register size_num_type i,n,_n; \
    register num_type *a=poly->a+1, *da=dpoly->a; \
    n = (poly->n >= 1) ? poly->n-1 : 0; \
    for (i=0; i<n; i++,a++,da++){ \
        *da = Cnum_Mul_Real(*a, i+1); \
        /* *da = *a * (i+1); */ \
    } \
    if (poly->_n > 0){ \
        _n=poly->_n; \
        a=poly->a-1; \
        da=dpoly->a-1; \
        *da = Cnum_Value(0, 0); \
        /* *da = 0.; */ \
        da--; \
        for (i=0; i<_n; i++,a--,da--){ \
            *da = Cnum_Mul_Real(*a, -1.*(i+1)); \
            /* *da = - *a * (i+1); */ \
        } \
    } \
} \

void Poly1d_Derivative_c(Poly1d_c *poly, Poly1d_c *dpoly)
POLY1D_DERIVATIVE_CNUM__ALGORITHM(Cnum, uint64_t, Cnum_Mul_Real, Cnum_Value)

void Poly1d_Derivative_c32(Poly1d_c32 *poly, Poly1d_c32 *dpoly)
POLY1D_DERIVATIVE_CNUM__ALGORITHM(Cnum_f32, uint64_t, Cnum_Mul_Real_f32, Cnum_Value_f32)

void Poly1d_Derivative_cl(Poly1d_cl *poly, Poly1d_cl *dpoly)
POLY1D_DERIVATIVE_CNUM__ALGORITHM(Cnum_fl, uint64_t, Cnum_Mul_Real_fl, Cnum_Value_fl)

#define POLY1D_DERIVATIVE_N_ORDER_CNUM_ALLOCATED__ALGORITHM(num_type, size_num_type, real_num_type, Cnum_Value, Cnum_Mul_Real) \
{ \
    size_num_type n_pos, n_neg; \
    int64_t i_order; \
    n_pos = (poly->n > n_order) ? poly->n - n_order : 0; \
    n_neg = poly->_n + n_order; \
    /* assign a, n, _n for dpoly */ \
    dpoly->a = dpoly->mem + n_neg; \
    dpoly->n = n_pos; \
    dpoly->_n = n_neg; \
    /* positive part */ \
    real_num_type rate=1; \
    for (i_order=1; i_order<n_order; i_order++){ \
        rate *= i_order + 1; \
    } \
    if (poly->n < n_order){ \
        dpoly->a[0] = Cnum_Value(0, 0); \
    }else{ \
        for (i_order=0; i_order<=poly->n; i_order++){ \
            dpoly->a[i_order] = Cnum_Mul_Real(poly->a[i_order+n_order], rate); \
            rate = (rate * (i_order+n_order+1)) / (i_order + 1); \
        } \
    } \
    /* negative part */ \
    for (i_order=-1, rate=1; -i_order<=n_order; i_order--){ \
        rate *= i_order; \
    } \
    for (i_order=-1; -i_order<=poly->_n; i_order--){ \
        dpoly->a[i_order-n_order] = Cnum_Mul_Real(poly->a[i_order], rate); \
        rate = (rate * (i_order-n_order)) / i_order; \
    } \
} \

void Poly1d_Derivative_N_order_c_Allocated(const Poly1d_c *poly, uint64_t n_order, Poly1d_c *dpoly)
POLY1D_DERIVATIVE_N_ORDER_CNUM_ALLOCATED__ALGORITHM(Cnum, uint64_t, double, Cnum_Value, Cnum_Mul_Real)

Poly1d_c Poly1d_Derivative_N_order_c(const Poly1d_c *poly, uint64_t n_order)
POLY1D_DERIVATIVE_N_ORDER__ALGORITHM(Cnum, uint64_t, Poly1d_c, Poly1d_Derivative_N_order_c_Allocated)

void Poly1d_Derivative_N_order_c32_Allocated(const Poly1d_c32 *poly, uint64_t n_order, Poly1d_c32 *dpoly)
POLY1D_DERIVATIVE_N_ORDER_CNUM_ALLOCATED__ALGORITHM(Cnum_f32, uint64_t, float, Cnum_Value_f32, Cnum_Mul_Real_f32)

Poly1d_c32 Poly1d_Derivative_N_order_c32(const Poly1d_c32 *poly, uint64_t n_order)
POLY1D_DERIVATIVE_N_ORDER__ALGORITHM(Cnum_f32, uint64_t, Poly1d_c32, Poly1d_Derivative_N_order_c32_Allocated)

void Poly1d_Derivative_N_order_cl_Allocated(const Poly1d_cl *poly, uint64_t n_order, Poly1d_cl *dpoly)
POLY1D_DERIVATIVE_N_ORDER_CNUM_ALLOCATED__ALGORITHM(Cnum_fl, uint64_t, long double, Cnum_Value_fl, Cnum_Mul_Real_fl)

Poly1d_cl Poly1d_Derivative_N_order_cl(const Poly1d_cl *poly, uint64_t n_order)
POLY1D_DERIVATIVE_N_ORDER__ALGORITHM(Cnum_fl, uint64_t, Poly1d_cl, Poly1d_Derivative_N_order_cl_Allocated)

#define POLY1D_INTEGRATE_CNUM__ALGORITHM(num_type, size_num_type, Cnum_Value, Cnum_Div_Real) \
{ \
    size_num_type i,n_1=poly->n+1; \
    num_type *a=poly->a+poly->n, *ia, zero = Cnum_Value(0, 0); \
    ipoly->a[0] = zero; \
    /*ipoly->a[0] = 0.;*/ \
    ia=ipoly->a+poly->n+1; \
    for (i=poly->n+1; i>=1; i--,ia--,a--){ \
        *ia = Cnum_Div_Real(*a, i); \
        /* *ia = *a / i; */ \
    } \
    a=poly->a-poly->_n; \
    ia=ipoly->a-poly->_n+1; \
    for (i=poly->_n-1; i>=1; i--,ia++,a++){ \
        *ia = Cnum_Div_Real(*a, -1.*i); \
        /* *ia = - *a / i; */ \
    } \
    return (poly->_n>=1) ? poly->a[-1] : zero; /* log coefficient */ \
} \

Cnum Poly1d_Integrate_c(Poly1d_c *poly,Poly1d_c *ipoly)
POLY1D_INTEGRATE_CNUM__ALGORITHM(Cnum, uint64_t, Cnum_Value, Cnum_Div_Real)

Cnum_f32 Poly1d_Integrate_c32(Poly1d_c32 *poly, Poly1d_c32 *ipoly)
POLY1D_INTEGRATE_CNUM__ALGORITHM(Cnum_f32, uint64_t, Cnum_Value_f32, Cnum_Div_Real_f32)

Cnum_fl Poly1d_Integrate_cl(Poly1d_cl *poly, Poly1d_cl *ipoly)
POLY1D_INTEGRATE_CNUM__ALGORITHM(Cnum_fl, uint64_t, Cnum_Value_fl, Cnum_Div_Real_fl)

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __complex128 */
Poly1d_c128 Poly1d_Create_c128(uint64_t n, uint64_t _n)
POLY1D_CREATE_CNUM__ALGORITHM(Cnum_f128, uint64_t, Poly1d_c128)

Poly1d_c128 Poly1d_Init_c128(uint64_t n, uint64_t _n, Cnum_f128 *a)
POLY1D_INIT__ALGORITHM(Cnum_f128, Poly1d_c128)

void Poly1d_Free_c128(Poly1d_c128 *poly)
POLY1D_FREE__ALGORITHM

Cnum_f128 Poly1d_Value_c128(Cnum_f128 x, Poly1d_c128 *poly)
POLY1D_VALUE_CNUM__ALGORITHM(Cnum_f128, uint64_t, Cnum_Add_f128, Real_Div_Cnum_f128, Cnum_Div_f128, Cnum_Mul_f128)

void Poly1d_Derivative_c128(Poly1d_c128 *poly, Poly1d_c128 *dpoly)
POLY1D_DERIVATIVE_CNUM__ALGORITHM(Cnum_f128, uint64_t, Cnum_Mul_Real_f128, Cnum_Value_f128)

void Poly1d_Derivative_N_order_c128_Allocated(const Poly1d_c128 *poly, uint64_t n_order, Poly1d_c128 *dpoly)
POLY1D_DERIVATIVE_N_ORDER_CNUM_ALLOCATED__ALGORITHM(Cnum_f128, uint64_t, __float128, Cnum_Value_f128, Cnum_Mul_Real_f128)

Poly1d_c128 Poly1d_Derivative_N_order_c128(const Poly1d_c128 *poly, uint64_t n_order)
POLY1D_DERIVATIVE_N_ORDER__ALGORITHM(Cnum_f128, uint64_t, Poly1d_c128, Poly1d_Derivative_N_order_c128_Allocated)

Cnum_f128 Poly1d_Integrate_c128(Poly1d_c128 *poly,Poly1d_c128 *ipoly)
POLY1D_INTEGRATE_CNUM__ALGORITHM(Cnum_f128, uint64_t, Cnum_Value_f128, Cnum_Div_Real_f128)
#endif /* ENABLE_QUADPRECISION */
