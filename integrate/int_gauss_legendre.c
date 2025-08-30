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
#include<lapacke.h>
#include"integrate.h"
#include"../polynomial/poly1d.h"
#include"../special_func/special_func.h"
#include"../basic/basic.h"

bool Integrate_Gauss_Legendre_x(uint64_t n_int_, double *x_int)
{
    uint64_t n_int = n_int_, i, j, nn=(uint64_t)n_int*n_int;
    size_t nn_size = nn*sizeof(double);
    double *mat=(double*)malloc(nn*sizeof(double)), b;
    if (mat == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for matrix.", __func__, nn_size);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    mat[0] = 0;
    for (i=1; i<n_int; i++){
        b = i/sqrt(4.*i*i-1);
        mat[(i-1)*n_int + i] = mat[i*n_int + i-1] = b;
        mat[i*n_int + i] = 0;
    }
    /* eigen */
    char jobz = 'V', uplo = 'U';
    lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, jobz, uplo, n_int, mat, n_int, x_int);

    free(mat);

    if (info){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        if (info < 0){
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: (source file: %hs)(line: %d) from LAPACKE_dsyev: the %d-th argument had an illegal value.", __func__, __FILE__, __LINE__, -info);
        }else{
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: (source file: %hs)(line: %d) from LAPACKE_dsyev: the algorithm failed to converge; %d off-diagonal elements of an intermediate tridiagonal form did not converge to zero.", __func__, __FILE__, __LINE__, info);
        }
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    return true;
}

bool Integrate_Gauss_Legendre_w(uint64_t n_int, double *x_int, double *w_int)
{
    uint64_t i;
    Poly1d poly = Poly1d_Create(n_int, 0), dpoly = Poly1d_Create(n_int-1, 0);
    Special_Func_Legendre(&poly);
    /*printf("legendre poly\n");
    for (i=0; i<=poly.n; i++){
        printf("a%u=\t%f\n", i, poly.a[i]);
    }*/
    Poly1d_Derivative(&poly, &dpoly);
    /*printf("legendre diff poly\n");
    for (i=0; i<=dpoly.n; i++){
        printf("a%u=\t%f\n", i, dpoly.a[i]);
    }*/
    /* weight */
    double p_prime, x; /* derivative of Legendre polynomial */
    for (i=0; i<n_int; i++){
        x = x_int[i];
        p_prime = Poly1d_Value(x, &dpoly);
        //printf("i=%u\tx=\t%f\tp'=\t%f\n", i, x, p_prime);
        w_int[i] = 2./( (1-x*x)*p_prime*p_prime );
    }
    /* free */
    Poly1d_Free(&poly);
    Poly1d_Free(&dpoly);
    return true;
}

double Integrate_Gauss_Legendre_via_xw(double func(double, void *), double x1, double x2, uint64_t n_int, void *other_param, double *x_int, double *w_int)
{
    double x_range = x2 - x1, x_mod = x_range/2, x_mid = (x1 + x2)/2, x, res = 0;
    uint64_t i;
    for (i=0; i<n_int; i++){
        x = x_mid + x_int[i] * x_mod;
        res += func(x, other_param) * w_int[i] * x_mod;
    }
    return res;
}

double Integrate_Gauss_Legendre(double func(double, void *), double x1, double x2, uint64_t n_int, void *other_param)
{
    double *x_int=(double*)malloc(2*(uint64_t)n_int*sizeof(double)), *w_int=x_int+n_int;
    if (x_int == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for x_int & w_int.", __func__, 2*n_int*sizeof(double));
        Madd_Error_Add(MADD_ERROR, error_info);
        free(x_int);
        return 0;
    }
    /* root */
    bool flag_x_int = Integrate_Gauss_Legendre_x(n_int, x_int);
    if (!flag_x_int){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: error when preparing integrate points (x). See info from %hs.", __func__, "Integrate_Gauss_Legendre_x");
        Madd_Error_Add(MADD_ERROR, error_info);
        free(x_int);
        return 0;
    }
    bool flag_w_int = Integrate_Gauss_Legendre_w(n_int, x_int, w_int);
    if (!flag_w_int){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: error when preparing integrate weights (w). See info from %hs.", __func__, "Integrate_Gauss_Legendre_w");
        Madd_Error_Add(MADD_ERROR, error_info);
        free(x_int);
        return 0;
    }
    double res = Integrate_Gauss_Legendre_via_xw(func, x1, x2, n_int, other_param, x_int, w_int);
    /* free */
    free(x_int);
    return res;
}