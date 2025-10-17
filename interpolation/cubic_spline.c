/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./interpolation/Lagrange.c
*/
#include<stdlib.h>
#include<string.h>

#include"interpolation.h"
#include"../basic/basic.h"
#include"../linalg/linalg.h"
#include"../sort/sort.h"

bool Interpolation_Cubic_Spline_Init(uint64_t n, const double *x, const double *y, Interpolation_Cubic_Spline_Param *icsp)
{
    if (n < 2){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n < 2.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (x == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: x is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (y == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: y is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (icsp == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: icsp is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }

    uint64_t n1 = n - 1, i;
    icsp->n = n;
    icsp->x = (double*)malloc(((uint64_t)n1 * 4 + n) * sizeof(double));
    if (icsp->x == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for icsp.", __func__, ((uint64_t)n1 * 4 + n) * sizeof(double));
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    icsp->a = icsp->x + n;
    icsp->b = icsp->a + n1;
    icsp->c = icsp->b + n1;
    icsp->d = icsp->c + n1;
    memcpy(icsp->x, x, (uint64_t)n * sizeof(double));
    Sort(n, sizeof(double), icsp->x, Sort_Compare_Ascending_f64, NULL);

    double *delta = (double*)malloc((n1*4 + n*3) * sizeof(double)), *Delta = delta + n1, *vec = Delta + n1, *lower = vec + n, *diag = lower + n1, *upper = diag + n;
    if (delta == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for delta & Delta & vec & lower & diag & upper.", __func__, (n1*4 + n*3) * sizeof(double));
        Madd_Error_Add(MADD_ERROR, error_info);
        free(icsp->x);
        return false;
    }
    /* get *a, *delta, *Delta */
    for (i=0; i<n1; i++){
        icsp->a[i] = y[i];
        delta[i] = icsp->x[i+1] - icsp->x[i];
        Delta[i] = y[i+1] - y[i];
    }
    /* get tridiagnal matrix param and *vec */
    vec[0] = vec[n1] = upper[0] = lower[n-2] = 0;
    diag[0] = diag[n1] = 1;
    for (i=1; i<n1; i++){
        lower[i-1] = delta[i-1];
        diag[i] = 2*delta[i-1] + delta[i];
        upper[i] = delta[i];
        vec[i] = 3*(Delta[i]/delta[i] - Delta[i-1]/delta[i-1]);
    }
    /* solve *c */
    bool ret_let = Linear_Equations_Tridiagonal(n, lower, diag, upper, 1, vec);
    if (!ret_let){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: see info from %hs.", __func__, "Linear_Equations_Tridiagonal");
        Madd_Error_Add(MADD_ERROR, error_info);
        free(icsp->x);
        free(delta);
        return false;
    }
    memcpy(icsp->c, vec, ((uint64_t)n1*sizeof(double)));
    /* *d and *b */
    for (i=0; i<n1; i++){
        icsp->d[i] = (vec[i+1]-vec[i])/(3*delta[i]);
        icsp->b[i] = Delta[i]/delta[i] - delta[i]*(2*vec[i]+vec[i+1])/3;
    }

    free(delta);
    return true;
}