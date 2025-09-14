/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./integrate/int_Clenshaw_Curtis.c
*/
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>

#include"integrate.h"
#include"../basic/basic.h"
#include"../fft/fft.h"

bool Integrate_Clenshaw_Curtis_x(double *x_int, uint64_t n_int)
{
    if (n_int == 0){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__);
        Madd_Error_Add(MADD_WARNING, error_info);
        return false;
    }
    if (x_int == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given x_int is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (n_int == 1){
        x_int[0] = 0;
        return true;
    }
    
    const double rate = _CONSTANT_PI / (2. * n_int);
    for (uint64_t i = 0; i < n_int; i++){
        x_int[i] = -cos( ((i<<1)+1) * rate );
    }
    return true;
}

bool Integrate_Clenshaw_Curtis_w(double *w_int, uint64_t n_int)
{
    if (n_int == 0){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__);
        Madd_Error_Add(MADD_WARNING, error_info);
        return false;
    }
    if (w_int == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given w_int is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    
    if (n_int == 1){
        w_int[0] = 2;
        return true;
    }
    
    uint64_t i;
    for (i=0; i<n_int; i++){
        if (i==0){
            w_int[i] = sqrt(2);
        }else if (i==1){
            w_int[i] = 0;
        }else{
            w_int[i] = (1 + ((i &0b1) ? -1 : 1)) / (1 - (double)i*i);
        }
    }
    Discrete_Cosine_Transform_2(n_int, w_int);
    
    return true;
}

double Integrate_Clenshaw_Curtis_via_xw(double func(double, void *), double xmin, double xmax,
                                        uint64_t n_int, void *other_param,
                                        double *x_int, double *w_int)
{
    if (n_int == 0){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__);
        Madd_Error_Add(MADD_WARNING, error_info);
        return 0;
    }
    if (x_int == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given x_int is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return 0;
    }
    if (w_int == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given w_int is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return 0;
    }
    
    double x_mod = (xmax - xmin) / 2;
    double x_mid = (xmax + xmin) / 2;
    
    double s = 0;
    for (uint64_t i = 0; i < n_int; i++){
        s += w_int[i] * func(x_mod * x_int[i] + x_mid, other_param);
    }
    s *= x_mod;
    
    return s;
}

double Integrate_Clenshaw_Curtis(double func(double, void *), double xmin, double xmax,
                                 uint64_t n_int, void *other_param)
{
    if (n_int == 0){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__);
        Madd_Error_Add(MADD_WARNING, error_info);
        return 0;
    }
    
    double *x_int = (double*)malloc(n_int * sizeof(double));
    if (x_int == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to allocate %llu bytes for x_int.", __func__, n_int * sizeof(double));
        Madd_Error_Add(MADD_ERROR, error_info);
        return 0;
    }
    Integrate_Clenshaw_Curtis_x(x_int, n_int);

    double *w_int = (double*)malloc(n_int * sizeof(double));
    if (w_int == NULL){
        free(x_int);
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to allocate %llu bytes for w_int.", __func__, n_int * sizeof(double));
        Madd_Error_Add(MADD_ERROR, error_info);
        return 0;
    }
    bool flag_get_w = Integrate_Clenshaw_Curtis_w(w_int, n_int);
    if (!flag_get_w){
        free(x_int);
        free(w_int);
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: see error info from %hs.", __func__, "Integrate_Clenshaw_Curtis_w");
        Madd_Error_Add(MADD_ERROR, error_info);
        return 0;
    }

    double res = Integrate_Clenshaw_Curtis_via_xw(func, xmin, xmax, n_int, other_param, x_int, w_int);

    free(x_int);
    free(w_int);
    return res;
}