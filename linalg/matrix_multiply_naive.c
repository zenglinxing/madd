/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/matrix_multiply_naive.c
*/
#include<stdint.h>
#include<stdbool.h>
#include"linalg.h"
#include"../basic/basic.h"

bool Matrix_Multiply_Naive(uint64_t m, uint64_t n, uint64_t l,
                           double *a, double *b, double *res)
{
    if (m == 0){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: m is 0.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (n == 0){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is 0.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (l == 0){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: l is 0.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (a == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix a is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (b == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix b is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }
    if (res == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix res is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return false;
    }

    uint64_t i, j, k;
    double *p1, *p2, *p3 = res;
    for (i=0, p1=a; i<m; i++, p1+=l){
        for (j=0; j<n; j++){
            double sum = 0;
            for (k=0, p2=b+j; k<l; k++, p2+=n){
                sum += p1[k] * *p2;
            }
            *p3 = sum;
            p3 ++;
        }
    }
    return true;
}