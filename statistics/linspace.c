/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./statistics/linspace.c
*/
#include<float.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>

bool Linspace(double start, double end, uint64_t n, double *arr, bool include_end)
{
    if (n < 2){
        return false;
    }
    if (arr == NULL){
        return false;
    }
	uint64_t n_gap = (include_end) ? n - 1 : n;
    double diff = end - start, diff_abs = fabs(diff);
    double n_gap_lower = diff_abs / DBL_MIN;
    if (n >= n_gap_lower){
        double gap = diff / n_gap;
        for (uint64_t i=0; i<n; i++){
            arr[i] = start + i * gap;
        }
    }else{
        double n_gap_upper = DBL_MAX / diff_abs, f_gap = n_gap;
        for (uint64_t i=0; i<n; i++){
            double increment;
            if (i <= n_gap_upper){
                increment = (i * diff) / n_gap;
            }else{
                double rate = i / f_gap;
                increment = diff * rate;
            }
            arr[i] = start + increment;
        }
    }
    return true;
}