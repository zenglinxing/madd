/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/dct1-naive.c
Discrete Cosine Transform (DCT-I)
*/
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>

#include"fft.h"
#include"../basic/basic.h"

bool Discrete_Cosine_Transform_1_Naive(uint64_t n, double *arr)
{
    if (n < 2){
        return false;
    }
    if (arr == NULL){
        return false;
    }

    uint64_t n1 = n - 1, N = 2 * n1;
    double angle_base = 2 * (double)_CONSTANT_PI / ((double)N);
    double *w = (double*)malloc(sizeof(double) * (N+n)), *narr=w+N;
    if (w == NULL){
        return false;
    }
    for (uint64_t i = 0; i < N; i++){
        w[i] = cos(angle_base * i);
    }

    double scale = sqrt((double)2 / n1);
    for (uint64_t k = 0; k < n; k++){
        double sum = 0;
        uint64_t index = 0;
        for (uint64_t i = 0; i < n; i++, index=(index+k)%N){
            sum += arr[i] * w[index];
        }
        narr[k] = sum;
        if (k == 0 || k == n1){
            narr[k] *= 0.5 * scale;
        }else{
            narr[k] *= scale;
        }
    }

    memcpy(arr, narr, sizeof(double)*n);
    free(w);
    return true;
}