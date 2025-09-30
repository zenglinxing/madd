/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/fft.cuh
*/
#ifndef FFT_CUH
#define FFT_CUH

#include<stdint.h>
#include<stdbool.h>

#include"../basic/cnum.h"

bool Fast_Fourier_Transform_cuda(int n, Cnum *arr, int fft_direction);
bool Fast_Fourier_Transform_cuda_c32(int n, Cnum32 *arr, int fft_direction);

void Madd_cufftPlan1d_error(int ret_plan, const char *func_name);
void Madd_cufftExec_error(int ret, const char *func_name, const char *func_exec_name);

#endif /* FFT_CUH */