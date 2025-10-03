/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/fft-bluestein.c
*/
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>

#include"fft.h"
#include"../basic/basic.h"

static void print_cnum_array(uint64_t n, Cnum *arr)
{
    uint64_t i;
    for (i=0; i<n; i++){
        printf("\t%f + %f*I\n", arr[i].real, arr[i].imag);
    }
}

bool Fast_Fourier_Transform_Bluestein(uint64_t n, Cnum *arr, int fft_direction)
{
    if (n == 0){
        return false;
    }
    if (arr == NULL){
        return false;
    }
    if (fft_direction != MADD_FFT_FORWARD && fft_direction != MADD_FFT_INVERSE){
        return false;
    }

    int m = (uint64_t)1 << Log2_Ceil(2*n-1);
    size_t size_abc = 3 * m, size_wm = m;

    Cnum *a = (Cnum*)malloc(size_abc), *b, *c;
    if (a == NULL){
        return false;
    }
    b = a + m;
    c = b + m;
    /* wm & wn */
    Cnum *wm = (Cnum*)malloc(size_wm);
    if (wm == NULL){
        free(a);
        return false;
    }
    //printf("getting weight\n");
    Fast_Fourier_Transform_Weight(m, wm, MADD_FFT_FORWARD);
    //printf("got weight\n");

    uint64_t i;
    for (i=0; i<n; i++){
        double angle = fft_direction * _CONSTANT_PI * i * i / n;
        Cnum chirp = {.real = cos(angle), .imag = sin(angle)};
        a[i] = Cnum_Mul(arr[i], chirp);
        b[i] = Cnum_Conj(chirp);
        if (i) b[m - i] = b[i];
    }
    /* Madd_Set0_c64(uint64_t N, Cnum *array): Set the length N of array to be 0; c64 means the real and imag of complex are all 64-bit */
    Madd_Set0_c64(m - n, a + n);
    Madd_Set0_c64(m - 2 * n + 1, b + n);
    //printf("before a b fft\n");
    /*printf("A array:\n");
    print_cnum_array(m, a);
    printf("B array:\n");
    print_cnum_array(m, b);*/

    Fast_Fourier_Transform_Radix2_Core(m, a, wm);
    //printf("A array:\n");
    //print_cnum_array(m, a);
    Fast_Fourier_Transform_Radix2_Core(m, b, wm);
    //printf("after a b fft\n");
    //printf("B array:\n");
    //print_cnum_array(m, b);

    for (i=0; i<m; i++){
        Cnum temp = Cnum_Mul(a[i], b[i]);
        c[i] = Cnum_Conj(temp);
    }
    //printf("c array:\n");
    //print_cnum_array(m, c);
    Fast_Fourier_Transform_Radix2_Core(m, c, wm);
    //printf("before c fft\n");
    //printf("c array:\n");
    //print_cnum_array(m, c);

    for (i=0; i<n; i++){
        double angle =  fft_direction * _CONSTANT_PI * i * i / n;
        Cnum chirp = {.real = cos(angle), .imag = sin(angle)};
        Cnum c_temp = Cnum_Div_Real(Cnum_Conj(c[i]), m);
        arr[i] = Cnum_Mul(c_temp, chirp);
        if (fft_direction == MADD_FFT_INVERSE){
            arr[i] = Cnum_Div_Real(arr[i], n);
        }
    }
    //printf("after arr fft output\n");

    free(a);
    free(wm);
    return true;
}