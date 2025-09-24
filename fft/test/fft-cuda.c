// coding: utf-8
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include"madd.h"

double tolerence = 1e-5;

int main(int argc, char *argv[])
{
    uint64_t n = 8, i;
    if (argc >= 2){
        n = atoi(argv[1]);
    }
    Cnum *arr = (Cnum*)malloc(n*sizeof(Cnum)), *origin = (Cnum*)malloc(n*sizeof(Cnum));
    for (i=0; i<n; i++){
        arr[i].real = origin[i].real = i+1;
        arr[i].imag = origin[i].imag = 0;
    }

    Fast_Fourier_Transform_cuda(n, arr, MADD_FFT_FORWARD);
    printf("after fft\n");
    for (i=0; i<n; i++){
        printf("i=%llu\t%f+%f*I\n", i, arr[i].real, arr[i].imag);
    }

    Fast_Fourier_Transform_cuda(n, arr, MADD_FFT_INVERSE);

    bool flag_difference = false;
    printf("after ifft\n");
    for (i=0; i<n; i++){
        printf("i=%llu\t%f+%f*I", i, arr[i].real, arr[i].imag);
        if (fabs(arr[i].real - origin[i].real) > tolerence || fabs(arr[i].imag - origin[i].imag) > tolerence){
            printf("(difference)");
            flag_difference = true;
        }
        printf("\n");
    }

    free(arr);
    free(origin);

    if (flag_difference){
        printf("** test failed **\n");
        exit(EXIT_FAILURE);
    }else{
        printf("test success\n");
    }
    return 0;
}