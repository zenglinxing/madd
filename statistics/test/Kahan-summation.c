/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    uint64_t n = 1000000, i, seed = 10;
    double *arr = (double*)malloc(n * sizeof(double));
    double sum_tradition=0, sum_kahan;

    RNG_Param rng = RNG_Init(seed, 0);
    for (i=0; i<n; i++){
        arr[i] = Rand(&rng);
        sum_tradition += arr[i];
    }
    sum_kahan = Kahan_Summation(n, arr);

    printf("traditional sum:\t%.10g\n", sum_tradition);
    printf("Kahan sum:\t%.10g\n", sum_kahan);

    free(arr);
    return 0;
}