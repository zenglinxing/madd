Kahan Summation
===

Kahan summation introduces a compensate part to cancel the loss of two float number addition. This method is important when sum up a large array.

Function
---

```C
// real number
double Kahan_Summation(uint64_t n, double *arr);
// complex number
Cnum Kahan_Summation_c(uint64_t n, Cnum *arr);
```

Example

```C
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include"madd.h"

int main(int argc, char *argv[])
{
    uint64_t n = 1000000, i, seed = 10;
    double *arr = (double*)malloc(n * sizeof(double));

    /* initialize the array */
    RNG_Param rng = RNG_Init(seed, RNG_XOSHIRO256SS);
    for (i=0; i<n; i++){
        arr[i] = Rand(&rng);
    }

    /* sum up the array by Kahan summation */
    double sum_kahan = Kahan_Summation(n, arr);
    printf("Kahan sum:\t%.10g\n", sum_kahan);

    free(arr);
    return 0;
}
```