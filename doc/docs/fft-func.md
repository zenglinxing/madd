FFT Functions
===

FFT Directions
---

The following macros is defined to determine the FFT directions.

```C
#define MADD_FFT_FORWARD 1
#define MADD_FFT_INVERSE -1
```

FFT Function
---

The following function is defined for FFT.
<!--Do note that the length of `arr` should be the least number greater than `n` that is a power of 2, even if your `n` is not a power of 2.-->
`fft_direction` should be either `MADD_FFT_FORWARD` or `MADD_FFT_INVERSE`.

```C
void Fast_Fourier_Transform(uint64_t n, Cnum *arr, int fft_direction);
```

Example
---

```C
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<madd/madd.h>

int main(int argc, char *argv[])
{
    uint64_t n = 5, i;
    Cnum *arr = (Cnum*)malloc(n*sizeof(Cnum));
    for (i=0; i<n; i++){
        arr[i].real = (i < 5) ? i + 1 : 0;
        printf("[%llu]\t%f + %fi\n", arr[i].real, arr[i].imag);
    }
    printf("\n");

    Fast_Fourier_Transform(n, arr, MADD_FFT_FORWARD);

    for (i=0; i<n; i++){
        printf("[%llu]\t%f + %fi\n", arr[i].real, arr[i].imag);
    }

    free(arr);
    return 0;
}
```