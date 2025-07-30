#include<stdio.h>
#include"madd.h"

#define N 10

int main(int argc, char *argv[])
{
    int i, n=N;
    double arr1[N], arr2[N];
    RNG_Clib_Init(10);
    for (i=0; i<n; i++){
        arr1[i] = Rand_Clib();
    }
    RNG_Clib_Init(10);
    for (i=0; i<n; i++){
        arr2[i] = Rand_Clib();
    }

    printf("array 1 & 2 compare:\n");
    for (i=0; i<n; i++){
        printf("%d\t%f\t%f\n", i, arr1[i], arr2[i]);
    }
    return 0;
}