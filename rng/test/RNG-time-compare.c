#include<stdio.h>
#include<time.h>
#include<stdint.h>
#include"madd.h"

uint64_t n=1e7;

int main(int argc, char *argv[])
{
    clock_t start, end;
    uint64_t i;
    double res, elapse;

    /* MT */
    RNG_MT_Param mt=RNG_MT_Init(10);
    start = clock();
    for (i=0; i<n; i++){
        res = Rand_MT(&mt);
    }
    end = clock();
    double elapse_mt=(double)(end-start)/CLOCKS_PER_SEC;
    printf("MT elapse:\t%f sec\n", elapse_mt);

    /* x86 */
#if defined(__x86_64__) || defined(_M_X64)
    start = clock();
    for (i=0; i<n; i++){
        res = Rand_x86();
    }
    end = clock();
    double elapse_x86 = (double)(end-start)/CLOCKS_PER_SEC;
    printf("x86 elapse:\t%f sec\n", elapse_x86);
#endif

    /* clib */
    RNG_Clib_Init(10);
    start = clock();
    for (i=0; i<n; i++){
        res = Rand_Clib();
    }
    end = clock();
    double elapse_clib = (double)(end-start)/CLOCKS_PER_SEC;
    printf("clib elapse:\t%f sec\n", elapse_clib);

    /* Xorshift64 */
    RNG_Xorshift64_Param rxp64=RNG_Xorshift64_Init(10);
    start = clock();
    for (i=0; i<n; i++){
        res = Rand_Xorshift64(&rxp64);
    }
    end = clock();
    double elapse_xorshift64 = (double)(end-start)/CLOCKS_PER_SEC;
    printf("Xorshift64 elapse:\t%f sec\n", elapse_xorshift64);

    /* Xorshift64s */
    RNG_Xorshift64_Param rxp64s=RNG_Xorshift64s_Init(10);
    start = clock();
    for (i=0; i<n; i++){
        res = Rand_Xorshift64s(&rxp64s);
    }
    end = clock();
    double elapse_xorshift64s = (double)(end-start)/CLOCKS_PER_SEC;
    printf("Xorshift64* elapse:\t%f sec\n", elapse_xorshift64s);

    /* Xorshift1024s */
    RNG_Xorshift1024_Param rxp1024s=RNG_Xorshift1024s_Init(10);
    start = clock();
    for (i=0; i<n; i++){
        res = Rand_Xorshift1024s(&rxp1024s);
    }
    end = clock();
    double elapse_xorshift1024s = (double)(end-start)/CLOCKS_PER_SEC;
    printf("Xorshift1024* elapse:\t%f sec\n", elapse_xorshift1024s);

    /* Xoshiro256ss */
    RNG_Xoshiro256_Param rxp256ss=RNG_Xoshiro256ss_Init(10);
    start = clock();
    for (i=0; i<n; i++){
        res = Rand_Xoshiro256ss(&rxp256ss);
    }
    end = clock();
    double elapse_xoshiro256ss = (double)(end-start)/CLOCKS_PER_SEC;
    printf("Xoshiro256** elapse:\t%f sec\n", elapse_xoshiro256ss);

    /* Xoshiro256p */
    RNG_Xoshiro256_Param rxp256p=RNG_Xoshiro256p_Init(10);
    start = clock();
    for (i=0; i<n; i++){
        res = Rand_Xoshiro256p(&rxp256p);
    }
    end = clock();
    double elapse_xoshiro256p = (double)(end-start)/CLOCKS_PER_SEC;
    printf("Xoshiro256+ elapse:\t%f sec\n", elapse_xoshiro256p);

    /* Xorwow */
    RNG_Xorwow_Param rxpwow=RNG_Xorwow_Init(10);
    start = clock();
    for (i=0; i<n; i++){
        res = Rand_Xorwow(&rxpwow);
    }
    end = clock();
    double elapse_xorwow = (double)(end-start)/CLOCKS_PER_SEC;
    printf("Xorwow elapse:\t%f sec\n", elapse_xorwow);
    
    return 0;
}