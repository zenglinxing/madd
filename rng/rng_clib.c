/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_clib.c
*/
#include<stdlib.h>
#include"rng_clib.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

uint64_t Madd_RNG_Clib_seed=0, Madd_RNG_Clib_n_gen=0;

void RNG_Clib_Init(uint64_t seed)
{
    Madd_RNG_Clib_seed=seed;
    srand(seed);
}

#define RAND_CLIB__ALGORITHM(float_type) \
{ \
    Madd_RNG_Clib_n_gen ++; \
    return rand()/(float_type)RAND_MAX; \
} \

double Rand_Clib(void)
RAND_CLIB__ALGORITHM(double)

float Rand_Clib_f32(void)
RAND_CLIB__ALGORITHM(float)

long double Rand_Clib_fl(void)
RAND_CLIB__ALGORITHM(long double)

#ifdef ENABLE_QUADPRECISION
__float128 Rand_Clib_f128(void)
RAND_CLIB__ALGORITHM(__float128)
#endif /* ENABLE_QUADPRECISION */