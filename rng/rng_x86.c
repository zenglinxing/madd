/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_x86.c
*/
#if defined(__x86_64__) || defined(_M_X64)

#include<stdint.h>
#include<immintrin.h>
#include"../basic/basic.h"
/*#include"../basic/constant.h""*/

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif

#include<stdint.h>

uint64_t Madd_RNG_x86_n_gen=0;

uint64_t RNG_x86_U64(void)
{
    Madd_RNG_x86_n_gen ++;
    unsigned long long val;
    int ret = _rdrand64_step(&val);
    if (ret){
        return val;
    }else{
        Madd_Error_Add(MADD_ERROR, L"RNG_x86: unable to generate random number by _rdrand64_step");
        return 0;
    }
}

uint32_t RNG_x86_U32(void)
{
    Madd_RNG_x86_n_gen ++;
    uint32_t val;
    int ret=_rdrand32_step(&val);
    if (ret){
        return val;
    }else{
        Madd_Error_Add(MADD_ERROR, L"RNG_x86_f32: unable to generate random number by _rdrand32_step");
        return 0;
    }
}

double Rand_x86(void)
{
    uint64_t val = RNG_x86_U64();
    return val / (double)BIN64;
}

float Rand_x86_f32(void)
{
    uint32_t val = RNG_x86_U32();
    return val / (float)BIN32;
}

long double Rand_x86_fl(void)
{
    uint64_t val = RNG_x86_U64();
    return val / (long double)BIN64;
}

#ifdef ENABLE_QUADPRECISION
__float128 Rand_x86_f128(void)
{
    uint64_t val = RNG_x86_U64();
    return val / (__float128)BIN64;
}
#endif

#endif