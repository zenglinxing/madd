/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_param.c
*/
#include<stdint.h>
#include"rng.h"
#include"../basic/basic.h"

RNG_Param RNG_Init(uint64_t seed, uint32_t rng_type)
{
    RNG_Param rng={.rng_type=rng_type};
    switch (rng_type){
        case RNG_XOSHIRO256SS:
            rng.rng.rx256 = RNG_Xoshiro256ss_Init(seed);
            rng.rand_max = BIN64;
            rng.ru32 = NULL;
            rng.ru64 = (RNG_U64_t)RNG_Xoshiro256ss_U64;
            rng.rand = (Rand_t)Rand_Xoshiro256ss;
            rng.rand32 = (Rand_f32_t)Rand_Xoshiro256ss_f32;
            rng.randl = (Rand_fl_t)Rand_Xoshiro256ss_fl;
#ifdef ENABLE_QUADPRECISION
            rng.rand128 = (Rand_f128_t)Rand_Xoshiro256ss_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_MT:
            rng.rng.mt = RNG_MT_Init(seed);
            rng.rand_max = BIN64;
            rng.ru32 = NULL;
            rng.ru64 = (RNG_U64_t)RNG_MT_U64;
            rng.rand = (Rand_t)Rand_MT;
            rng.rand32 = (Rand_f32_t)Rand_MT_f32;
            rng.randl = (Rand_fl_t)Rand_MT_fl;
#ifdef ENABLE_QUADPRECISION
            rng.rand128 = (Rand_f128_t)Rand_MT_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_CLIB:
            RNG_Clib_Init(seed);
            rng.rand_max = BIN64;
            rng.ru32 = NULL;
            rng.ru64 = (RNG_U64_t)RNG_Clib_U64;
            rng.rand = (Rand_t)Rand_Clib;
            rng.rand32 = (Rand_f32_t)Rand_Clib_f32;
            rng.randl = (Rand_fl_t)Rand_Clib_fl;
#ifdef ENABLE_QUADPRECISION
            rng.rand128 = (Rand_f128_t)Rand_Clib_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_XORSHIFT64:
            rng.rng.rx64 = RNG_Xorshift64_Init(seed);
            rng.rand_max = BIN64;
            rng.ru32 = NULL;
            rng.ru64 = (RNG_U64_t)RNG_Xorshift64_U64;
            rng.rand = (Rand_t)Rand_Xorshift64;
            rng.rand32 = (Rand_f32_t)Rand_Xorshift64_f32;
            rng.randl = (Rand_fl_t)Rand_Xorshift64_fl;
#ifdef ENABLE_QUADPRECISION
            rng.rand128 = (Rand_f128_t)Rand_Xorshift64_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_XORSHIFT64S:
            rng.rng.rx64 = RNG_Xorshift64s_Init(seed);
            rng.rand_max = BIN64;
            rng.ru32 = NULL;
            rng.ru64 = (RNG_U64_t)RNG_Xorshift64s_U64;
            rng.rand = (Rand_t)Rand_Xorshift64s;
            rng.rand32 = (Rand_f32_t)Rand_Xorshift64s_f32;
            rng.randl = (Rand_fl_t)Rand_Xorshift64s_fl;
#ifdef ENABLE_QUADPRECISION
            rng.rand128 = (Rand_f128_t)Rand_Xorshift64s_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_XORSHIFT1024S:
            rng.rng.rx1024 = RNG_Xorshift1024s_Init(seed);
            rng.rand_max = BIN64;
            rng.ru32 = NULL;
            rng.ru64 = (RNG_U64_t)RNG_Xorshift1024s_U64;
            rng.rand = (Rand_t)Rand_Xorshift1024s;
            rng.rand32 = (Rand_f32_t)Rand_Xorshift1024s_f32;
            rng.randl = (Rand_fl_t)Rand_Xorshift1024s_fl;
#ifdef ENABLE_QUADPRECISION
            rng.rand128 = (Rand_f128_t)Rand_Xorshift1024s_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_XOSHIRO256P:
            rng.rng.rx256 = RNG_Xoshiro256p_Init(seed);
            rng.rand_max = BIN64;
            rng.ru32 = NULL;
            rng.ru64 = (RNG_U64_t)RNG_Xoshiro256p_U64;
            rng.rand = (Rand_t)Rand_Xoshiro256p;
            rng.rand32 = (Rand_f32_t)Rand_Xoshiro256p_f32;
            rng.randl = (Rand_fl_t)Rand_Xoshiro256p_fl;
#ifdef ENABLE_QUADPRECISION
            rng.rand128 = (Rand_f128_t)Rand_Xoshiro256p_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_XORWOW:
            rng.rng.rxw = RNG_Xorwow_Init(seed);
            rng.rand_max = BIN32;
            rng.ru32 = (RNG_U32_t)RNG_Xorwow_U32;
            rng.ru64 = NULL;
            rng.rand = (Rand_t)Rand_Xorwow;
            rng.rand32 = (Rand_f32_t)Rand_Xorwow_f32;
            rng.randl = (Rand_fl_t)Rand_Xorwow_fl;
#ifdef ENABLE_QUADPRECISION
            rng.rand128 = (Rand_f128_t)Rand_Xorwow_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_X86:
            rng.rand_max = BIN64;
            rng.ru32 = NULL;
            rng.ru64 = (RNG_U64_t)RNG_x86_U64;
            rng.rand = (Rand_t)Rand_x86;
            rng.rand32 = (Rand_f32_t)Rand_x86_f32;
            rng.randl = (Rand_fl_t)Rand_x86_fl;
#ifdef ENABLE_QUADPRECISION
            rng.rand128 = (Rand_f128_t)Rand_x86_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        default:
            Madd_Error_Add(MADD_ERROR, L"RNG_Init: unknown rng_type input.");
    }
    return rng;
}

uint64_t Rand_Uint(RNG_Param *rng)
{
    if (rng->ru32 == NULL){
        return rng->ru64(&rng->rng);
    }else{
        return rng->ru32(&rng->rng);
    }
}

double Rand(RNG_Param *rng)
{
    return rng->rand(&rng->rng);
}

float Rand_f32(RNG_Param *rng)
{
    return rng->rand32(&rng->rng);
}

long double Rand_fl(RNG_Param *rng)
{
    return rng->randl(&rng->rng);
}

#ifdef ENABLE_QUADPRECISION
float Rand_f128(RNG_Param *rng)
{
    return rng->rand128(&rng->rng);
}
#endif /* ENABLE_QUADPRECISION */