/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_param.c
*/
#include<stdio.h>
#include<stdint.h>
#include"rng.h"
#include"../basic/basic.h"

void RNG_Init_Pointer(uint32_t rng_type, RNG_Param *rng)
{
    switch (rng_type){
        case RNG_XOSHIRO256SS:
            rng->rand_max = BIN64;
            rng->ru32 = NULL;
            rng->ru64 = (RNG_U64_t)RNG_Xoshiro256ss_U64;
            rng->rand = (Rand_t)Rand_Xoshiro256ss;
            rng->rand32 = (Rand_f32_t)Rand_Xoshiro256ss_f32;
            rng->randl = (Rand_fl_t)Rand_Xoshiro256ss_fl;
#ifdef ENABLE_QUADPRECISION
            rng->rand128 = (Rand_f128_t)Rand_Xoshiro256ss_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_MT:
            rng->rand_max = BIN64;
            rng->ru32 = NULL;
            rng->ru64 = (RNG_U64_t)RNG_MT_U64;
            rng->rand = (Rand_t)Rand_MT;
            rng->rand32 = (Rand_f32_t)Rand_MT_f32;
            rng->randl = (Rand_fl_t)Rand_MT_fl;
#ifdef ENABLE_QUADPRECISION
            rng->rand128 = (Rand_f128_t)Rand_MT_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_CLIB:
            rng->rand_max = RAND_MAX;
            rng->ru32 = NULL;
            rng->ru64 = (RNG_U64_t)RNG_Clib_U64;
            rng->rand = (Rand_t)Rand_Clib;
            rng->rand32 = (Rand_f32_t)Rand_Clib_f32;
            rng->randl = (Rand_fl_t)Rand_Clib_fl;
#ifdef ENABLE_QUADPRECISION
            rng->rand128 = (Rand_f128_t)Rand_Clib_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_XORSHIFT64:
            rng->rand_max = BIN64;
            rng->ru32 = NULL;
            rng->ru64 = (RNG_U64_t)RNG_Xorshift64_U64;
            rng->rand = (Rand_t)Rand_Xorshift64;
            rng->rand32 = (Rand_f32_t)Rand_Xorshift64_f32;
            rng->randl = (Rand_fl_t)Rand_Xorshift64_fl;
#ifdef ENABLE_QUADPRECISION
            rng->rand128 = (Rand_f128_t)Rand_Xorshift64_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_XORSHIFT64S:
            rng->rand_max = BIN64;
            rng->ru32 = NULL;
            rng->ru64 = (RNG_U64_t)RNG_Xorshift64s_U64;
            rng->rand = (Rand_t)Rand_Xorshift64s;
            rng->rand32 = (Rand_f32_t)Rand_Xorshift64s_f32;
            rng->randl = (Rand_fl_t)Rand_Xorshift64s_fl;
#ifdef ENABLE_QUADPRECISION
            rng->rand128 = (Rand_f128_t)Rand_Xorshift64s_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_XORSHIFT1024S:
            rng->rand_max = BIN64;
            rng->ru32 = NULL;
            rng->ru64 = (RNG_U64_t)RNG_Xorshift1024s_U64;
            rng->rand = (Rand_t)Rand_Xorshift1024s;
            rng->rand32 = (Rand_f32_t)Rand_Xorshift1024s_f32;
            rng->randl = (Rand_fl_t)Rand_Xorshift1024s_fl;
#ifdef ENABLE_QUADPRECISION
            rng->rand128 = (Rand_f128_t)Rand_Xorshift1024s_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_XOSHIRO256P:
            rng->rand_max = BIN64;
            rng->ru32 = NULL;
            rng->ru64 = (RNG_U64_t)RNG_Xoshiro256p_U64;
            rng->rand = (Rand_t)Rand_Xoshiro256p;
            rng->rand32 = (Rand_f32_t)Rand_Xoshiro256p_f32;
            rng->randl = (Rand_fl_t)Rand_Xoshiro256p_fl;
#ifdef ENABLE_QUADPRECISION
            rng->rand128 = (Rand_f128_t)Rand_Xoshiro256p_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_XORWOW:
            rng->rand_max = BIN32;
            rng->ru32 = (RNG_U32_t)RNG_Xorwow_U32;
            rng->ru64 = NULL;
            rng->rand = (Rand_t)Rand_Xorwow;
            rng->rand32 = (Rand_f32_t)Rand_Xorwow_f32;
            rng->randl = (Rand_fl_t)Rand_Xorwow_fl;
#ifdef ENABLE_QUADPRECISION
            rng->rand128 = (Rand_f128_t)Rand_Xorwow_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
        case RNG_X86:
#if defined(__x86_64__) || defined(_M_X64)
            rng->rand_max = BIN64;
            rng->ru32 = NULL;
            rng->ru64 = (RNG_U64_t)RNG_x86_U64;
            rng->rand = (Rand_t)Rand_x86_param;
            rng->rand32 = (Rand_f32_t)Rand_x86_param_f32;
            rng->randl = (Rand_fl_t)Rand_x86_param_fl;
#ifdef ENABLE_QUADPRECISION
            rng->rand128 = (Rand_f128_t)Rand_x86_param_f128;
#endif /* ENABLE_QUADPRECISION */
            break;
#else
            Madd_Error_Add(MADD_ERROR, L"RNG_Init_Pointer: the rng type x86 is not supported on this platform.");
#endif
        default:
            Madd_Error_Add(MADD_ERROR, L"RNG_Init_Pointer: unknown rng_type input.");
    }
}

RNG_Param RNG_Init(uint64_t seed, uint32_t rng_type)
{
    RNG_Param rng={.rng_type=rng_type};
    switch (rng_type){
        case RNG_XOSHIRO256SS:
            rng.rng.rx256 = RNG_Xoshiro256ss_Init(seed);
            break;
        case RNG_MT:
            rng.rng.mt = RNG_MT_Init(seed);
            break;
        case RNG_CLIB:
            RNG_Clib_Init(seed);
            break;
        case RNG_XORSHIFT64:
            rng.rng.rx64 = RNG_Xorshift64_Init(seed);
            break;
        case RNG_XORSHIFT64S:
            rng.rng.rx64 = RNG_Xorshift64s_Init(seed);
            break;
        case RNG_XORSHIFT1024S:
            rng.rng.rx1024 = RNG_Xorshift1024s_Init(seed);
            break;
        case RNG_XOSHIRO256P:
            rng.rng.rx256 = RNG_Xoshiro256p_Init(seed);
            break;
        case RNG_XORWOW:
            rng.rng.rxw = RNG_Xorwow_Init(seed);
            break;
#if defined(__x86_64__) || defined(_M_X64)
        case RNG_X86:
            break;
#endif
        default:
            Madd_Error_Add(MADD_ERROR, L"RNG_Init: unknown rng_type input.");
    }
    RNG_Init_Pointer(rng_type, &rng);
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

RNG_Param RNG_Read_BE(FILE *fp)
{
    RNG_Param rng;
    union _union32 u32 = Read_4byte_BE(fp);;
    uint32_t rng_type = u32.u;
    switch (rng_type){
        case RNG_XOSHIRO256SS:
            rng.rng.rx256 = RNG_Xoshiro256ss_Read_BE(fp);
            break;
        case RNG_MT:
            rng.rng.mt = RNG_MT_Read_BE(fp);
            break;
        case RNG_CLIB:
            Madd_Error_Add(MADD_WARNING, L"RNG_Read_BE: the RNG type read from file is standard C library, unable to reproduce.");
            break;
        case RNG_XORSHIFT64:
            rng.rng.rx64 = RNG_Xorshift64_Read_BE(fp);
            break;
        case RNG_XORSHIFT64S:
            rng.rng.rx64 = RNG_Xorshift64s_Read_BE(fp);
            break;
        case RNG_XORSHIFT1024S:
            rng.rng.rx1024 = RNG_Xorshift1024s_Read_BE(fp);
            break;
        case RNG_XOSHIRO256P:
            rng.rng.rx256 = RNG_Xoshiro256p_Read_BE(fp);
            break;
        case RNG_XORWOW:
            rng.rng.rxw = RNG_Xorwow_Read_BE(fp);
            break;
        case RNG_X86:
#if defined(__x86_64__) || defined(_M_X64)
            Madd_Error_Add(MADD_WARNING, L"RNG_Read_BE: the RNG type read from file is x86 CPU, unable to reproduce.");
#else
            Madd_Error_Add(MADD_ERROR, L"RNG_Read_BE: the RNG type read from file is x86 CPU, but this is not a x86 platform.");
#endif
            break;
        default:
            Madd_Error_Add(MADD_ERROR, L"RNG_Read_BE: unknown rng_type read from file.");
    }
    RNG_Init_Pointer(rng_type, &rng);
    return rng;
}

RNG_Param RNG_Read_LE(FILE *fp)
{
    RNG_Param rng;
    union _union32 u32 = Read_4byte_LE(fp);;
    uint32_t rng_type = u32.u;
    switch (rng_type){
        case RNG_XOSHIRO256SS:
            rng.rng.rx256 = RNG_Xoshiro256ss_Read_LE(fp);
            break;
        case RNG_MT:
            rng.rng.mt = RNG_MT_Read_LE(fp);
            break;
        case RNG_CLIB:
            Madd_Error_Add(MADD_WARNING, L"RNG_Read_LE: the RNG type read from file is standard C library, unable to reproduce.");
            break;
        case RNG_XORSHIFT64:
            rng.rng.rx64 = RNG_Xorshift64_Read_LE(fp);
            break;
        case RNG_XORSHIFT64S:
            rng.rng.rx64 = RNG_Xorshift64s_Read_LE(fp);
            break;
        case RNG_XORSHIFT1024S:
            rng.rng.rx1024 = RNG_Xorshift1024s_Read_LE(fp);
            break;
        case RNG_XOSHIRO256P:
            rng.rng.rx256 = RNG_Xoshiro256p_Read_LE(fp);
            break;
        case RNG_XORWOW:
            rng.rng.rxw = RNG_Xorwow_Read_LE(fp);
            break;
        case RNG_X86:
#if defined(__x86_64__) || defined(_M_X64)
            Madd_Error_Add(MADD_WARNING, L"RNG_Read_LE: the RNG type read from file is x86 CPU, unable to reproduce.");
#else
            Madd_Error_Add(MADD_ERROR, L"RNG_Read_LE: the RNG type read from file is x86 CPU, but this is not a x86 platform.");
#endif
            break;
        default:
            Madd_Error_Add(MADD_ERROR, L"RNG_Read_LE: unknown rng_type read from file.");
    }
    RNG_Init_Pointer(rng_type, &rng);
    return rng;
}

void RNG_Write_BE(RNG_Param *rng, FILE *fp)
{
    union _union32 u32={.u=rng->rng_type};
    Write_4byte_BE(fp, &u32);
    switch (rng->rng_type){
        case RNG_XOSHIRO256SS:
            RNG_Xoshiro256ss_Write_BE(&rng->rng.rx256, fp);
            break;
        case RNG_MT:
            RNG_MT_Write_BE(&rng->rng.mt, fp);
            break;
        case RNG_CLIB:
            Madd_Error_Add(MADD_WARNING, L"RNG_Write_BE: the RNG type write to file is standard C library, unable to record.");
            break;
        case RNG_XORSHIFT64:
            RNG_Xorshift64_Write_BE(&rng->rng.rx64, fp);
            break;
        case RNG_XORSHIFT64S:
            RNG_Xorshift64s_Write_BE(&rng->rng.rx64, fp);
            break;
        case RNG_XORSHIFT1024S:
            RNG_Xorshift1024s_Write_BE(&rng->rng.rx1024, fp);
            break;
        case RNG_XOSHIRO256P:
            RNG_Xoshiro256p_Write_BE(&rng->rng.rx256, fp);
            break;
        case RNG_XORWOW:
            RNG_Xorwow_Write_BE(&rng->rng.rxw, fp);
            break;
        case RNG_X86:
#if defined(__x86_64__) || defined(_M_X64)
            Madd_Error_Add(MADD_WARNING, L"RNG_Write_BE: the RNG type write to file is x86 CPU, unable to record.");
#else
            Madd_Error_Add(MADD_ERROR, L"RNG_Write_BE: the RNG type write to file is x86 CPU, but this is not a x86 platform.");
#endif
            break;
        default:
            Madd_Error_Add(MADD_ERROR, L"RNG_Write_BE: unknown rng_type write to file.");
    }
}

void RNG_Write_LE(RNG_Param *rng, FILE *fp)
{
    union _union32 u32={.u=rng->rng_type};
    Write_4byte_LE(fp, &u32);
    switch (rng->rng_type){
        case RNG_XOSHIRO256SS:
            RNG_Xoshiro256ss_Write_LE(&rng->rng.rx256, fp);
            break;
        case RNG_MT:
            RNG_MT_Write_LE(&rng->rng.mt, fp);
            break;
        case RNG_CLIB:
            Madd_Error_Add(MADD_WARNING, L"RNG_Write_LE: the RNG type write to file is standard C library, unable to record.");
            break;
        case RNG_XORSHIFT64:
            RNG_Xorshift64_Write_LE(&rng->rng.rx64, fp);
            break;
        case RNG_XORSHIFT64S:
            RNG_Xorshift64s_Write_LE(&rng->rng.rx64, fp);
            break;
        case RNG_XORSHIFT1024S:
            RNG_Xorshift1024s_Write_LE(&rng->rng.rx1024, fp);
            break;
        case RNG_XOSHIRO256P:
            RNG_Xoshiro256p_Write_LE(&rng->rng.rx256, fp);
            break;
        case RNG_XORWOW:
            RNG_Xorwow_Write_LE(&rng->rng.rxw, fp);
            break;
        case RNG_X86:
#if defined(__x86_64__) || defined(_M_X64)
            Madd_Error_Add(MADD_WARNING, L"RNG_Write_LE: the RNG type write to file is x86 CPU, unable to record.");
#else
            Madd_Error_Add(MADD_ERROR, L"RNG_Write_LE: the RNG type write to file is x86 CPU, but this is not a x86 platform.");
#endif
            break;
        default:
            Madd_Error_Add(MADD_ERROR, L"RNG_Write_LE: unknown rng_type write to file.");
    }
}