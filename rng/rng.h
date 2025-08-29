/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng.h
*/
#ifndef MADD_RNG_H
#define MADD_RNG_H

#include<stdio.h>
#include<stdint.h>
#include"rng_MT.h"
#include"rng_clib.h"
#include"rng_xorshift64.h"
#include"rng_xorshift1024.h"
#include"rng_xoshiro256.h"
#include"rng_xorwow.h"
#include"rng_x86.h"


#define RNG_XOSHIRO256SS    0
#define RNG_MT              1
#define RNG_CLIB            2
#define RNG_XORSHIFT64      3
#define RNG_XORSHIFT64S     4
#define RNG_XORSHIFT1024S   5
#define RNG_XOSHIRO256P     6
#define RNG_XORWOW          7
#define RNG_X86             10000

typedef union{
    RNG_MT_Param mt;
    RNG_Xorshift64_Param rx64;
    RNG_Xorshift1024_Param rx1024;
    RNG_Xoshiro256_Param rx256;
    RNG_Xorwow_Param rxw;
} RNG_Union;

typedef uint64_t (*RNG_U64_t)(void *);
typedef uint32_t (*RNG_U32_t)(void *);
typedef double (*Rand_t)(void *);
typedef float (*Rand_f32_t)(void *);
typedef long double (*Rand_fl_t)(void *);
#ifdef ENABLE_QUADPRECISION
typedef __float128 (*Rand_f128_t)(void *);
#endif /* ENABLE_QUADPRECISION */

typedef struct{
    uint32_t rng_type;
    uint64_t rand_max;
    RNG_U32_t ru32;
    RNG_U64_t ru64;
    Rand_t rand;
    Rand_f32_t rand32;
    Rand_fl_t randl;
#ifdef ENABLE_QUADPRECISION
    Rand_f128_t rand128;
#endif /* ENABLE_QUADPRECISION */
    RNG_Union rng;
} RNG_Param;

void RNG_Init_Pointer(uint32_t rng_type, RNG_Param *rng);
RNG_Param RNG_Init(uint64_t seed, uint32_t rng_type);
uint64_t Rand_Uint(RNG_Param *rng);
double Rand(RNG_Param *rng);
float Rand_f32(RNG_Param *rng);
long double Rand_fl(RNG_Param *rng);
#ifdef ENABLE_QUADPRECISION
float Rand_f128(RNG_Param *rng);
#endif /* ENABLE_QUADPRECISION */

RNG_Param RNG_Read_BE(FILE *fp);
RNG_Param RNG_Read_LE(FILE *fp);
void RNG_Write_BE(RNG_Param *rng, FILE *fp);
void RNG_Write_LE(RNG_Param *rng, FILE *fp);

#endif /* MADD_RNG_H */