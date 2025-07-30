/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_xorshift64.h
*/
#ifndef _RNG_XORSHIFT64_H
#define _RNG_XORSHIFT64_H

#include<stdio.h>
#include<stdint.h>

typedef struct{
    uint64_t seed, n_gen, state;
    uint64_t s1, s2, s3, mul;
} RNG_Xorshift64_Param;

/* Xorshift64 */
RNG_Xorshift64_Param RNG_Xorshift64_Init(uint64_t seed);
uint64_t RNG_Xorshift64_U64(RNG_Xorshift64_Param *rxp);

double Rand_Xorshift64(RNG_Xorshift64_Param *rxp);
float Rand_Xorshift64_f32(RNG_Xorshift64_Param *rxp);
long double Rand_Xorshift64_fl(RNG_Xorshift64_Param *rxp);
#ifdef ENABLE_QUADPRECISION
__float128 Rand_Xorshift64_f128(RNG_Xorshift64_Param *rxp);
#endif /* ENABLE_QUADPRECISION */

RNG_Xorshift64_Param RNG_Xorshift64_Read_BE(FILE *fp);
RNG_Xorshift64_Param RNG_Xorshift64_Read_LE(FILE *fp);
void RNG_Xorshift64_Write_BE(RNG_Xorshift64_Param rxp, FILE *fp);
void RNG_Xorshift64_Write_LE(RNG_Xorshift64_Param rxp, FILE *fp);

/* Xorshift64* */
RNG_Xorshift64_Param RNG_Xorshift64s_Init(uint64_t seed);
uint64_t RNG_Xorshift64s_U64(RNG_Xorshift64_Param *rxp);

double Rand_Xorshift64s(RNG_Xorshift64_Param *rxp);
float Rand_Xorshift64s_f32(RNG_Xorshift64_Param *rxp);
long double Rand_Xorshift64s_fl(RNG_Xorshift64_Param *rxp);
#ifdef ENABLE_QUADPRECISION
__float128 Rand_Xorshift64s_f128(RNG_Xorshift64_Param *rxp);
#endif /* ENABLE_QUADPRECISION */

RNG_Xorshift64_Param RNG_Xorshift64s_Read_BE(FILE *fp);
RNG_Xorshift64_Param RNG_Xorshift64s_Read_LE(FILE *fp);
void RNG_Xorshift64s_Write_BE(RNG_Xorshift64_Param rxp, FILE *fp);
void RNG_Xorshift64s_Write_LE(RNG_Xorshift64_Param rxp, FILE *fp);

#endif /* _RNG_XORSHIFT64_H */