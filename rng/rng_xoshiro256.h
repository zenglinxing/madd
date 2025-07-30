/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_xoshiro256.h
*/
#ifndef _RNG_XOSHIRO256_H
#define _RNG_XOSHIRO256_H

#include<stdio.h>
#include<stdint.h>

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

typedef struct{
    uint8_t k1, k2;
    uint64_t seed, n_gen, state[4];
    uint64_t mul1, mul2, s1;
} RNG_Xoshiro256_Param;

/* Xoshiro256** */
RNG_Xoshiro256_Param RNG_Xoshiro256ss_Init(uint64_t seed_);
uint64_t RNG_Xoshiro256ss_U64(RNG_Xoshiro256_Param *rxp);

double Rand_Xoshiro256ss(RNG_Xoshiro256_Param *rxp);
float Rand_Xoshiro256ss_f32(RNG_Xoshiro256_Param *rxp);
long double Rand_Xoshiro256ss_fl(RNG_Xoshiro256_Param *rxp);
#ifdef ENABLE_QUADPRECISION
__float128 Rand_Xoshiro256ss_f128(RNG_Xoshiro256_Param *rxp);
#endif /* ENABLE_QUADPRECISION */

RNG_Xoshiro256_Param RNG_Xoshiro256ss_Read_BE(FILE *fp);
RNG_Xoshiro256_Param RNG_Xoshiro256ss_Read_LE(FILE *fp);
void RNG_Xoshiro256ss_Write_BE(RNG_Xoshiro256_Param rxp, FILE *fp);
void RNG_Xoshiro256ss_Write_LE(RNG_Xoshiro256_Param rxp, FILE *fp);

/* Xoshiro256+ */
RNG_Xoshiro256_Param RNG_Xoshiro256p_Init(uint64_t seed_);
uint64_t RNG_Xoshiro256p_U64(RNG_Xoshiro256_Param *rxp);

double Rand_Xoshiro256p(RNG_Xoshiro256_Param *rxp);
float Rand_Xoshiro256p_f32(RNG_Xoshiro256_Param *rxp);
long double Rand_Xoshiro256p_fl(RNG_Xoshiro256_Param *rxp);
#ifdef ENABLE_QUADPRECISION
__float128 Rand_Xoshiro256p_f128(RNG_Xoshiro256_Param *rxp);
#endif /* ENABLE_QUADPRECISION */

RNG_Xoshiro256_Param RNG_Xoshiro256p_Read_BE(FILE *fp);
RNG_Xoshiro256_Param RNG_Xoshiro256p_Read_LE(FILE *fp);
void RNG_Xoshiro256p_Write_BE(RNG_Xoshiro256_Param rxp, FILE *fp);
void RNG_Xoshiro256p_Write_LE(RNG_Xoshiro256_Param rxp, FILE *fp);

#endif /* _RNG_XOSHIRO256_H */