/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_xorshift1024.h
*/
#ifndef MADD_RNG_XORSHIFT1024_H
#define MADD_RNG_XORSHIFT1024_H

#include<stdio.h>
#include<stdint.h>

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define RNG_XORSHIFT1024S_JUMP_LEN 16

typedef struct{
    uint64_t seed, n_gen, state[16];
    uint64_t s1, s2, s3, mul, index;
} RNG_Xorshift1024_Param;

extern uint64_t RNG_Xorshift1024s_default_jump[RNG_XORSHIFT1024S_JUMP_LEN];

/* Xorshift64 */
RNG_Xorshift1024_Param RNG_Xorshift1024s_Init(uint64_t seed_);
uint64_t RNG_Xorshift1024s_U64(RNG_Xorshift1024_Param *rxp);

double Rand_Xorshift1024s(RNG_Xorshift1024_Param *rxp);
float Rand_Xorshift1024s_f32(RNG_Xorshift1024_Param *rxp);
long double Rand_Xorshift1024s_fl(RNG_Xorshift1024_Param *rxp);
#ifdef ENABLE_QUADPRECISION
__float128 Rand_Xorshift1024s_f128(RNG_Xorshift1024_Param *rxp);
#endif /* ENABLE_QUADPRECISION */

void RNG_Xorshift1024s_Jump(RNG_Xorshift1024_Param *rxp);

RNG_Xorshift1024_Param RNG_Xorshift1024s_Read_BE(FILE *fp);
RNG_Xorshift1024_Param RNG_Xorshift1024s_Read_LE(FILE *fp);
void RNG_Xorshift1024s_Write_BE(RNG_Xorshift1024_Param *rxp, FILE *fp);
void RNG_Xorshift1024s_Write_LE(RNG_Xorshift1024_Param *rxp, FILE *fp);

#endif /* MADD_RNG_XORSHIFT1024_H */