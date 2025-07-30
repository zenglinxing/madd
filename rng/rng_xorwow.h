/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_xorwow.h
*/
#ifndef _RNG_XORWOW_H
#define _RNG_XORWOW_H

#include<stdio.h>
#include<stdint.h>

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif

typedef struct{
    uint32_t seed, counter;
    uint32_t s1, s2, s3, add;
    uint32_t state[5];
    uint64_t n_gen;
} RNG_Xorwow_Param;

extern uint32_t RNG_Xorwow_default_state[5];

RNG_Xorwow_Param RNG_Xorwow_Init(uint32_t seed);
uint32_t RNG_Xorwow_U32(RNG_Xorwow_Param *rxp);

double Rand_Xorwow(RNG_Xorwow_Param *rxp);
float Rand_Xorwow_f32(RNG_Xorwow_Param *rxp);
long double Rand_Xorwow_fl(RNG_Xorwow_Param *rxp);
#ifdef ENABLE_QUADPRECISION
__float128 Rand_Xorwow_f128(RNG_Xorwow_Param *rxp);
#endif

RNG_Xorwow_Param RNG_Xorwow_Read_BE(FILE *fp);
RNG_Xorwow_Param RNG_Xorwow_Read_LE(FILE *fp);
void RNG_Xorwow_Write_BE(RNG_Xorwow_Param rxp, FILE *fp);
void RNG_Xorwow_Write_LE(RNG_Xorwow_Param rxp, FILE *fp);

#endif /* _RNG_XORWOW_H */
