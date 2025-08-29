/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_MT.H
Mersenne Twister Generator
MT19937-64
*/
#ifndef MADD_RNG_MT_H
#define MADD_RNG_MT_H

#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include"../basic/constant.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

/*struct _RNG_MT_Element{
    uint64_t seed;
    struct _RNG_MT_Element *next;
};

typedef struct _RNG_MT_Element RNG_MT_Element;*/

typedef struct{
    uint16_t i;
    /*RNG_MT_Element *p1,*p2,seeds[312];*/
    uint64_t n_gen,seed,*p1,*p2,seeds[312];
} RNG_MT_Param;

RNG_MT_Param RNG_MT_Init(uint64_t seed);

/* Generate seed */
uint64_t RNG_MT_U64(RNG_MT_Param *mt);

/* Get & Set seed */
void RNG_MT_Set_Seed(RNG_MT_Param *mt,uint64_t seed);
uint64_t RNG_MT_Get_Seed(RNG_MT_Param *mt);

/* random number: [0, 1) */
double Rand_MT(RNG_MT_Param *mt);
float Rand_MT_f32(RNG_MT_Param *mt);
long double Rand_MT_fl(RNG_MT_Param *mt);

#ifdef ENABLE_QUADPRECISION
__float128 Rand_MT_f128(RNG_MT_Param *mt);
#endif /* ENABLE_QUADPRECISION */

/* Write & Read RNG_MT_Param */
RNG_MT_Param RNG_MT_Read_BE(FILE *fp);
RNG_MT_Param RNG_MT_Read_LE(FILE *fp);
void RNG_MT_Write_BE(RNG_MT_Param *mt, FILE *fp);
void RNG_MT_Write_LE(RNG_MT_Param *mt, FILE *fp);

#endif /* MADD_RNG_MT_H */