/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_xorshift64.c
*/
#include<stdio.h>
#include<stdint.h>
#include"rng_xorshift64.h"
#include"../basic/basic.h"
#include"../basic/constant.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif

RNG_Xorshift64_Param RNG_Xorshift64_Init(uint64_t seed)
{
    RNG_Xorshift64_Param rxp={.seed=seed, .n_gen=0, .state=seed};
    rxp.s1 = 13;
    rxp.s2 = 7;
    rxp.s3 = 17;
    return rxp;
}

uint64_t RNG_Xorshift64_U64(RNG_Xorshift64_Param *rxp)
{
    rxp->n_gen ++;
    uint64_t state = rxp->state;
    state ^= state << rxp->s1;
    state ^= state >> rxp->s2;
    state ^= state << rxp->s3;
    return rxp->state = state;
}

double Rand_Xorshift64(RNG_Xorshift64_Param *rxp)
{
    uint64_t state = RNG_Xorshift64_U64(rxp);
    return state / (double)BIN64;
}

float Rand_Xorshift64_f32(RNG_Xorshift64_Param *rxp)
{
    uint64_t state = RNG_Xorshift64_U64(rxp);
    return state / (float)BIN64;
}

long double Rand_Xorshift64_fl(RNG_Xorshift64_Param *rxp)
{
    uint64_t state = RNG_Xorshift64_U64(rxp);
    return state / (long double)BIN64;
}

#ifdef ENABLE_QUADPRECISION
__float128 Rand_Xorshift64_f128(RNG_Xorshift64_Param *rxp)
{
    uint64_t state = RNG_Xorshift64_U64(rxp);
    return state / (__float128)BIN64;
}
#endif /* ENABLE_QUADPRECISION */

RNG_Xorshift64_Param RNG_Xorshift64_Read_BE(FILE *fp)
{
    RNG_Xorshift64_Param rxp;
    rxp.seed = Read_8byte_BE(fp).u;
    rxp.n_gen = Read_8byte_BE(fp).u;
    rxp.state = Read_8byte_BE(fp).u;
    rxp.s1 = Read_8byte_BE(fp).u;
    rxp.s2 = Read_8byte_BE(fp).u;
    rxp.s3 = Read_8byte_BE(fp).u;
    return rxp;
}

RNG_Xorshift64_Param RNG_Xorshift64_Read_LE(FILE *fp)
{
    RNG_Xorshift64_Param rxp;
    rxp.seed = Read_8byte_LE(fp).u;
    rxp.n_gen = Read_8byte_LE(fp).u;
    rxp.state = Read_8byte_LE(fp).u;
    rxp.s1 = Read_8byte_LE(fp).u;
    rxp.s2 = Read_8byte_LE(fp).u;
    rxp.s3 = Read_8byte_LE(fp).u;
    return rxp;
}

void RNG_Xorshift64_Write_BE(RNG_Xorshift64_Param *rxp, FILE *fp)
{
    Write_8byte_BE(fp, &rxp->seed);
    Write_8byte_BE(fp, &rxp->n_gen);
    Write_8byte_BE(fp, &rxp->state);
    Write_8byte_BE(fp, &rxp->s1);
    Write_8byte_BE(fp, &rxp->s2);
    Write_8byte_BE(fp, &rxp->s3);
}

void RNG_Xorshift64_Write_LE(RNG_Xorshift64_Param *rxp, FILE *fp)
{
    Write_8byte_LE(fp, &rxp->seed);
    Write_8byte_LE(fp, &rxp->n_gen);
    Write_8byte_LE(fp, &rxp->state);
    Write_8byte_LE(fp, &rxp->s1);
    Write_8byte_LE(fp, &rxp->s2);
    Write_8byte_LE(fp, &rxp->s3);
}