/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_xorshift64s.c
*/
#include<stdio.h>
#include<stdint.h>
#include"rng_xorshift64.h"
#include"../basic/basic.h"
#include"../basic/constant.h"

RNG_Xorshift64_Param RNG_Xorshift64s_Init(uint64_t seed)
{
    RNG_Xorshift64_Param rxp={.seed=seed, .n_gen=0, .state=seed};
    rxp.s1 = 12;
    rxp.s2 = 25;
    rxp.s3 = 27;
    rxp.mul = 0x2545f4914f6cdd1dL;
    return rxp;
}

uint64_t RNG_Xorshift64s_U64(RNG_Xorshift64_Param *rxp)
{
    rxp->n_gen ++;
    uint64_t x = rxp->state;
    x ^= x >> rxp->s1;
    x ^= x << rxp->s2;
    x ^= x >> rxp->s3;
    rxp->state = x;
    return x * rxp->mul;
}

double Rand_Xorshift64s(RNG_Xorshift64_Param *rxp)
{
    uint64_t state = RNG_Xorshift64s_U64(rxp);
    return state / (double)BIN64;
}

float Rand_Xorshift64s_f32(RNG_Xorshift64_Param *rxp)
{
    uint64_t state = RNG_Xorshift64s_U64(rxp);
    return state / (float)BIN64;
}

long double Rand_Xorshift64s_fl(RNG_Xorshift64_Param *rxp)
{
    uint64_t state = RNG_Xorshift64s_U64(rxp);
    return state / (long double)BIN64;
}

#ifdef ENABLE_QUADPRECISION
__float128 Rand_Xorshift64s_f128(RNG_Xorshift64_Param *rxp)
{
    uint64_t state = RNG_Xorshift64s_U64(rxp);
    return state / (__float128)BIN64;
}
#endif /* ENABLE_QUADPRECISION */

RNG_Xorshift64_Param RNG_Xorshift64s_Read_BE(FILE *fp)
{
    RNG_Xorshift64_Param rxp;
    rxp.seed = Read_8byte_BE(fp).u;
    rxp.n_gen = Read_8byte_BE(fp).u;
    rxp.state = Read_8byte_BE(fp).u;
    rxp.s1 = Read_8byte_BE(fp).u;
    rxp.s2 = Read_8byte_BE(fp).u;
    rxp.s3 = Read_8byte_BE(fp).u;
    rxp.mul = Read_8byte_BE(fp).u;
    return rxp;
}

RNG_Xorshift64_Param RNG_Xorshift64s_Read_LE(FILE *fp)
{
    RNG_Xorshift64_Param rxp;
    rxp.seed = Read_8byte_LE(fp).u;
    rxp.n_gen = Read_8byte_LE(fp).u;
    rxp.state = Read_8byte_LE(fp).u;
    rxp.s1 = Read_8byte_LE(fp).u;
    rxp.s2 = Read_8byte_LE(fp).u;
    rxp.s3 = Read_8byte_LE(fp).u;
    rxp.mul = Read_8byte_LE(fp).u;
    return rxp;
}

void RNG_Xorshift64s_Write_BE(RNG_Xorshift64_Param *rxp, FILE *fp)
{
    Write_8byte_BE(fp, &rxp->seed);
    Write_8byte_BE(fp, &rxp->n_gen);
    Write_8byte_BE(fp, &rxp->state);
    Write_8byte_BE(fp, &rxp->s1);
    Write_8byte_BE(fp, &rxp->s2);
    Write_8byte_BE(fp, &rxp->s3);
    Write_8byte_BE(fp, &rxp->mul);
}

void RNG_Xorshift64s_Write_LE(RNG_Xorshift64_Param *rxp, FILE *fp)
{
    Write_8byte_LE(fp, &rxp->seed);
    Write_8byte_LE(fp, &rxp->n_gen);
    Write_8byte_LE(fp, &rxp->state);
    Write_8byte_LE(fp, &rxp->s1);
    Write_8byte_LE(fp, &rxp->s2);
    Write_8byte_LE(fp, &rxp->s3);
    Write_8byte_LE(fp, &rxp->mul);
}