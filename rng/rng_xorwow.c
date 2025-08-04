/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_xorwow.c
The initialization of state is recommand by Doubao AI.
*/
#include<stdio.h>
#include<stdint.h>
#include<string.h>
#include<stdbool.h>
#include"rng_xorwow.h"
#include"../basic/basic.h"
#include"../basic/constant.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif

uint32_t RNG_Xorwow_default_state[5] = {
    123456789, 362436069, 521288629, 88675123, 5783321
};

inline uint32_t hash64to32(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return (uint32_t)(z ^ (z >> 31));
}

RNG_Xorwow_Param RNG_Xorwow_Init(uint32_t seed)
{
    RNG_Xorwow_Param rxp={.seed=seed};
    if (seed == 0){
        memcpy(rxp.state, RNG_Xorwow_default_state, 5*sizeof(uint32_t));
    }else{
        rxp.state[0] = hash64to32(seed);
        rxp.state[1] = hash64to32(seed ^ 0x5555555555555555ULL);
        rxp.state[2] = hash64to32(seed ^ 0xAAAAAAAAAAAAAAAULL);
        rxp.state[3] = hash64to32(seed ^ 0x123456789ABCDEFULL);
        rxp.state[4] = hash64to32(seed ^ 0xFEDCBA987654321ULL);
        
        bool all_zero = true;
        for (int i = 0; i < 5; ++i) {
            if (rxp.state[i] != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            rxp.state[0] = 1;
        }
    }
    rxp.counter = 6615241;
    rxp.s1 = 2;
    rxp.s2 = 1;
    rxp.s3 = 4;
    rxp.add = 362437;
    return rxp;
}

uint32_t RNG_Xorwow_U32(RNG_Xorwow_Param *rxp)
{
    rxp->n_gen ++;
    uint32_t t = rxp->state[0], v = rxp->state[4];
    t ^= t >> rxp->s1;
    t ^= t << rxp->s2;
    memmove(rxp->state, rxp->state+1, 4*sizeof(uint32_t));
    v ^= v << rxp->s3;
    v ^= t;
    rxp->state[4] = v;
    rxp->counter += rxp->add;
    return rxp->counter + v /* rxp->a5 */;
}

double Rand_Xorwow(RNG_Xorwow_Param *rxp)
{
    uint32_t state = RNG_Xorwow_U32(rxp);
    return state / (double)BIN32;
}

float Rand_Xorwow_f32(RNG_Xorwow_Param *rxp)
{
    uint32_t state = RNG_Xorwow_U32(rxp);
    return state / (float)BIN32;
}

long double Rand_Xorwow_fl(RNG_Xorwow_Param *rxp)
{
    uint32_t state = RNG_Xorwow_U32(rxp);
    return state / (long double)BIN32;
}

#ifdef ENABLE_QUADPRECISION
__float128 Rand_Xorwow_f128(RNG_Xorwow_Param *rxp)
{
    uint32_t state = RNG_Xorwow_U32(rxp);
    return state / (__float128)BIN32;
}
#endif

RNG_Xorwow_Param RNG_Xorwow_Read_BE(FILE *fp)
{
    RNG_Xorwow_Param rxp;
    rxp.seed = Read_4byte_BE(fp).u;
    rxp.n_gen = Read_8byte_BE(fp).u;
    Read_Array_BE(fp, rxp.state, 5, sizeof(uint32_t));
    rxp.counter = Read_4byte_BE(fp).u;
    rxp.s1 = Read_4byte_BE(fp).u;
    rxp.s2 = Read_4byte_BE(fp).u;
    rxp.s3 = Read_4byte_BE(fp).u;
    rxp.add = Read_4byte_BE(fp).u;
    return rxp;
}

RNG_Xorwow_Param RNG_Xorwow_Read_LE(FILE *fp)
{
    RNG_Xorwow_Param rxp;
    rxp.seed = Read_4byte_LE(fp).u;
    rxp.n_gen = Read_8byte_LE(fp).u;
    Read_Array_LE(fp, rxp.state, 5, sizeof(uint32_t));
    rxp.counter = Read_4byte_LE(fp).u;
    rxp.s1 = Read_4byte_LE(fp).u;
    rxp.s2 = Read_4byte_LE(fp).u;
    rxp.s3 = Read_4byte_LE(fp).u;
    rxp.add = Read_4byte_LE(fp).u;
    return rxp;
}

void RNG_Xorwow_Write_BE(RNG_Xorwow_Param *rxp, FILE *fp)
{
    Write_4byte_BE(fp, &rxp->seed);
    Write_8byte_BE(fp, &rxp->n_gen);
    Write_Array_BE(fp, rxp->state, 5, sizeof(uint32_t));
    Write_4byte_BE(fp, &rxp->counter);
    Write_4byte_BE(fp, &rxp->s1);
    Write_4byte_BE(fp, &rxp->s2);
    Write_4byte_BE(fp, &rxp->s3);
    Write_4byte_BE(fp, &rxp->add);
}

void RNG_Xorwow_Write_LE(RNG_Xorwow_Param *rxp, FILE *fp)
{
    Write_4byte_LE(fp, &rxp->seed);
    Write_8byte_LE(fp, &rxp->n_gen);
    Write_Array_LE(fp, rxp->state, 5, sizeof(uint32_t));
    Write_4byte_LE(fp, &rxp->counter);
    Write_4byte_LE(fp, &rxp->s1);
    Write_4byte_LE(fp, &rxp->s2);
    Write_4byte_LE(fp, &rxp->s3);
    Write_4byte_LE(fp, &rxp->add);
}