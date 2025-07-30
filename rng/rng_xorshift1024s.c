/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_xorshift1024s.c
Xorshift2014**
*/
#include<stdio.h>
#include<stdint.h>
#include<string.h>
#include<stdbool.h>
#include"rng_xorshift1024.h"
#include"../basic/basic.h"
#include"../basic/constant.h"

uint64_t RNG_Xorshift1024s_default_jump[RNG_XORSHIFT1024S_JUMP_LEN] = {
    0x84242f96eca9c41dUL, 0xa3c65b8776f96855UL, 0x5b34a39f070b5837UL, 0x4489affce4f31a1eUL,
    0x2ffeeb0a48316f40UL, 0xdc2d9891fe68c022UL, 0x3659132bb12fea70UL, 0xaac17d8efa43cab8UL,
    0xc4cb815590989b13UL, 0x5ee975283d71c93bUL, 0x691548c86c1bd540UL, 0x7910c41d10a1e6a5UL,
    0x0b5fc64563b3e2a8UL, 0x047f7684e9fc949dUL, 0xb99181f2d8f685caUL, 0x284600e3f30e38c3UL
};

/* Xorshift64 */
RNG_Xorshift1024_Param RNG_Xorshift1024s_Init(uint64_t seed_)
{
    RNG_Xorshift1024_Param rxp;
    uint64_t seed = (seed_) ? seed_ : 1, s;
    rxp.seed = seed;
    int i;
    for (i=1; i<16; i++){
        s = rxp.state[i-1];
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        rxp.state[i] = s * 0x2545F4914F6CDD1DULL;
    }
    rxp.s1 = 31;
    rxp.s2 = 11;
    rxp.s3 = 30;
    rxp.mul = 1181783497276652981UL;
    rxp.index = 0;
    rxp.n_gen = 0;
    return rxp;
}

uint64_t RNG_Xorshift1024s_U64(RNG_Xorshift1024_Param *rxp)
{
    rxp->n_gen ++;
    uint64_t s0 = rxp->state[rxp->index];
    rxp->index = (rxp->index + 1) & 15;
    uint64_t s1 = rxp->state[rxp->index];
    s1 ^= s1 << rxp->s1;
    rxp->state[rxp->index] = s1 ^ (s1 >> rxp->s2) ^ s0 ^ (s0 >> rxp->s3);
    return rxp->state[rxp->index] * rxp->mul;
}

double Rand_Xorshift1024s(RNG_Xorshift1024_Param *rxp)
{
    uint64_t state = RNG_Xorshift1024s_U64(rxp);
    return state / (double)BIN64;
}

float Rand_Xorshift1024s_f32(RNG_Xorshift1024_Param *rxp)
{
    uint64_t state = RNG_Xorshift1024s_U64(rxp);
    return state / (float)BIN64;
}

long double Rand_Xorshift1024s_fl(RNG_Xorshift1024_Param *rxp)
{
    uint64_t state = RNG_Xorshift1024s_U64(rxp);
    return state / (long double)BIN64;
}

#ifdef ENABLE_QUADPRECISION
__float128 Rand_Xorshift1024s_f128(RNG_Xorshift1024_Param *rxp)
{
    uint64_t state = RNG_Xorshift1024s_U64(rxp);
    return state / (__float128)BIN64;
}
#endif /* ENABLE_QUADPRECISION */

void RNG_Xorshift1024s_Jump(RNG_Xorshift1024_Param *rxp)
{
    uint64_t *jump = &RNG_Xorshift1024s_default_jump[0];
    uint64_t t[16];
    memset(t, 0, 16*sizeof(uint64_t));
    int i, j, b;
    for (i=0; i<RNG_XORSHIFT1024S_JUMP_LEN; i++){
        for (b=0; b<64; b++){
            if (jump[i] & 1UL << b){
                for (j=0; j<16; j++){
                    t[j] ^= rxp->state[(j+rxp->index) & 15];
                }
            }
            RNG_Xorshift1024s_U64(rxp);
        }
    }
    for (j=0; j<16; j++){
        rxp->state[(j+rxp->index) & 15] = t[j];
    }
}

RNG_Xorshift1024_Param RNG_Xorshift1024s_Read_BE(FILE *fp)
{
    RNG_Xorshift1024_Param rxp;
    rxp.seed = Read_8byte_BE(fp).u;
    rxp.n_gen = Read_8byte_BE(fp).u;
    Read_Array_BE(fp, rxp.state, 16, sizeof(uint64_t));
    rxp.s1 = Read_8byte_BE(fp).u;
    rxp.s2 = Read_8byte_BE(fp).u;
    rxp.s3 = Read_8byte_BE(fp).u;
    rxp.mul = Read_8byte_BE(fp).u;
    rxp.index = Read_8byte_BE(fp).u;
    return rxp;
}

RNG_Xorshift1024_Param RNG_Xorshift1024s_Read_LE(FILE *fp)
{
    RNG_Xorshift1024_Param rxp;
    rxp.seed = Read_8byte_LE(fp).u;
    rxp.n_gen = Read_8byte_LE(fp).u;
    Read_Array_LE(fp, rxp.state, 16, sizeof(uint64_t));
    rxp.s1 = Read_8byte_LE(fp).u;
    rxp.s2 = Read_8byte_LE(fp).u;
    rxp.s3 = Read_8byte_LE(fp).u;
    rxp.mul = Read_8byte_LE(fp).u;
    rxp.index = Read_8byte_LE(fp).u;
    return rxp;
}

void RNG_Xorshift1024s_Write_BE(RNG_Xorshift1024_Param rxp, FILE *fp)
{
    Write_8byte_BE(fp, &rxp.seed);
    Write_8byte_BE(fp, &rxp.n_gen);
    Write_Array_BE(fp, rxp.state, 16, sizeof(uint64_t));
    Write_8byte_BE(fp, &rxp.s1);
    Write_8byte_BE(fp, &rxp.s2);
    Write_8byte_BE(fp, &rxp.s3);
    Write_8byte_BE(fp, &rxp.mul);
    Write_8byte_BE(fp, &rxp.index);
}

void RNG_Xorshift1024s_Write_LE(RNG_Xorshift1024_Param rxp, FILE *fp)
{
    Write_8byte_LE(fp, &rxp.seed);
    Write_8byte_LE(fp, &rxp.n_gen);
    Write_Array_LE(fp, rxp.state, 16, sizeof(uint64_t));
    Write_8byte_LE(fp, &rxp.s1);
    Write_8byte_LE(fp, &rxp.s2);
    Write_8byte_LE(fp, &rxp.s3);
    Write_8byte_LE(fp, &rxp.mul);
    Write_8byte_LE(fp, &rxp.index);
}