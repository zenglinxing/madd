/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_xoshiro256ss.c
Xoshiro256**
The initialization of state is recommand by Doubao AI.
*/
#include<stdio.h>
#include<stdint.h>
#include<stdbool.h>
#include"rng_xoshiro256.h"
#include"../basic/basic.h"
#include"../basic/constant.h"

inline uint64_t rol64(uint64_t x, uint8_t k)
{
    uint64_t x1, x2;
    x1 = x << k;
    x2 = x >> (64 - k);
    return x1 | x2;
}

inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

RNG_Xoshiro256_Param RNG_Xoshiro256ss_Init(uint64_t seed_)
{
    uint64_t seed = (seed_) ? seed_ : 1;
    RNG_Xoshiro256_Param rxp={.n_gen=0, .seed=seed, .mul1=5, .mul2=9, .k1=7, .k2=45, .s1=17};
    rxp.state[0] = splitmix64(&seed);
    rxp.state[1] = splitmix64(&seed);
    rxp.state[2] = splitmix64(&seed);
    rxp.state[3] = splitmix64(&seed);

    bool all_zero = true;
    for (int i = 0; i < 4; ++i) {
        if (rxp.state[i] != 0) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        rxp.state[0] = 1; // 避免全0状态
    }
    return rxp;
}

uint64_t RNG_Xoshiro256ss_U64(RNG_Xoshiro256_Param *rxp)
{
    rxp->n_gen ++;
    uint64_t *s = rxp->state,
             res = rol64(rxp->state[1]*rxp->mul1, rxp->k1) * rxp->mul2,
             t = rxp->state[1] << rxp->s1;
    rxp->state[2] ^= rxp->state[0];
    rxp->state[3] ^= rxp->state[1];
    rxp->state[1] ^= rxp->state[2];
    rxp->state[0] ^= rxp->state[3];
    rxp->state[2] ^= t;
    rxp->state[3] = rol64(rxp->state[3], rxp->k2);
    return res;
}

double Rand_Xoshiro256ss(RNG_Xoshiro256_Param *rxp)
{
    uint64_t res = RNG_Xoshiro256ss_U64(rxp);
    return res / (double)BIN64;
}

float Rand_Xoshiro256ss_f32(RNG_Xoshiro256_Param *rxp)
{
    uint64_t res = RNG_Xoshiro256ss_U64(rxp);
    return res / (float)BIN64;
}

long double Rand_Xoshiro256ss_fl(RNG_Xoshiro256_Param *rxp)
{
    uint64_t res = RNG_Xoshiro256ss_U64(rxp);
    return res / (long double)BIN64;
}

#ifdef ENABLE_QUADPRECISION
__float128 Rand_Xoshiro256ss_f128(RNG_Xoshiro256_Param *rxp)
{
    uint64_t res = RNG_Xoshiro256ss_U64(rxp);
    return res / (__float128)BIN64;
}
#endif /* ENABLE_QUADPRECISION */

RNG_Xoshiro256_Param RNG_Xoshiro256ss_Read_BE(FILE *fp)
{
    RNG_Xoshiro256_Param rxp;
    rxp.seed = Read_8byte_BE(fp).u;
    rxp.n_gen = Read_8byte_BE(fp).u;
    Read_Array_BE(fp, rxp.state, 4, sizeof(uint64_t));
    rxp.mul1 = Read_8byte_BE(fp).u;
    rxp.mul2 = Read_8byte_BE(fp).u;
    rxp.s1 = Read_8byte_BE(fp).u;
    rxp.k1 = Read_1byte(fp).u;
    rxp.k2 = Read_1byte(fp).u;
    return rxp;
}

RNG_Xoshiro256_Param RNG_Xoshiro256ss_Read_LE(FILE *fp)
{
    RNG_Xoshiro256_Param rxp;
    rxp.seed = Read_8byte_LE(fp).u;
    rxp.n_gen = Read_8byte_LE(fp).u;
    Read_Array_LE(fp, rxp.state, 4, sizeof(uint64_t));
    rxp.mul1 = Read_8byte_LE(fp).u;
    rxp.mul2 = Read_8byte_LE(fp).u;
    rxp.s1 = Read_8byte_LE(fp).u;
    rxp.k1 = Read_1byte(fp).u;
    rxp.k2 = Read_1byte(fp).u;
    return rxp;
}

void RNG_Xoshiro256ss_Write_BE(RNG_Xoshiro256_Param rxp, FILE *fp)
{
    Write_8byte_BE(fp, &rxp.seed);
    Write_8byte_BE(fp, &rxp.n_gen);
    Write_Array_BE(fp, rxp.state, 4, sizeof(uint64_t));
    Write_8byte_BE(fp, &rxp.mul1);
    Write_8byte_BE(fp, &rxp.mul2);
    Write_8byte_BE(fp, &rxp.s1);
    Write_1byte(fp, &rxp.k1);
    Write_1byte(fp, &rxp.k2);
}

void RNG_Xoshiro256ss_Write_LE(RNG_Xoshiro256_Param rxp, FILE *fp)
{
    Write_8byte_LE(fp, &rxp.seed);
    Write_8byte_LE(fp, &rxp.n_gen);
    Write_Array_LE(fp, rxp.state, 4, sizeof(uint64_t));
    Write_8byte_LE(fp, &rxp.mul1);
    Write_8byte_LE(fp, &rxp.mul2);
    Write_8byte_LE(fp, &rxp.s1);
    Write_1byte(fp, &rxp.k1);
    Write_1byte(fp, &rxp.k2);
}