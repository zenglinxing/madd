/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_xoshiro256p.c
Xoshiro256+
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

inline uint64_t splitmix64(uint64_t *seed) {
    uint64_t z = (*seed += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

RNG_Xoshiro256_Param RNG_Xoshiro256p_Init(uint64_t seed_)
{
    uint64_t seed = (seed_) ? seed_ : 1;
    RNG_Xoshiro256_Param rxp={.seed=seed, .n_gen=0, .k2=45, .s1=17};
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
        rxp.state[0] = 1; 
    }
    return rxp;
}

uint64_t RNG_Xoshiro256p_U64(RNG_Xoshiro256_Param *rxp)
{
    rxp->n_gen ++;
    uint64_t *s = rxp->state, res = s[0] + s[3], t = s[1] << rxp->s1;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rol64(s[3], rxp->k2);
    return res;
}

double Rand_Xoshiro256p(RNG_Xoshiro256_Param *rxp)
{
    uint64_t res = RNG_Xoshiro256p_U64(rxp);
    return res / (double)BIN64;
}

float Rand_Xoshiro256p_f32(RNG_Xoshiro256_Param *rxp)
{
    uint64_t res = RNG_Xoshiro256p_U64(rxp);
    return res / (float)BIN64;
}

long double Rand_Xoshiro256p_fl(RNG_Xoshiro256_Param *rxp)
{
    uint64_t res = RNG_Xoshiro256p_U64(rxp);
    return res / (long double)BIN64;
}

#ifdef ENABLE_QUADPRECISION
__float128 Rand_Xoshiro256p_f128(RNG_Xoshiro256_Param *rxp)
{
    uint64_t res = RNG_Xoshiro256p_U64(rxp);
    return res / (__float128)BIN64;
}
#endif /* ENABLE_QUADPRECISION */

RNG_Xoshiro256_Param RNG_Xoshiro256p_Read_BE(FILE *fp)
{
    RNG_Xoshiro256_Param rxp;
    rxp.seed = Read_8byte_BE(fp).u;
    rxp.n_gen = Read_8byte_BE(fp).u;
    Read_Array_BE(fp, rxp.state, 4, sizeof(uint64_t));
    rxp.s1 = Read_8byte_BE(fp).u;
    rxp.k2 = Read_1byte(fp).u;
    return rxp;
}

RNG_Xoshiro256_Param RNG_Xoshiro256p_Read_LE(FILE *fp)
{
    RNG_Xoshiro256_Param rxp;
    rxp.seed = Read_8byte_LE(fp).u;
    rxp.n_gen = Read_8byte_LE(fp).u;
    Read_Array_LE(fp, rxp.state, 4, sizeof(uint64_t));
    rxp.s1 = Read_8byte_LE(fp).u;
    rxp.k2 = Read_1byte(fp).u;
    return rxp;
}

void RNG_Xoshiro256p_Write_BE(RNG_Xoshiro256_Param rxp, FILE *fp)
{
    Write_8byte_BE(fp, &rxp.seed);
    Write_8byte_BE(fp, &rxp.n_gen);
    Write_Array_BE(fp, rxp.state, 4, sizeof(uint64_t));
    Write_8byte_BE(fp, &rxp.s1);
    Write_1byte(fp, &rxp.k2);
}

void RNG_Xoshiro256p_Write_LE(RNG_Xoshiro256_Param rxp, FILE *fp)
{
    Write_8byte_LE(fp, &rxp.seed);
    Write_8byte_LE(fp, &rxp.n_gen);
    Write_Array_LE(fp, rxp.state, 4, sizeof(uint64_t));
    Write_8byte_LE(fp, &rxp.s1);
    Write_1byte(fp, &rxp.k2);
}