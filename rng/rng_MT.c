/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_MT.c
Mersenne Twister Generator
MT19937-64
*/
#include<stdlib.h>
#include<stdint.h>
#include"rng_MT.h"
#include"../basic/basic.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define MT_HALF  156
#define MATRIX_A 0xB5026F5AA96619E9ULL
#define UMASK64  0xFFFFFFFF80000000ULL
#define LMASK64  0x000000007FFFFFFFULL

static uint64_t umask64=BIN64>>31<<31,
                lmask64=BIN64<<33>>33;

RNG_MT_Param RNG_MT_Init(uint64_t seed)
{
    if (seed == 0) seed = 0xac23dbf;
    RNG_MT_Param mt={.n_gen=0, .seed=seed};
    uint64_t *p1=&mt.seeds[0],*p2=&mt.seeds[1];
    /*printf("%p\t%p\t%p\n",&mt->seeds[0], p1, p2);*/
    uint16_t i;
    *p1 = seed;
    for (i=1; i<312; i++,p1++,p2++){
        *p2 = 6364136223846793005 * (*p1 ^ (*p1 >> 62)) + i;
    }
    mt.i=312;
    return mt;
}

/* Generate seed */
uint64_t RNG_MT_U64(RNG_MT_Param *mt)
{
    mt->n_gen++;
    if (mt->i >= 312) {
        for (uint16_t i=0; i<312; i++) {
            uint64_t x = (mt->seeds[i] & UMASK64) | (mt->seeds[(i+1)%312] & LMASK64);
            uint64_t xA = x >> 1;
            if (x & 1) xA ^= MATRIX_A;
            mt->seeds[i] = mt->seeds[(i+MT_HALF)%312] ^ xA;
        }
        mt->i = 0;
    }
    uint64_t y = mt->seeds[mt->i++];
    y ^= (y >> 29) & 0x5555555555555555;
    y ^= (y << 17) & 0x71d67fffeda60000;
    y ^= (y << 37) & 0xfff7eee000000000;
    y ^= (y >> 43);
    return y;
}

/* Get & Set seed */
void RNG_MT_Set_Seed(RNG_MT_Param *mt,uint64_t seed)
{
    register uint64_t i_next=mt->i;
    i_next = (i_next>=312) ? 0 : i_next+1;
    mt->seeds[i_next]=seed;
}

uint64_t RNG_MT_Get_Seed(RNG_MT_Param *mt)
{
    register uint64_t i_next=mt->i;
    i_next = (i_next>=312) ? 0 : i_next+1;
    return mt->seeds[i_next];
}

/* Rand 01 */
#define RAND_MT__ALGORITHM(num_type) \
{ \
    num_type bin64 = (num_type)BIN64; \
    uint64_t y = RNG_MT_U64(mt); \
    return y/bin64; \
} \

double Rand_MT(RNG_MT_Param *mt)
RAND_MT__ALGORITHM(double)

float Rand_MT_f32(RNG_MT_Param *mt)
RAND_MT__ALGORITHM(float)

long double Rand_MT_fl(RNG_MT_Param *mt)
RAND_MT__ALGORITHM(long double)

#ifdef ENABLE_QUADPRECISION
__float128 Rand_MT_f128(RNG_MT_Param *mt)
RAND_MT__ALGORITHM(__float128)
#endif /* ENABLE_QUADPRECISION */

/* Write & Read RNG_MT_Param */
RNG_MT_Param RNG_MT_Read_BE(FILE *fp)
{
    RNG_MT_Param mt;
    mt.i = Read_2byte_BE(fp).u;
    mt.n_gen = Read_8byte_BE(fp).u;
    mt.seed = Read_8byte_BE(fp).u;
    Read_Array_BE(fp, mt.seeds, 312, sizeof(uint64_t));
    return mt;
}

RNG_MT_Param RNG_MT_Read_LE(FILE *fp)
{
    RNG_MT_Param mt;
    mt.i = Read_2byte_LE(fp).u;
    mt.n_gen = Read_8byte_LE(fp).u;
    mt.seed = Read_8byte_LE(fp).u;
    Read_Array_LE(fp, mt.seeds, 312, sizeof(uint64_t));
    return mt;
}

void RNG_MT_Write_BE(RNG_MT_Param mt, FILE *fp)
{
    Write_2byte_BE(fp, &mt.i);
    Write_8byte_BE(fp, &mt.n_gen);
    Write_8byte_BE(fp, &mt.seed);
    Write_Array_BE(fp, mt.seeds, 312, sizeof(uint64_t));
}

void RNG_MT_Write_LE(RNG_MT_Param mt, FILE *fp)
{
    Write_2byte_LE(fp, &mt.i);
    Write_8byte_LE(fp, &mt.n_gen);
    Write_8byte_LE(fp, &mt.seed);
    Write_Array_LE(fp, mt.seeds, 312, sizeof(uint64_t));
}