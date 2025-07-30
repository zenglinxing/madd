/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_clib.h
*/
#ifndef _RNG_CLIB_H
#define _RNG_CLIB_H

#include<stdint.h>

extern uint64_t Madd_RNG_Clib_seed, Madd_RNG_Clib_n_gen;

void RNG_Clib_Init(uint64_t seed);
uint64_t RNG_Clib_U64(void);
double Rand_Clib(void);
float Rand_Clib_f32(void);
long double Rand_Clib_fl(void);
#ifdef ENABLE_QUADPRECISION
__float128 Rand_Clib_f128(void);
#endif /* ENABLE_QUADPRECISION */

#endif /* _RNG_CLIB_H */