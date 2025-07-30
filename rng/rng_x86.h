/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/rng_x86.h
*/
#ifndef _RNG_X86_H
#define _RNG_X86_H

#if defined(__x86_64__) || defined(_M_X64)

#include<stdint.h>

extern uint64_t Madd_RNG_x86_n_gen;

double Rand_x86(void);
float Rand_x86_f32(void);
long double Rand_x86_fl(void);
#ifdef ENABLE_QUADPRECISION
__float128 Rand_x86_f128(void);
#endif

#endif /* __x86_64__ _M_X64 */

#endif /* _RNG_X86_H */