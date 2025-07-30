/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/constant.h

This file is aimed to collect majority of prevalent constants used in math.
*/
#ifndef _CONSTANT_H
#define _CONSTANT_H

#include<float.h>
#include<math.h>
#include<stdint.h>

/* For GCC, it could support quad-precision float type */
#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

/* _CONSTANT_C_ALLOW_GLOBAL_VAR_CALC_INIT */
#if defined(__GNUC__)
#define _CONSTANT_C_ALLOW_GLOBAL_VAR_CALC_INIT 1
#else
#define _CONSTANT_C_ALLOW_GLOBAL_VAR_CALC_INIT 0
#endif

/* Masks */
#define BIN4  0x0f
#define BIN7  0x7f
#define BIN8  0xff
#define BIN15 0x7fff
#define BIN16 0xffff
#define BIN31 0x7fffffff
#define BIN32 0xffffffff
#define BIN63 0x7fffffffffffffff
#define BIN64 0xffffffffffffffff
extern uint8_t Bin4, Bin5, Bin6, Bin8, Bin7;
extern uint16_t Bin16, Bin15;
extern uint32_t Bin32, Bin31;
extern uint64_t Bin64, Bin63;

/* Basic constants */
#define _CONSTANT_PI 3.1415926535897932384626
#define _CONSTANT_E  2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274
/*
------
double
------
*/
extern double Pi, E_Nat;
extern double Inf, NaN;

/*
-----
float
-----
*/
extern float Pi_f32, E_Nat_f32;
extern float Inf_f32, NaN_f32;

/*
-----------
long double
-----------
*/
extern long double Pi_fl, E_Nat_fl;
extern long double Inf_fl, NaN_fl;

/*
----------
__float128
----------
*/
#ifdef ENABLE_QUADPRECISION
extern __float128 Pi_f128, E_Nat_f128, Inf_f128, NaN_f128;
#endif /* ENABLE_QUADPRECISION */

extern uint8_t binary_number_of_1_8bit[256];

#endif /* _CONSTANT_H */
