/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/cnum.c
*/
#ifndef MADD_CNUM_H
#define MADD_CNUM_H

#include<math.h>
#include<stdbool.h>
#include"constant.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

typedef struct{
    double real, imag;
} Cnum;

typedef struct{
    float real, imag;
} Cnum32;

typedef struct{
    long double real, imag;
} Cnuml;

#ifdef ENABLE_QUADPRECISION
typedef struct{
    __float128 real, imag;
} Cnum128;
#endif /* ENABLE_QUADPRECISION */

#define CNUM_ADD__ALGORITHM(Cnum) \
{ \
    Cnum c; \
    c.real = a.real + b.real; \
    c.imag = a.imag + b.imag; \
    return c; \
} \

inline Cnum Cnum_Add(Cnum a, Cnum b)
CNUM_ADD__ALGORITHM(Cnum)

inline Cnum32 Cnum_Add_c32(Cnum32 a, Cnum32 b)
CNUM_ADD__ALGORITHM(Cnum32)

inline Cnuml Cnum_Add_cl(Cnuml a, Cnuml b)
CNUM_ADD__ALGORITHM(Cnuml)

#ifdef ENABLE_QUADPRECISION
inline Cnum128 Cnum_Add_c128(Cnum128 a, Cnum128 b)
CNUM_ADD__ALGORITHM(Cnum128)
#endif /* ENABLE_QUADPRECISION */

#define CNUM_SUB__ALGORITHM(Cnum) \
{ \
    Cnum c; \
    c.real = a.real - b.real; \
    c.imag = a.imag - b.imag; \
    return c; \
} \

inline Cnum Cnum_Sub(Cnum a, Cnum b)
CNUM_SUB__ALGORITHM(Cnum)

inline Cnum32 Cnum_Sub_c32(Cnum32 a, Cnum32 b)
CNUM_SUB__ALGORITHM(Cnum32)

 inline Cnuml Cnum_Sub_cl(Cnuml a, Cnuml b)
CNUM_SUB__ALGORITHM(Cnuml)

#ifdef ENABLE_QUADPRECISION
inline Cnum128 Cnum_Sub_c128(Cnum128 a, Cnum128 b)
CNUM_SUB__ALGORITHM(Cnum128)
#endif /* ENABLE_QUADPRECISION */

#define CNUM_MUL__ALGORITHM(Cnum) \
{ \
    Cnum c; \
    c.real = a.real * b.real - a.imag * b.imag; \
    c.imag = a.real * b.imag + a.imag * b.real; \
    return c; \
} \

inline Cnum Cnum_Mul(Cnum a, Cnum b)
CNUM_MUL__ALGORITHM(Cnum)

inline Cnum32 Cnum_Mul_c32(Cnum32 a, Cnum32 b)
CNUM_MUL__ALGORITHM(Cnum32)

inline Cnuml Cnum_Mul_cl(Cnuml a, Cnuml b)
CNUM_MUL__ALGORITHM(Cnuml)

#ifdef ENABLE_QUADPRECISION
inline Cnum128 Cnum_Mul_c128(Cnum128 a, Cnum128 b)
CNUM_MUL__ALGORITHM(Cnum128)
#endif /* ENABLE_QUADPRECISION */

#define CNUM_MUL_REAL_ALGORITHM(Cnum) \
{ \
    Cnum c; \
    c.real = a.real * b; \
    c.imag = a.imag * b; \
    return c; \
} \

inline Cnum Cnum_Mul_Real(Cnum a, double b)
CNUM_MUL_REAL_ALGORITHM(Cnum)

inline Cnum32 Cnum_Mul_Real_c32(Cnum32 a, float b)
CNUM_MUL_REAL_ALGORITHM(Cnum32)

inline Cnuml Cnum_Mul_Real_cl(Cnuml a, long double b)
CNUM_MUL_REAL_ALGORITHM(Cnuml)

#ifdef ENABLE_QUADPRECISION
inline Cnum128 Cnum_Mul_Real_c128(Cnum128 a, __float128 b)
CNUM_MUL_REAL_ALGORITHM(Cnum128)
#endif /* ENABLE_QUADPRECISION */

#define CNUM_DIV__ALGORITHM(Cnum, num_type) \
{ \
    num_type temp=b.real*b.real+b.imag*b.imag; \
    Cnum c; \
    c.real = (a.real*b.real + a.imag*b.imag)/temp; \
    c.imag = (a.imag*b.real - a.real*b.imag)/temp; \
    return c; \
} \

inline Cnum Cnum_Div(Cnum a, Cnum b)
CNUM_DIV__ALGORITHM(Cnum, double)

inline Cnum32 Cnum_Div_c32(Cnum32 a, Cnum32 b)
CNUM_DIV__ALGORITHM(Cnum32, float)

inline Cnuml Cnum_Div_cl(Cnuml a, Cnuml b)
CNUM_DIV__ALGORITHM(Cnuml, long double)

#ifdef ENABLE_QUADPRECISION
inline Cnum128 Cnum_Div_c128(Cnum128 a, Cnum128 b)
CNUM_DIV__ALGORITHM(Cnum128, __float128)
#endif /* ENABLE_QUADPRECISION */

#define CNUM_VALUE__ALGORITHM(Cnum) \
{ \
    Cnum num; \
    num.real = real; \
    num.imag = imag; \
    return num; \
} \

inline Cnum Cnum_Value(double real, double imag)
CNUM_VALUE__ALGORITHM(Cnum)

inline Cnum32 Cnum_Value_c32(float real, float imag)
CNUM_VALUE__ALGORITHM(Cnum32)

inline Cnuml Cnum_Value_cl(long double real, long double imag)
CNUM_VALUE__ALGORITHM(Cnuml)

#ifdef ENABLE_QUADPRECISION
inline Cnum128 Cnum_Value_c128(__float128 real, __float128 imag)
CNUM_VALUE__ALGORITHM(Cnum128)
#endif /* ENABLE_QUADPRECISION */

#define CNUM_IS_EQUAL__ALGORITHM(Cnum) \
{ \
    char flag_real = (a.real==b.real), flag_imag = (a.imag==b.imag); \
    return flag_real && flag_imag; \
} \

inline bool Cnum_Eq(Cnum a, Cnum b)
CNUM_IS_EQUAL__ALGORITHM(Cnum)

inline bool Cnum_Eq_c32(Cnum32 a, Cnum32 b)
CNUM_IS_EQUAL__ALGORITHM(Cnum32)

inline bool Cnum_Eq_cl(Cnuml a, Cnuml b)
CNUM_IS_EQUAL__ALGORITHM(Cnuml)

#ifdef ENABLE_QUADPRECISION
inline bool Cnum_Eq_c128(Cnum128 a, Cnum128 b)
CNUM_IS_EQUAL__ALGORITHM(Cnum128)
#endif /* ENABLE_QUADPRECISION */

#define CNUM_DIV_REAL__ALGORITHM(Cnum) \
{ \
    Cnum c; \
    c.real = a.real/b; \
    c.imag = a.imag/b; \
    return c; \
} \

inline Cnum Cnum_Div_Real(Cnum a, double b)
CNUM_DIV_REAL__ALGORITHM(Cnum)

inline Cnum32 Cnum_Div_Real_c32(Cnum32 a, float b)
CNUM_DIV_REAL__ALGORITHM(Cnum32)

inline Cnuml Cnum_Div_Real_cl(Cnuml a, long double b)
CNUM_DIV_REAL__ALGORITHM(Cnuml)

#ifdef ENABLE_QUADPRECISION
inline Cnum128 Cnum_Div_Real_c128(Cnum128 a, __float128 b)
CNUM_DIV_REAL__ALGORITHM(Cnum128)
#endif /* ENABLE_QUADPRECISION */

#define CNUM_REAL_DIV_CNUM__ALGORITHM(Cnum, num_type) \
{ \
    num_type temp = a / (b.real*b.real + b.imag*b.imag); \
    Cnum c; \
    c.real = temp * b.real; \
    c.imag = - temp * b.imag; \
    return c; \
} \

inline Cnum Real_Div_Cnum(double a, Cnum b)
CNUM_REAL_DIV_CNUM__ALGORITHM(Cnum, double)

inline Cnum32 Real_Div_Cnum_c32(float a, Cnum32 b)
CNUM_REAL_DIV_CNUM__ALGORITHM(Cnum32, float)

inline Cnuml Real_Div_Cnum_cl(long double a, Cnuml b)
CNUM_REAL_DIV_CNUM__ALGORITHM(Cnuml, long double)

#ifdef ENABLE_QUADPRECISION
inline Cnum128 Real_Div_Cnum_c128(__float128 a, Cnum128 b)
CNUM_REAL_DIV_CNUM__ALGORITHM(Cnum128, __float128)
#endif /* ENABLE_QUADPRECISION */

#define CNUM_MOD2__ALGORITHM(num_type) \
{ \
    num_type real2=a.real*a.real, imag2=a.imag*a.imag; \
    return real2+imag2; \
} \

inline double Cnum_Mod2(Cnum a)
CNUM_MOD2__ALGORITHM(double)

inline float Cnum_Mod2_c32(Cnum32 a)
CNUM_MOD2__ALGORITHM(float)

inline long double Cnum_Mod2_cl(Cnuml a)
CNUM_MOD2__ALGORITHM(long double)

#ifdef ENABLE_QUADPRECISION
inline __float128 Cnum_Mod2_c128(Cnum128 a)
CNUM_MOD2__ALGORITHM(__float128)
#endif /* ENABLE_QUADPRECISION */

/*
Cnum_Dot(x1 + x2 i, x3 + x4 i) => (x1 - x2 i) * (x3 + x4 i)
*/
#define CNUM_DOT__ALGORITHM(Cnum) \
{ \
    Cnum c; \
    c.real = a.real*b.real + a.imag*b.imag; \
    c.imag = a.real*b.imag - a.imag*b.real; \
    return c; \
} \

inline Cnum Cnum_Dot(Cnum a, Cnum b)
CNUM_DOT__ALGORITHM(Cnum)

inline Cnum32 Cnum_Dot_c32(Cnum32 a, Cnum32 b)
CNUM_DOT__ALGORITHM(Cnum32)

inline Cnuml Cnum_Dot_cl(Cnuml a, Cnuml b)
CNUM_DOT__ALGORITHM(Cnuml)

#ifdef ENABLE_QUADPRECISION
inline Cnum128 Cnum_Dot_c128(Cnum128 a, Cnum128 b)
CNUM_DOT__ALGORITHM(Cnum128)
#endif /* ENABLE_QUADPRECISION */

#define CNUM_CONJ__ALGORITHM(Cnum) \
{ \
    Cnum b; \
    b.real = a.real; \
    b.imag = - a.imag; \
    return b; \
} \

inline Cnum Cnum_Conj(Cnum a)
CNUM_CONJ__ALGORITHM(Cnum)

inline Cnum32 Cnum_Conj_c32(Cnum32 a)
CNUM_CONJ__ALGORITHM(Cnum32)

inline Cnuml Cnum_Conj_cl(Cnuml a)
CNUM_CONJ__ALGORITHM(Cnuml)

#ifdef ENABLE_QUADPRECISION
inline Cnum128 Cnum_Conj_c128(Cnum128 a)
CNUM_CONJ__ALGORITHM(Cnum128)
#endif /* ENABLE_QUADPRECISION */

#define CNUM_RADIUS__ALGORITHM(num_type, sqrt) \
{ \
    num_type radius2 = a.real*a.real + a.imag*a.imag; \
    return sqrt(radius2); \
} \

inline double Cnum_Radius(Cnum a)
CNUM_RADIUS__ALGORITHM(double, sqrt)

inline float Cnum_Radius_c32(Cnum32 a)
CNUM_RADIUS__ALGORITHM(float, sqrtf)

inline long double Cnum_Radius_cl(Cnuml a)
CNUM_RADIUS__ALGORITHM(long double, sqrtl)

#ifdef ENABLE_QUADPRECISION
inline __float128 Cnum_Radius_c128(Cnum128 a)
CNUM_RADIUS__ALGORITHM(__float128, sqrtq)
#endif /* ENABLE_QUADPRECISION */

#define CNUM_ANGLE__ALGORITHM(atan2) \
{ \
    return atan2(a.imag, a.real); \
} \

inline double Cnum_Angle(Cnum a)
CNUM_ANGLE__ALGORITHM(atan2)

inline float Cnum_Angle_c32(Cnum32 a)
CNUM_ANGLE__ALGORITHM(atan2f)

inline long double Cnum_Angle_cl(Cnuml a)
CNUM_ANGLE__ALGORITHM(atan2l)

#ifdef ENABLE_QUADPRECISION
inline __float128 Cnum_Angle_c128(Cnum128 a)
CNUM_ANGLE__ALGORITHM(atan2q)
#endif /* ENABLE_QUADPRECISION */

#define CNUM_POLE__ALGORITHM(Cnum, cos, sin) \
{ \
    Cnum cnum; \
    cnum.real = radius * cos(angle); \
    cnum.imag = radius * sin(angle); \
    return cnum; \
} \

inline Cnum Cnum_Pole(double radius, double angle)
CNUM_POLE__ALGORITHM(Cnum, cos, sin)

inline Cnum32 Cnum_Pole_c32(float radius, float angle)
CNUM_POLE__ALGORITHM(Cnum32, cosf, sinf)

inline Cnuml Cnum_Pole_cl(long double radius, long double angle)
CNUM_POLE__ALGORITHM(Cnuml, cosl, sinl)

#ifdef ENABLE_QUADPRECISION
inline Cnum128 Cnum_Pole_c128(__float128 radius, __float128 angle)
CNUM_POLE__ALGORITHM(Cnum128, cosq, sinq)
#endif /* ENABLE_QUADPRECISION */

#endif /* MADD_CNUM_H */