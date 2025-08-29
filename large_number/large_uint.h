/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./large_number/large_uint.h
*/
#ifndef MADD_LARGE_UINT_H
#define MADD_LARGE_UINT_H

#include<stdint.h>

typedef union{
    uint64_t u64[2];
    uint32_t u32[4];
    uint16_t u16[8];
    uint8_t  u8[16];
} Uint128_LE;

typedef union{
    uint64_t u64[4];
    uint32_t u32[8];
    uint16_t u16[16];
    uint8_t  u8[32];
} Uint256_LE;

extern Uint128_LE Bin128, Bin128_Zero, Bin128_One;
extern Uint256_LE Bin256, Bin256_Zero, Bin256_One;

Uint128_LE Uint128_And(Uint128_LE n1, Uint128_LE n2);
Uint128_LE Uint128_Or(Uint128_LE n1, Uint128_LE n2);
Uint128_LE Uint128_Xor(Uint128_LE n1, Uint128_LE n2);
Uint128_LE Uint128_Not(Uint128_LE n);
Uint128_LE UInt128_Left_Shift(Uint128_LE n, uint64_t shift);
Uint128_LE UInt128_Right_Shift(Uint128_LE n, uint64_t shift);
bool Uint128_Eq(Uint128_LE n1, Uint128_LE n2);
bool Uint128_Ge(Uint128_LE n1, Uint128_LE n2);
bool Uint128_Geq(Uint128_LE n1, Uint128_LE n2);
bool Uint128_Le(Uint128_LE n1, Uint128_LE n2);
bool Uint128_Leq(Uint128_LE n1, Uint128_LE n2);
Uint128_LE Uint128_Add(Uint128_LE n1, Uint128_LE n2);
Uint256_LE Uint128_Add_256(Uint128_LE n1, Uint128_LE n2);
Uint128_LE Uint128_Sub(Uint128_LE n1, Uint128_LE n2);
Uint256_LE Uint128_Mul_256(Uint128_LE n1, Uint128_LE n2);
Uint128_LE Uint128_Mul(Uint128_LE n1, Uint128_LE n2);
void Uint128_Log2(Uint128_LE num, uint64_t *log2_low, uint64_t *log2_high);

#endif /* MADD_LARGE_UINT_H */