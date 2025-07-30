/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./large_number/large_uint128.c
*/
#include<stdint.h>
#include<string.h>
#include<stdbool.h>
#include"large_uint.h"
#include"../basic/basic.h"
/*#include"../basic/constant.h"*/

Uint128_LE Bin128 = {.u64={BIN64, BIN64}},
           Bin128_Zero = {.u64={0, 0}},
           Bin128_One = {.u32={1, 0, 0, 0}}
;

Uint128_LE Uint128_And(Uint128_LE n1, Uint128_LE n2)
{
    Uint128_LE res;
    res.u64[0] = n1.u64[0] & n2.u64[0];
    res.u64[1] = n1.u64[1] & n2.u64[1];
    return res;
}

Uint128_LE Uint128_Or(Uint128_LE n1, Uint128_LE n2)
{
    Uint128_LE res;
    res.u64[0] = n1.u64[0] | n2.u64[0];
    res.u64[1] = n1.u64[1] | n2.u64[1];
    return res;
}

Uint128_LE Uint128_Xor(Uint128_LE n1, Uint128_LE n2)
{
    Uint128_LE res;
    res.u64[0] = n1.u64[0] ^ n2.u64[0];
    res.u64[1] = n1.u64[1] ^ n2.u64[1];
    return res;
}

Uint128_LE Uint128_Not(Uint128_LE n)
{
    Uint128_LE res;
    res.u64[0] = ~n.u64[0];
    res.u64[1] = ~n.u64[1];
    return res;
}

Uint128_LE UInt128_Left_Shift(Uint128_LE n, uint64_t shift)
{
    uint64_t word_shift = shift >> 5;
    uint64_t bit_shift = shift & 0x1F;
    Uint128_LE res = Bin128_Zero;
    
    if (word_shift >= 4) return res;
    for (int i = 0; i < 4 - word_shift; i++) {
        uint32_t val = n.u32[i] << bit_shift;
        if (i < 3 && bit_shift != 0) {
            val |= n.u32[i+1] >> (32 - bit_shift);
        }
        res.u32[i + word_shift] = val;
    }
    return res;
}

Uint128_LE UInt128_Right_Shift(Uint128_LE n, uint64_t shift)
{
    uint64_t word_shift = shift >> 5;
    uint64_t bit_shift = shift & 0x1F;
    Uint128_LE res = Bin128_Zero;
    if (word_shift >= 4) {
        return res;
    }
    for (int i = 0; i < 4 - word_shift; i++) {
        uint32_t current = n.u32[i + word_shift];
        uint32_t val = current >> bit_shift;
        if (bit_shift != 0 && (i + word_shift + 1) < 4) {
            uint32_t next = n.u32[i + word_shift + 1];
            val |= next << (32 - bit_shift);
        }
        res.u32[i] = val;
    }
    return res;
}

bool Uint128_Eq(Uint128_LE n1, Uint128_LE n2)
{
    uint64_t i;
    for (i=0; i<2; i++){
        if (n1.u64[i] != n2.u64[i]) return false;
    }
    return true;
}

bool Uint128_Ge(Uint128_LE n1, Uint128_LE n2) /* > */
{
    uint64_t hi1 = (uint64_t)n1.u32[3] << 32 | n1.u32[2];
    uint64_t hi2 = (uint64_t)n2.u32[3] << 32 | n2.u32[2];
    if (hi1 != hi2) return hi1 > hi2;
    
    uint64_t lo1 = (uint64_t)n1.u32[1] << 32 | n1.u32[0];
    uint64_t lo2 = (uint64_t)n2.u32[1] << 32 | n2.u32[0];
    return lo1 > lo2;
}

bool Uint128_Geq(Uint128_LE n1, Uint128_LE n2) /* >= */
{
    uint64_t hi1 = (uint64_t)n1.u32[3] << 32 | n1.u32[2];
    uint64_t hi2 = (uint64_t)n2.u32[3] << 32 | n2.u32[2];
    if (hi1 != hi2) return hi1 > hi2;
    
    uint64_t lo1 = (uint64_t)n1.u32[1] << 32 | n1.u32[0];
    uint64_t lo2 = (uint64_t)n2.u32[1] << 32 | n2.u32[0];
    return lo1 >= lo2;
}

bool Uint128_Le(Uint128_LE n1, Uint128_LE n2) /* < */
{
    uint64_t hi1 = (uint64_t)n1.u32[3] << 32 | n1.u32[2];
    uint64_t hi2 = (uint64_t)n2.u32[3] << 32 | n2.u32[2];
    if (hi1 != hi2) return hi1 < hi2;
    
    uint64_t lo1 = (uint64_t)n1.u32[1] << 32 | n1.u32[0];
    uint64_t lo2 = (uint64_t)n2.u32[1] << 32 | n2.u32[0];
    return lo1 < lo2;
}

bool Uint128_Leq(Uint128_LE n1, Uint128_LE n2) /* <= */
{
    uint64_t hi1 = (uint64_t)n1.u32[3] << 32 | n1.u32[2];
    uint64_t hi2 = (uint64_t)n2.u32[3] << 32 | n2.u32[2];
    if (hi1 != hi2) return hi1 < hi2;
    
    uint64_t lo1 = (uint64_t)n1.u32[1] << 32 | n1.u32[0];
    uint64_t lo2 = (uint64_t)n2.u32[1] << 32 | n2.u32[0];
    return lo1 <= lo2;
}

Uint128_LE Uint128_Add(Uint128_LE n1, Uint128_LE n2)
{
    int i;
    uint64_t carry=0, a1, a2, b;
    Uint128_LE res=Bin128_Zero;
    for (i=0; i<4; i+=2){
        a1 = ( (uint64_t)n1.u32[i] ) | ( (uint64_t)n1.u32[i+1] << 32 );
        a2 = ( (uint64_t)n2.u32[i] ) | ( (uint64_t)n2.u32[i+1] << 32 );
        b = a1 + a2 + carry;
        carry = (b < a1 || (/*b==a1 &&*/ carry && a2==BIN64)) ? 1 : 0;
        res.u32[i] = b & BIN32;
        res.u32[i+1] = b >> 32;
    }
    return res;
}

Uint256_LE Uint128_Add_256(Uint128_LE n1, Uint128_LE n2)
{
    int i;
    uint64_t carry=0, a1, a2, b;
    Uint256_LE res=Bin256_Zero;
    for (i=0; i<4; i+=2){
        a1 = ( (uint64_t)n1.u32[i] ) | ( (uint64_t)n1.u32[i+1] << 32 );
        a2 = ( (uint64_t)n2.u32[i] ) | ( (uint64_t)n2.u32[i+1] << 32 );
        b = a1 + a2 + carry;
        carry = (b < a1 || (/*b==a1 &&*/ carry && a2==BIN64)) ? 1 : 0;
        res.u32[i] = b & BIN32;
        res.u32[i+1] = b >> 32;
    }
    res.u32[i] = carry;
    return res;
}

Uint128_LE Uint128_Sub(Uint128_LE n1, Uint128_LE n2)
{
    uint64_t a1, a2, b, carry=0, i;
    Uint128_LE res=Bin128_Zero;
    for (i=0; i<4; i+=2){
        a1 = ( (uint64_t)n1.u32[i] ) | ( (uint64_t)n1.u32[i+1] << 32 );
        a2 = ( (uint64_t)n2.u32[i] ) | ( (uint64_t)n2.u32[i+1] << 32 );
        b = a1 - a2 - carry;
        carry = (b > a1 || (/*b==a1 &&*/ carry && a2==BIN64)) ? 1 : 0;
        res.u32[i] = b & BIN32;
        res.u32[i+1] = b >> 32;
    }
    return res;
}

Uint256_LE Uint128_Mul_256(Uint128_LE n1, Uint128_LE n2)
{
    uint64_t a1, a2, b, i1, i2, carry;
    uint32_t *p;
    Uint256_LE res=Bin256_Zero;
    for (i1=0; i1<4; i1++){
        a1 = n1.u32[i1];
        carry = 0;
        for (i2=0; i2<4; i2++){
            a2 = n2.u32[i2];
            b = a1 * a2 + carry + res.u32[i1+i2];
            res.u32[i1+i2] = b & BIN32;
            carry = b >> 32;
        }
        p = res.u32 + i1 + i2;
        while (carry){
            b = *p + carry;
            *p = b & BIN32;
            carry = b >> 32;
            p ++;
        }
    }
    return res;
}

Uint128_LE Uint128_Mul(Uint128_LE n1, Uint128_LE n2)
{
    uint64_t a1, a2, b, i1, i2, carry;
    Uint128_LE res=Bin128_Zero;
    for (i1=0; i1<4; i1++){
        a1 = n1.u32[i1];
        carry = 0;
        for (i2=0; i2<4-i1; i2++){
            a2 = n2.u32[i2];
            b = a1 * a2 + carry + res.u32[i1+i2];
            res.u32[i1+i2] = b & BIN32;
            carry = b >> 32;
        }
    }
    return res;
}

void Uint128_Log2(Uint128_LE num, uint64_t *log2_low, uint64_t *log2_high)
{
    bool flag_zero = true;
    uint8_t i_u;
    uint32_t u32;
    uint64_t offset_low, offset_high;
    for (i_u=0; i_u<4; i_u++){
        u32 = num.u32[4-1-i_u];
        if (u32){
            if (flag_zero){
                Log2_Full(u32, &offset_low, &offset_high);
                *log2_low = 32 * (4-1-i_u) + offset_low;
                if (offset_high != offset_low){
                    *log2_high = *log2_low + 1;
                    return;
                }
                flag_zero = false;
            }else{
                /* as flag_zero is false, it means that one higher .u32 is not zero */
                *log2_high = *log2_low + 1;
                return;
            }   
        }
    }
    if (flag_zero){
        *log2_low = *log2_high = 0;
        Madd_Error_Add(MADD_WARNING, L"Uint128_Log2: got input 0, ought to return -inf. The 2 results are 0 here.");
    }else{
        *log2_high = *log2_low;
    }
}