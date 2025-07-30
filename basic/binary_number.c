/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/binary_number.c

This file is aimed to collect majority of prevalent constants used in math and physics.
*/
#include"constant.h"
#include"basic.h"

#ifdef __x86_64__
#include<immintrin.h>
#endif

uint8_t Binary_Number_of_1_8bit(union _union8 u8)
{
#ifdef __x86_64__
#ifdef __APPLE__
    return _mm_popcnt_u32(u8.u);
#else
    return _popcnt32(u8.u);
#endif
#else
    return binary_number_of_1_8bit[u8.u];
#endif
}

uint8_t Binary_Number_of_1_16bit(union _union16 u16)
{
#ifdef __x86_64__
#ifdef __APPLE__
    return _mm_popcnt_u32(u16.u);
#else
    return _popcnt32(u16.u);
#endif
#else
    uint8_t n=0;
    n += binary_number_of_1_8bit[u16.u8[0]];
    n += binary_number_of_1_8bit[u16.u8[1]];
    return n;
#endif
}

uint8_t Binary_Number_of_1_32bit(union _union32 u32)
{
#ifdef __x86_64__
#ifdef __APPLE__
    return _mm_popcnt_u32(u32.u);
#else
    return _popcnt32(u32.u);
#endif
#else
    uint8_t n=0;
    n += binary_number_of_1_8bit[u32.u8[0]];
    n += binary_number_of_1_8bit[u32.u8[1]];
    n += binary_number_of_1_8bit[u32.u8[2]];
    n += binary_number_of_1_8bit[u32.u8[3]];
    return n;
#endif
}

uint8_t Binary_Number_of_1_64bit(union _union64 u64)
{
#ifdef __x86_64__
#ifdef __APPLE__
    return _mm_popcnt_u64(u64.u);
#else
    return _popcnt64(u64.u);
#endif
#else
    uint8_t n=0;
    n += binary_number_of_1_8bit[u64.u8[0]];
    n += binary_number_of_1_8bit[u64.u8[1]];
    n += binary_number_of_1_8bit[u64.u8[2]];
    n += binary_number_of_1_8bit[u64.u8[3]];
    n += binary_number_of_1_8bit[u64.u8[4]];
    n += binary_number_of_1_8bit[u64.u8[5]];
    n += binary_number_of_1_8bit[u64.u8[6]];
    n += binary_number_of_1_8bit[u64.u8[7]];
    return n;
#endif
}