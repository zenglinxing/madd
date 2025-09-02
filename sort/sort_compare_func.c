/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort_compare_func.c
*/
#include<stdint.h>
#include<stdbool.h>
#include"../basic/basic.h"

/* less or equal */
#define SORT_LEQ__ALGORITHM(num_type) \
{ \
    return *(num_type*)a <= *(num_type*)b; \
} \

bool Sort_leq_f64(void *a, void *b, void *other_param)
SORT_LEQ__ALGORITHM(double)

bool Sort_leq_f32(void *a, void *b, void *other_param)
SORT_LEQ__ALGORITHM(float)

bool Sort_leq_fl(void *a, void *b, void *other_param)
SORT_LEQ__ALGORITHM(long double)

bool Sort_leq_i8(void *a, void *b, void *other_param)
SORT_LEQ__ALGORITHM(int8_t)

bool Sort_leq_u8(void *a, void *b, void *other_param)
SORT_LEQ__ALGORITHM(uint8_t)

bool Sort_leq_i16(void *a, void *b, void *other_param)
SORT_LEQ__ALGORITHM(int16_t)

bool Sort_leq_u16(void *a, void *b, void *other_param)
SORT_LEQ__ALGORITHM(uint16_t)

bool Sort_leq_i32(void *a, void *b, void *other_param)
SORT_LEQ__ALGORITHM(int32_t)

bool Sort_leq_u32(void *a, void *b, void *other_param)
SORT_LEQ__ALGORITHM(uint32_t)

bool Sort_leq_i64(void *a, void *b, void *other_param)
SORT_LEQ__ALGORITHM(int64_t)

bool Sort_leq_u64(void *a, void *b, void *other_param)
SORT_LEQ__ALGORITHM(uint64_t)

/* less */
#define SORT_LE__ALGORITHM(num_type) \
{ \
    return *(num_type*)a < *(num_type*)b; \
} \

bool Sort_le_f64(void *a, void *b, void *other_param)
SORT_LE__ALGORITHM(double)

bool Sort_le_f32(void *a, void *b, void *other_param)
SORT_LE__ALGORITHM(float)

bool Sort_le_fl(void *a, void *b, void *other_param)
SORT_LE__ALGORITHM(long double)

bool Sort_le_i8(void *a, void *b, void *other_param)
SORT_LE__ALGORITHM(int8_t)

bool Sort_le_u8(void *a, void *b, void *other_param)
SORT_LE__ALGORITHM(uint8_t)

bool Sort_le_i16(void *a, void *b, void *other_param)
SORT_LE__ALGORITHM(int16_t)

bool Sort_le_u16(void *a, void *b, void *other_param)
SORT_LE__ALGORITHM(uint16_t)

bool Sort_le_i32(void *a, void *b, void *other_param)
SORT_LE__ALGORITHM(int32_t)

bool Sort_le_u32(void *a, void *b, void *other_param)
SORT_LE__ALGORITHM(uint32_t)

bool Sort_le_i64(void *a, void *b, void *other_param)
SORT_LE__ALGORITHM(int64_t)

bool Sort_le_u64(void *a, void *b, void *other_param)
SORT_LE__ALGORITHM(uint64_t)

/* greater or equal */
#define SORT_GEQ__ALGORITHM(num_type) \
{ \
    return *(num_type*)a >= *(num_type*)b; \
} \

bool Sort_geq_f64(void *a, void *b, void *other_param)
SORT_GEQ__ALGORITHM(double)

bool Sort_geq_f32(void *a, void *b, void *other_param)
SORT_GEQ__ALGORITHM(float)

bool Sort_geq_fl(void *a, void *b, void *other_param)
SORT_GEQ__ALGORITHM(long double)

bool Sort_geq_i8(void *a, void *b, void *other_param)
SORT_GEQ__ALGORITHM(int8_t)

bool Sort_geq_u8(void *a, void *b, void *other_param)
SORT_GEQ__ALGORITHM(uint8_t)

bool Sort_geq_i16(void *a, void *b, void *other_param)
SORT_GEQ__ALGORITHM(int16_t)

bool Sort_geq_u16(void *a, void *b, void *other_param)
SORT_GEQ__ALGORITHM(uint16_t)

bool Sort_geq_i32(void *a, void *b, void *other_param)
SORT_GEQ__ALGORITHM(int32_t)

bool Sort_geq_u32(void *a, void *b, void *other_param)
SORT_GEQ__ALGORITHM(uint32_t)

bool Sort_geq_i64(void *a, void *b, void *other_param)
SORT_GEQ__ALGORITHM(int64_t)

bool Sort_geq_u64(void *a, void *b, void *other_param)
SORT_GEQ__ALGORITHM(uint64_t)

/* less */
#define SORT_GE__ALGORITHM(num_type) \
{ \
    return *(num_type*)a < *(num_type*)b; \
} \

bool Sort_ge_f64(void *a, void *b, void *other_param)
SORT_GE__ALGORITHM(double)

bool Sort_ge_f32(void *a, void *b, void *other_param)
SORT_GE__ALGORITHM(float)

bool Sort_ge_fl(void *a, void *b, void *other_param)
SORT_GE__ALGORITHM(long double)

bool Sort_ge_i8(void *a, void *b, void *other_param)
SORT_GE__ALGORITHM(int8_t)

bool Sort_ge_u8(void *a, void *b, void *other_param)
SORT_GE__ALGORITHM(uint8_t)

bool Sort_ge_i16(void *a, void *b, void *other_param)
SORT_GE__ALGORITHM(int16_t)

bool Sort_ge_u16(void *a, void *b, void *other_param)
SORT_GE__ALGORITHM(uint16_t)

bool Sort_ge_i32(void *a, void *b, void *other_param)
SORT_GE__ALGORITHM(int32_t)

bool Sort_ge_u32(void *a, void *b, void *other_param)
SORT_GE__ALGORITHM(uint32_t)

bool Sort_ge_i64(void *a, void *b, void *other_param)
SORT_GE__ALGORITHM(int64_t)

bool Sort_ge_u64(void *a, void *b, void *other_param)
SORT_GE__ALGORITHM(uint64_t)

/* compare */
#define SORT_COMPARE_ASCENDING__ALGORITHM(num_type) \
{ \
    double *aa=a, *bb=b; \
    if (*aa < *bb) return MADD_LESS; \
    else if (*aa > *bb) return MADD_GREATER; \
    else return MADD_SAME; \
} \

char Sort_Compare_Ascending_f64(void *a, void *b, void *other_param)
SORT_COMPARE_ASCENDING__ALGORITHM(double)

char Sort_Compare_Ascending_f32(void *a, void *b, void *other_param)
SORT_COMPARE_ASCENDING__ALGORITHM(float)

char Sort_Compare_Ascending_fl(void *a, void *b, void *other_param)
SORT_COMPARE_ASCENDING__ALGORITHM(long double)

char Sort_Compare_Ascending_i8(void *a, void *b, void *other_param)
SORT_COMPARE_ASCENDING__ALGORITHM(int8_t)

char Sort_Compare_Ascending_u8(void *a, void *b, void *other_param)
SORT_COMPARE_ASCENDING__ALGORITHM(uint8_t)

char Sort_Compare_Ascending_i16(void *a, void *b, void *other_param)
SORT_COMPARE_ASCENDING__ALGORITHM(int16_t)

char Sort_Compare_Ascending_u16(void *a, void *b, void *other_param)
SORT_COMPARE_ASCENDING__ALGORITHM(uint16_t)

char Sort_Compare_Ascending_i32(void *a, void *b, void *other_param)
SORT_COMPARE_ASCENDING__ALGORITHM(int32_t)

char Sort_Compare_Ascending_u32(void *a, void *b, void *other_param)
SORT_COMPARE_ASCENDING__ALGORITHM(uint32_t)

char Sort_Compare_Ascending_i64(void *a, void *b, void *other_param)
SORT_COMPARE_ASCENDING__ALGORITHM(int64_t)

char Sort_Compare_Ascending_u64(void *a, void *b, void *other_param)
SORT_COMPARE_ASCENDING__ALGORITHM(uint64_t)

#define SORT_COMPARE_DESCENDING__ALGORITHM(num_type) \
{ \
    double *aa=a, *bb=b; \
    if (*aa < *bb) return MADD_GREATER; \
    else if (*aa > *bb) return MADD_LESS; \
    else return MADD_SAME; \
} \

char Sort_Compare_Descending_f64(void *a, void *b, void *other_param)
SORT_COMPARE_DESCENDING__ALGORITHM(double)

char Sort_Compare_Descending_f32(void *a, void *b, void *other_param)
SORT_COMPARE_DESCENDING__ALGORITHM(float)

char Sort_Compare_Descending_fl(void *a, void *b, void *other_param)
SORT_COMPARE_DESCENDING__ALGORITHM(long double)

char Sort_Compare_Descending_i8(void *a, void *b, void *other_param)
SORT_COMPARE_DESCENDING__ALGORITHM(int8_t)

char Sort_Compare_Descending_u8(void *a, void *b, void *other_param)
SORT_COMPARE_DESCENDING__ALGORITHM(uint8_t)

char Sort_Compare_Descending_i16(void *a, void *b, void *other_param)
SORT_COMPARE_DESCENDING__ALGORITHM(int16_t)

char Sort_Compare_Descending_u16(void *a, void *b, void *other_param)
SORT_COMPARE_DESCENDING__ALGORITHM(uint16_t)

char Sort_Compare_Descending_i32(void *a, void *b, void *other_param)
SORT_COMPARE_DESCENDING__ALGORITHM(int32_t)

char Sort_Compare_Descending_u32(void *a, void *b, void *other_param)
SORT_COMPARE_DESCENDING__ALGORITHM(uint32_t)

char Sort_Compare_Descending_i64(void *a, void *b, void *other_param)
SORT_COMPARE_DESCENDING__ALGORITHM(int64_t)

char Sort_Compare_Descending_u64(void *a, void *b, void *other_param)
SORT_COMPARE_DESCENDING__ALGORITHM(uint64_t)