/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/basic.h

This file is aimed to collect majority of prevalent constants used in math and physics.
*/
#ifndef _BASIC_H
#define _BASIC_H

#include<stdio.h>
#include<stdint.h>
#include<stdbool.h>
#include<wchar.h>
#include<time.h>
#include"cnum.h"
#include"constant.h"
#include"cuda_base.cuh"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define MADD_LESS 0
#define MADD_SAME 1
#define MADD_GREATER 2

union _union8{
    uint8_t u;
    int8_t i;
};

union _union16{
    uint16_t u;
    int16_t i;
    uint8_t u8[2];
    int8_t i8[2];
};

union _union32{
    uint32_t u;
    int32_t i;
    float f;
    uint8_t u8[4];
    int8_t i8[4];
    uint16_t u16[2];
    uint16_t i16[2];
};

union _union64{
    uint64_t u;
    int64_t i;
    double f;
    uint8_t u8[8];
    int8_t i8[8];
    uint16_t u16[4];
    int16_t i16[4];
    uint32_t u32[2];
    int32_t i32[2];
    float f32[2];
};

/* binary_number.c */
uint8_t Binary_Number_of_1_8bit(union _union8 u8);
uint8_t Binary_Number_of_1_16bit(union _union16 u16);
uint8_t Binary_Number_of_1_32bit(union _union32 u32);
uint8_t Binary_Number_of_1_64bit(union _union64 u64);

/* log2_integer.c */
uint64_t Log2_Floor(uint64_t x);
uint64_t Log2_Ceil(uint64_t x);
void Log2_Full(uint64_t x, uint64_t *lower, uint64_t *upper);

/* norm.c */
double Norm1(uint64_t n, double *x);
float Norm1_f32(uint64_t n, float *x);
long double Norm1_fl(uint64_t n, long double *x);
double Norm1_c64(uint64_t n, Cnum *x);
float Norm1_c32(uint64_t n, Cnum_f32 *x);
long double Norm1_cl(uint64_t n, Cnum_fl *x);
#ifdef ENABLE_QUADPRECISION
__float128 Norm1_f128(uint64_t n, __float128 *x);
__float128 Norm1_c128(uint64_t n, Cnum_f128 *x);
#endif /* ENABLE_QUADPRECISION */
double Norm2(uint64_t n, double *x);
float Norm2_f32(uint64_t n, float *x);
long double Norm2_fl(uint64_t n, long double *x);
double Norm2_c64(uint64_t n, Cnum *x);
float Norm2_c32(uint64_t n, Cnum_f32 *x);
long double Norm2_cl(uint64_t n, Cnum_fl *x);
#ifdef ENABLE_QUADPRECISION
__float128 Norm2_f128(uint64_t n, __float128 *x);
__float128 Norm2_c128(uint64_t n, Cnum_f128 *x);
#endif /* ENABLE_QUADPRECISION */
double Norm_Infinity(uint64_t n, double *x);
float Norm_Infinity_f32(uint64_t n, float *x);
long double Norm_Infinity_fl(uint64_t n, long double *x);
double Norm_Infinity_c64(uint64_t n, Cnum *x);
float Norm_Infinity_c32(uint64_t n, Cnum_f32 *x);
long double Norm_Infinity_cl(uint64_t n, Cnum_fl *x);
#ifdef ENABLE_QUADPRECISION
__float128 Norm_Infinity_f128(uint64_t n, __float128 *x);
__float128 Norm_Infinity_c128(uint64_t n, Cnum_f128 *x);
#endif /* ENABLE_QUADPRECISION */

/* bit reverse*/
inline union _union8 Bit_Reverse_8(union _union8 x)
{
    uint8_t y = x.u;
    union _union8 ret;
    y = ((y>>1) & 0x55) | ((y&0x55) << 1); /* swap 1 bit */
    y = ((y>>2) & 0x33) | ((y&0x33) << 2); /* swap 2 bit */
    y = ((y>>4) & 0x0f) | ((y&0x0f) << 4); /* swap 4 bit */
    ret.u = y;
    return ret;
}

inline union _union16 Bit_Reverse_16(union _union16 x)
{
    uint16_t y = x.u;
    union _union16 ret;
    y = ((y>>1) & 0x5555) | ((y&0x5555) << 1); /* swap 1 bit */
    y = ((y>>2) & 0x3333) | ((y&0x3333) << 2); /* swap 2 bit */
    y = ((y>>4) & 0x0f0f) | ((y&0x0f0f) << 4); /* swap 4 bit */
    y = ((y>>8) & 0x00ff) | ((y&0x00ff) << 8); /* swap 8 bit */
    ret.u = y;
    return ret;
}

inline union _union32 Bit_Reverse_32(union _union32 x)
{
    uint32_t y = x.u;
    union _union32 ret;
    y = ((y>>1)  & 0x55555555) | ((y&0x55555555) << 1); /* swap 1 bit */
    y = ((y>>2)  & 0x33333333) | ((y&0x33333333) << 2); /* swap 2 bit */
    y = ((y>>4)  & 0x0f0f0f0f) | ((y&0x0f0f0f0f) << 4); /* swap 4 bit */
    y = ((y>>8)  & 0x00ff00ff) | ((y&0x00ff00ff) << 8); /* swap 8 bit */
    y = ((y>>16) & 0x0000ffff) | ((y&0x0000ffff) << 16); /* swap 16 bit */
    ret.u = y;
    return ret;
}

inline union _union64 Bit_Reverse_64(union _union64 x)
{
    uint64_t y = x.u;
    union _union64 ret;
    y = ((y>>1)  & 0x5555555555555555L) | ((y&0x5555555555555555L) << 1); /* swap 1 bit */
    y = ((y>>2)  & 0x3333333333333333L) | ((y&0x3333333333333333L) << 2); /* swap 2 bit */
    y = ((y>>4)  & 0x0f0f0f0f0f0f0f0fL) | ((y&0x0f0f0f0f0f0f0f0fL) << 4); /* swap 4 bit */
    y = ((y>>8)  & 0x00ff00ff00ff00ffL) | ((y&0x00ff00ff00ff00ffL) << 8); /* swap 8 bit */
    y = ((y>>16) & 0x0000ffff0000ffffL) | ((y&0x0000ffff0000ffffL) << 16); /* swap 16 bit */
    y = ((y>>32) & 0x00000000ffffffffL) | ((y&0x00000000ffffffffL) << 32); /* swap 32 bit */
    ret.u = y;
    return ret;
}

/* endian type */
inline bool Endian_Type(void)
{
    union _union32 u32;
    u32.u=0x00010000;
    return u32.u16[1] == 0;
}

/* byte_reverse.c */
union _union16 Byte_Reverse_16(union _union16 u);
union _union32 Byte_Reverse_32(union _union32 u);
union _union64 Byte_Reverse_64(union _union64 u);
void Byte_Reverse_Allocated(uint64_t n, void *arr, void *narr);
void *Byte_Reverse(uint64_t n, void *arr);

/* file_endian.c */
extern uint64_t madd_file_endian_buf_length;

union _union8 Read_1byte(FILE *fp);
union _union16 Read_2byte_LE(FILE *fp);
union _union16 Read_2byte_BE(FILE *fp);
union _union32 Read_4byte_LE(FILE *fp);
union _union32 Read_4byte_BE(FILE *fp);
union _union64 Read_8byte_LE(FILE *fp);
union _union64 Read_8byte_BE(FILE *fp);
void Write_1byte(FILE *fp, void *unit);
void Write_2byte_LE(FILE *fp , void *unit);
void Write_2byte_BE(FILE *fp , void *unit);
void Write_4byte_LE(FILE *fp , void *unit);
void Write_4byte_BE(FILE *fp , void *unit);
void Write_8byte_LE(FILE *fp , void *unit);
void Write_8byte_BE(FILE *fp , void *unit);
void Read_Array_LE(FILE *fp, void *buf_, size_t n_element, size_t element_size);
void Read_Array_BE(FILE *fp, void *buf_, size_t n_element, size_t element_size);
void Write_Array_LE(FILE *fp, void *buf_, size_t n_element, size_t element_size);
void Write_Array_BE(FILE *fp, void *buf_, size_t n_element, size_t element_size);

/* error_info.c */
#define MADD_SUCCESS 0
#define MADD_ERROR 1
#define MADD_WARNING 2

#define MADD_ERROR_MAX 20
#define MADD_ERROR_INFO_LEN 1024

typedef struct{
    char sign;
    time_t time_stamp;
    wchar_t info[MADD_ERROR_INFO_LEN];
} Madd_Error_Item;

typedef struct{
    bool flag_n_exceed;
    uint64_t n, n_error, n_warning;
    Madd_Error_Item item[MADD_ERROR_MAX];
} Madd_Error;

extern bool madd_error_keep_print, madd_error_print_wide, madd_error_save_wide;
extern Madd_Error madd_error;
extern uint64_t madd_error_n;

bool Madd_Error_Enable_Logfile(const char *log_file_name);
void Madd_Error_Set_Logfile(FILE *fp);
void Madd_Error_Close_Logfile(void);
void Madd_Error_Add(char sign, const wchar_t *info);
void Madd_Error_Print_Last(void);
void Madd_Error_Save_Last(FILE *fp);
void Madd_Error_Print_All(void);
void Madd_Error_Save_All(FILE *fp);
char Madd_Error_Get_Last(Madd_Error_Item *mei);

/* time_stamp.c */
#define MADD_TIME_STAMP_STRING_LEN 100

void Time_Stamp_String(time_t t, wchar_t *str);

#endif /* _BASIC_H */