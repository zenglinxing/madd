/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/file_endian.c
*/
#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<string.h>
#include"basic.h"

uint64_t madd_file_endian_buf_length = 1 << 24;

union _union8 Read_1byte(FILE *fp)
{
    union _union8 u8;
    fread(&u8, 1, 1, fp);
    return u8;
}

union _union16 Read_2byte_LE(FILE *fp)
{
    union _union16 u;
    fread(&u, 2, 1, fp);
    if (Endian_Type()){
        u = Byte_Reverse_16(u);
    }
    return u;
}

union _union16 Read_2byte_BE(FILE *fp)
{
    union _union16 u;
    fread(&u, 2, 1, fp);
    if (!Endian_Type()){
        u = Byte_Reverse_16(u);
    }
    return u;
}

union _union32 Read_4byte_LE(FILE *fp)
{
    union _union32 u;
    fread(&u, 4, 1, fp);
    if (Endian_Type()){
        u = Byte_Reverse_32(u);
    }
    return u;
}

union _union32 Read_4byte_BE(FILE *fp)
{
    union _union32 u;
    fread(&u, 4, 1, fp);
    if (!Endian_Type()){
        u = Byte_Reverse_32(u);
    }
    return u;
}

union _union64 Read_8byte_LE(FILE *fp)
{
    union _union64 u;
    fread(&u, 8, 1, fp);
    if (Endian_Type()){
        u = Byte_Reverse_64(u);
    }
    return u;
}

union _union64 Read_8byte_BE(FILE *fp)
{
    union _union64 u;
    fread(&u, 8, 1, fp);
    if (!Endian_Type()){
        u = Byte_Reverse_64(u);
    }
    return u;
}

void Write_1byte(FILE *fp, void *unit)
{
    fwrite(unit, 1, 1, fp);
}

void Write_2byte_LE(FILE *fp , void *unit)
{
    union _union16 ret=*((union _union16*)unit);
    if (Endian_Type()){
        ret = Byte_Reverse_16(ret);
    }
    fwrite(&ret, 2, 1, fp);
}

void Write_2byte_BE(FILE *fp , void *unit)
{
    union _union16 ret=*((union _union16*)unit);
    if (!Endian_Type()){
        ret = Byte_Reverse_16(ret);
    }
    fwrite(&ret, 2, 1, fp);
}

void Write_4byte_LE(FILE *fp , void *unit)
{
    union _union32 ret=*((union _union32*)unit);
    if (Endian_Type()){
        ret = Byte_Reverse_32(ret);
    }
    fwrite(&ret, 4, 1, fp);
}

void Write_4byte_BE(FILE *fp , void *unit)
{
    union _union32 ret=*((union _union32*)unit);
    if (!Endian_Type()){
        ret = Byte_Reverse_32(ret);
    }
    fwrite(&ret, 4, 1, fp);
}

void Write_8byte_LE(FILE *fp , void *unit)
{
    union _union64 ret=*((union _union64*)unit);
    if (Endian_Type()){
        ret = Byte_Reverse_64(ret);
    }
    fwrite(&ret, 8, 1, fp);
}

void Write_8byte_BE(FILE *fp , void *unit)
{
    union _union64 ret=*((union _union64*)unit);
    if (!Endian_Type()){
        ret = Byte_Reverse_64(ret);
    }
    fwrite(&ret, 8, 1, fp);
}

void Read_Array_LE(FILE *fp, void *buf_, size_t n_element, size_t element_size)
{
    bool endian_type=Endian_Type();
    size_t len = n_element * element_size;
    size_t real_buf_len = (madd_file_endian_buf_length/element_size)*element_size;
    size_t size1=element_size-1, size2=element_size>>1;
    unsigned char *buf=(unsigned char*)buf_, *temp_arr=(unsigned char*)malloc(real_buf_len), *p_element, temp_swap;
    uint64_t i_loop, n_loop=len/real_buf_len, n_rest=len%real_buf_len;
    uint64_t i_element, i_size;
    for (i_loop=0; i_loop<n_loop; i_loop++){
        fread(temp_arr, element_size, n_element, fp);
        if (endian_type && element_size){
            /* for each element */
            for (i_element=0,p_element=temp_arr; i_element<n_element; i_element++,p_element+=element_size){
                for (i_size=0; i_size<size2; i_size++){
                    temp_swap = p_element[i_size];
                    p_element[i_size] = p_element[size1-i_size];
                    p_element[size1-i_size] = temp_swap;
                }
            }
        }
        memcpy(buf+i_loop*real_buf_len, temp_arr, real_buf_len);
    }
    uint64_t n_rest_element=n_rest/element_size+(n_rest%element_size!=0);
    if (n_rest){
        memset(temp_arr, 0, real_buf_len);
        fread(temp_arr, 1, n_rest, fp);
        if (endian_type && element_size){
            for (i_element=0,p_element=temp_arr; i_element<n_rest_element; i_element++,p_element+=element_size){
                for (i_size=0; i_size<size2; i_size++){
                    temp_swap = p_element[i_size];
                    p_element[i_size] = p_element[size1-i_size];
                    p_element[size1-i_size] = temp_swap;
                }
            }
        }
        memcpy(buf+n_loop*real_buf_len, temp_arr, n_rest);
    }
    free(temp_arr);
}

void Read_Array_BE(FILE *fp, void *buf_, size_t n_element, size_t element_size)
{
    bool endian_type=Endian_Type();
    size_t len = n_element * element_size;
    size_t real_buf_len = (madd_file_endian_buf_length/element_size)*element_size;
    size_t size1=element_size-1, size2=element_size>>1;
    unsigned char *buf=(unsigned char*)buf_, *temp_arr=(unsigned char*)malloc(real_buf_len), *p_element, temp_swap;
    uint64_t i_loop, n_loop=len/real_buf_len, n_rest=len%real_buf_len;
    uint64_t i_element, i_size;
    for (i_loop=0; i_loop<n_loop; i_loop++){
        fread(temp_arr, element_size, n_element, fp);
        if (!endian_type && element_size){
            /* for each element */
            for (i_element=0,p_element=temp_arr; i_element<n_element; i_element++,p_element+=element_size){
                for (i_size=0; i_size<size2; i_size++){
                    temp_swap = p_element[i_size];
                    p_element[i_size] = p_element[size1-i_size];
                    p_element[size1-i_size] = temp_swap;
                }
            }
        }
        memcpy(buf+i_loop*real_buf_len, temp_arr, real_buf_len);
    }
    uint64_t n_rest_element=n_rest/element_size+(n_rest%element_size!=0);
    if (n_rest){
        memset(temp_arr, 0, real_buf_len);
        fread(temp_arr, 1, n_rest, fp);
        if (!endian_type && element_size){
            for (i_element=0,p_element=temp_arr; i_element<n_rest_element; i_element++,p_element+=element_size){
                for (i_size=0; i_size<size2; i_size++){
                    temp_swap = p_element[i_size];
                    p_element[i_size] = p_element[size1-i_size];
                    p_element[size1-i_size] = temp_swap;
                }
            }
        }
        memcpy(buf+n_loop*real_buf_len, temp_arr, n_rest);
    }
    free(temp_arr);
}

void Write_Array_LE(FILE *fp, void *buf_, size_t n_element, size_t element_size)
{
    bool endian_type=Endian_Type();
    size_t len = n_element * element_size;
    size_t real_buf_len = (madd_file_endian_buf_length/element_size)*element_size;
    size_t size1=element_size-1, size2=element_size>>1;
    unsigned char *buf=(unsigned char*)buf_, *temp_arr=(unsigned char*)malloc(real_buf_len), *p_element, temp_swap;
    uint64_t i_loop, n_loop=len/real_buf_len, n_rest=len%real_buf_len;
    uint64_t i_element, i_size;
    for (i_loop=0; i_loop<n_loop; i_loop++){
        memcpy(temp_arr, buf+i_loop*real_buf_len, real_buf_len);
        if (endian_type && element_size){
            /* for each element */
            for (i_element=0,p_element=temp_arr; i_element<n_element; i_element++,p_element+=element_size){
                for (i_size=0; i_size<size2; i_size++){
                    temp_swap = p_element[i_size];
                    p_element[i_size] = p_element[size1-i_size];
                    p_element[size1-i_size] = temp_swap;
                }
            }
        }
        fwrite(temp_arr, element_size, n_element, fp);
    }
    uint64_t n_rest_element=n_rest/element_size+(n_rest%element_size!=0);
    if (n_rest){
        memset(temp_arr, 0, real_buf_len);
        memcpy(temp_arr, buf+n_loop*real_buf_len, n_rest);
        if (endian_type && element_size){
            for (i_element=0,p_element=temp_arr; i_element<n_rest_element; i_element++,p_element+=element_size){
                for (i_size=0; i_size<size2; i_size++){
                    temp_swap = p_element[i_size];
                    p_element[i_size] = p_element[size1-i_size];
                    p_element[size1-i_size] = temp_swap;
                }
            }
        }
        fwrite(temp_arr, 1, n_rest, fp);
    }
    free(temp_arr);
}

void Write_Array_BE(FILE *fp, void *buf_, size_t n_element, size_t element_size)
{
    bool endian_type=Endian_Type();
    size_t len = n_element * element_size;
    size_t real_buf_len = (madd_file_endian_buf_length/element_size)*element_size;
    size_t size1=element_size-1, size2=element_size>>1;
    unsigned char *buf=(unsigned char*)buf_, *temp_arr=(unsigned char*)malloc(real_buf_len), *p_element, temp_swap;
    uint64_t i_loop, n_loop=len/real_buf_len, n_rest=len%real_buf_len;
    uint64_t i_element, i_size;
    for (i_loop=0; i_loop<n_loop; i_loop++){
        memcpy(temp_arr, buf+i_loop*real_buf_len, real_buf_len);
        if (!endian_type && element_size){
            /* for each element */
            for (i_element=0,p_element=temp_arr; i_element<n_element; i_element++,p_element+=element_size){
                for (i_size=0; i_size<size2; i_size++){
                    temp_swap = p_element[i_size];
                    p_element[i_size] = p_element[size1-i_size];
                    p_element[size1-i_size] = temp_swap;
                }
            }
        }
        fwrite(temp_arr, element_size, n_element, fp);
    }
    uint64_t n_rest_element=n_rest/element_size+(n_rest%element_size!=0);
    if (n_rest){
        memset(temp_arr, 0, real_buf_len);
        memcpy(temp_arr, buf+n_loop*real_buf_len, n_rest);
        if (!endian_type && element_size){
            for (i_element=0,p_element=temp_arr; i_element<n_rest_element; i_element++,p_element+=element_size){
                for (i_size=0; i_size<size2; i_size++){
                    temp_swap = p_element[i_size];
                    p_element[i_size] = p_element[size1-i_size];
                    p_element[size1-i_size] = temp_swap;
                }
            }
        }
        fwrite(temp_arr, 1, n_rest, fp);
    }
    free(temp_arr);
}