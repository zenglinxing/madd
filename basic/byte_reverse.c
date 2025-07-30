/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/byte_reverse.h
*/
#include<stdint.h>
#include<stdlib.h>
#include"basic.h"

#ifdef __x86_64__
#include<immintrin.h>
#endif

union _union16 Byte_Reverse_16(union _union16 u)
{
    uint16_t y = u.u;
    y = ((y & 0x00ff) << 8) | ((y >> 8) & 0x00ff);
    union _union16 ret={.u=y};
    return ret;
}

union _union32 Byte_Reverse_32(union _union32 u)
{
#if defined(__x86_64__) && !defined(__APPLE__)
    uint32_t y = _bswap(u.u);
#else
    uint32_t y = u.u;
    y = ((y & 0x00ff00ff) <<  8) | ((y >>  8) & 0x00ff00ff);
    y = ((y & 0x0000ffff) << 16) | ((y >> 16) & 0x0000ffff);
#endif
    union _union32 ret={.u=y};
    return ret;
}

union _union64 Byte_Reverse_64(union _union64 u)
{
#if defined(__x86_64__) && !defined(__APPLE__)
    uint64_t y = _bswap64(u.u);
#else
    uint64_t y = u.u;
    y = ((y & 0x00ff00ff00ff00ffL) <<  8) | ((y >>  8) & 0x00ff00ff00ff00ffL);
    y = ((y & 0x0000ffff0000ffffL) << 16) | ((y >> 16) & 0x0000ffff0000ffffL);
    y = ((y & 0x00000000ffffffffL) << 32) | ((y >> 32) & 0x00000000ffffffffL);
#endif
    union _union64 ret={.u=y};
    return ret;
}

void Byte_Reverse_Allocated(uint64_t n, void *arr, void *narr)
{
    if (n==0 || arr==NULL) return;
    unsigned char *parr=(uint8_t*)arr, *pnarr=(unsigned char*)narr;
    uint64_t i;
    for (i=0; i<n; i++){
        pnarr[i] = parr[n-1-i];
    }
}

void *Byte_Reverse(uint64_t n, void *arr)
{
    if (n==0 || arr==NULL) return NULL;
    void *narr=malloc(n);
    Byte_Reverse_Allocated(n, arr, narr);
    return narr;
}