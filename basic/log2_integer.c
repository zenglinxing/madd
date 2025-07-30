/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/log2_integer.c
*/
#include<stdint.h>
#include"constant.h"

uint64_t Log2_Floor(uint64_t x)
{
    uint64_t mask=Bin32, shift=32, x1=x, x2, i, res=0;
    for (i=0; i<6; i++){
        x2 = x1 >> shift;
        if (x2){
            res += shift;
            x1 = x2;
        }
        else{
            /*x1 &= mask;*/
        }
        shift >>= 1;
        mask >>= shift;
    }
    return res;
}

uint64_t Log2_Ceil(uint64_t x)
{
    uint64_t res_floor=Log2_Floor(x), x_shifted=x<<(64-res_floor), res=res_floor;
    if (x==1);
    else if (x_shifted) res++;
    return res;
}

void Log2_Full(uint64_t x, uint64_t *lower, uint64_t *upper)
{
    *lower = *upper = Log2_Floor(x);
    uint64_t x_shifted=x<<(64 - *lower);
    if (x==1);
    else if (x_shifted) (*upper) ++;
}