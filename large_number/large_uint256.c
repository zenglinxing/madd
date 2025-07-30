/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./large_number/large_uint256.c
*/
#include<stdint.h>
#include<string.h>
#include<stdbool.h>
#include"large_uint.h"
#include"../basic/constant.h"

Uint256_LE Bin256 = {.u64={BIN64, BIN64, BIN64, BIN64}},
           Bin256_Zero = {.u64={0, 0, 0, 0}},
           Bin256_One = {.u32={1, 0, 0, 0, 0, 0, 0, 0}}
;