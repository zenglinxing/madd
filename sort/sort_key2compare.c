/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort_key2compare.c
*/
#include<stdint.h>
#include"sort.h"
#include"../basic/basic.h"

char Sort_Key_Func_to_Compare_Func(void *a, void *b, void *input_param)
{
    Sort_Key_Func_to_Compare_Func_Param *param = input_param;
    uint64_t a_value = param->get_key_func(a, param->other_param);
    uint64_t b_value = param->get_key_func(b, param->other_param);
    if (a_value < b_value) return MADD_LESS;
    else if (a_value > b_value) return MADD_GREATER;
    else return MADD_SAME;
}