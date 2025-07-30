/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/binary_search.c
*/
#include<wchar.h>
#include<stdint.h>
#include<stdlib.h>
#include<stdbool.h>
#include"../basic/basic.h"

uint64_t Binary_Search(uint64_t n, size_t usize, void *arr_, void *element,
                       char func_compare(void *a, void *b, void *other_param), void *other_param)
{
    if (n == 0){
        Madd_Error_Add(MADD_WARNING, L"Binary_Search: array length is 0.");
        return 0;
    }
    unsigned char *p1, *p2, *arr=(unsigned char*)arr_;
    char compare;
    uint64_t i1=0, i2=n, i_middle, i1_last=0, i2_last=n;
    wchar_t err_info[MADD_ERROR_INFO_LEN];
    p1 = arr;
    p2 = arr + i2*usize;
    i_middle = (i1 + i2) >> 1;
    compare = func_compare((void*)(arr+i_middle*usize), element, other_param);
    while (compare != MADD_SAME){
        i1_last = i1;
        i2_last = i2;
        if (compare == MADD_LESS){
            i1 = i_middle;
        }else if (compare == MADD_GREATER){
            i2 = i_middle;
        }else{
            swprintf(err_info, MADD_ERROR_INFO_LEN, L"Binary_Search: the given func_compare function returns unexpected value %x. Only expects MADD_LESS, MADD_SAME and MADD_GREATER.", compare);
            Madd_Error_Add(MADD_ERROR, err_info);
            return i_middle;
        }
        if (i1==i1_last && i2==i2_last){
            swprintf(err_info, MADD_ERROR_INFO_LEN, L"Binary_Search: the given element (0x%llx) could not be found in the given array (0x%llx).", element, arr_);
            Madd_Error_Add(MADD_ERROR, err_info);
            return i_middle;
        }
        i_middle = (i1 + i2) >> 1;
        compare = func_compare((void*)(arr+i_middle*usize), element, other_param);
    }
    return i_middle;
}

uint64_t Binary_Search_Insert(uint64_t n, size_t usize, void *arr_, void *element,
                              bool func_compare(void *a, void *b, void *other_param), void *other_param)
{
    if (n == 0){
        return 0;
    }

    unsigned char *p1, *p2, *arr=(unsigned char*)arr_;
    bool compare;
    uint64_t i1=0, i2=n, i_middle, i1_last=0, i2_last=n;
    /*wchar_t err_info[MADD_ERROR_INFO_LEN];*/
    p1 = arr;
    p2 = arr + i2*usize;

    if (func_compare(element, p1 /* first one */, other_param)) return 0;
    if (func_compare(p2-usize /* last one */, element, other_param)) return n;

    i_middle = (i1 + i2) >> 1;
    compare = func_compare((void*)(arr+i_middle*usize), element, other_param);
    while (1){
        i1_last = i1;
        i2_last = i2;
        if (compare){
            i1 = i_middle;
        }else{
            i2 = i_middle;
        }
        if (i1==i1_last && i2==i2_last){
            return i2;
        }
        i_middle = (i1 + i2) >> 1;
        compare = func_compare((void*)(arr+i_middle*usize), element, other_param);
    }
    return i_middle;
}