/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/binary_search.c
*/
#include<wchar.h>
#include<stdint.h>
#include<stdlib.h>
#include"binary_search.h"
#include"../basic/basic.h"

uint64_t Binary_Search(uint64_t n, size_t usize, void *arr_, void *element,
                       char func_compare(void *a, void *b, void *other_param), void *other_param)
{
    if (n == 0){
        Madd_Error_Add(MADD_WARNING, L"Binary_Search: array length is 0.");
        return 0;
    }
    uint64_t i1 = 0, i2 = n;
    unsigned char *arr = (unsigned char*)arr_;

    while (i1 < i2) {
        uint64_t i_middle = i1 + (i2 - i1) / 2;
        char compare = func_compare(arr + i_middle * usize, element, other_param);

        if (compare == MADD_LESS) {
            i1 = i_middle + 1; // 关键修复：向右缩小区间
        } else if (compare == MADD_GREATER) {
            i2 = i_middle;
        } else {
            return i_middle;
        }
    }

    // 未找到：返回第一个 ≥ 目标的位置 (i1)，并告警
    wchar_t err_info[MADD_ERROR_INFO_LEN];
    swprintf(err_info, MADD_ERROR_INFO_LEN, 
             L"Binary_Search: Element not found. Insertion point: %llu", i1);
    Madd_Error_Add(MADD_WARNING, err_info);
    return i1;
}

uint64_t Binary_Search_Insert(uint64_t n, size_t usize, void *arr_, void *element,
                              char func_compare(void *a, void *b, void *other_param), void *other_param)
{
    if (n == 0){
        return 0;
    }

    unsigned char *arr = (unsigned char*)arr_;
    
    // 修复1：首元素比较逻辑
    if (func_compare(element, arr, other_param) == MADD_LESS) {
        return 0;
    }
    
    // 修复2：尾元素比较逻辑
    if (func_compare(element, arr + (n-1)*usize, other_param) == MADD_GREATER) {
        return n;
    }

    uint64_t low = 0;
    uint64_t high = n - 1;  // 改为闭区间[0, n-1]

    while (low <= high) {
        uint64_t mid = low + (high - low) / 2;
        char cmp = func_compare(arr + mid*usize, element, other_param);
        
        if (cmp == MADD_LESS) {
            low = mid + 1;
        } else if (cmp == MADD_GREATER) {
            if (mid == 0) return 0;  // 边界保护
            high = mid - 1;
        } else {
            return mid;  // 找到相等元素
        }
    }
    return low;  // 返回插入位置
}