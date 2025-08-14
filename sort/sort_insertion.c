/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort_insertion.c
*/
#include<stdio.h>
#include<stdint.h>
#include<stdlib.h>
#include<string.h>
#include<stdbool.h>
#include"binary_search.h"
#include"../basic/basic.h"

void Sort_Insertion(uint64_t n_element, size_t usize, void *arr_,
                    bool func_compare(void *a1, void *a2, void *other_param), void *other_param)
{
    if (n_element < 2){
        Madd_Error_Add(MADD_WARNING, L"Sort_Insertion: array length is less than 2, unnecessary to sort.");
        return;
    }
    if (usize == 0){
        Madd_Error_Add(MADD_ERROR, L"Sort_Insertion: usize is 0.");
        return;
    }
    if (arr_ == NULL){
        Madd_Error_Add(MADD_ERROR, L"Sort_Insertion: array pointer is NULL.");
        return;
    }

    unsigned char stack_temp[1024];
    unsigned char *ptemp = (usize <= sizeof(stack_temp)) ? stack_temp : malloc(usize);
    unsigned char *arr=arr_, *parr;
    if (!ptemp && ptemp != stack_temp){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"Sort_Insertion: can't allocate %llu bytes for buffer.", usize);
        Madd_Error_Add(MADD_ERROR, error_info);
    }

    uint64_t i_element, where_insert, move_count;
    for (i_element=1,parr=arr+usize; i_element<n_element; i_element++,parr+=usize){
        where_insert = Binary_Search_Insert(i_element, usize, arr_, parr, func_compare, other_param);
        if (where_insert != i_element){
            memcpy(ptemp, parr, usize);
            move_count = i_element-where_insert;
            if (move_count){
                memmove(arr+(where_insert+1)*usize, arr+where_insert*usize, move_count*usize);
            }
            memcpy(arr+where_insert*usize, ptemp, usize);
        }
        
    }
    if (ptemp != stack_temp){
        free(ptemp);
    }
}