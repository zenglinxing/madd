/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort_counting.c
*/
#include<wchar.h>
#include<stdint.h>
#include<stdlib.h>
#include<string.h>
#include"sort.h"
#include"../basic/basic.h"

void Sort_Counting(uint64_t n_element, size_t usize, void *arr_,
                   uint64_t get_key(void *element, void *other_param), void *other_param)
{
    if (n_element < 2){
        Madd_Error_Add(MADD_WARNING, L"Sort_Counting: array length is less than 2, unnecessary to sort.");
        return;
    }
    if (usize == 0){
        Madd_Error_Add(MADD_ERROR, L"Sort_Counting: usize is 0.");
        return;
    }
    if (arr_ == NULL){
        Madd_Error_Add(MADD_ERROR, L"Sort_Counting: array pointer is NULL.");
        return;
    }

    wchar_t err_info[MADD_ERROR_INFO_LEN];
    uint64_t i_element, key, min_key=UINT64_MAX, max_key=0, n_key;
    unsigned char *arr=arr_, *parr;
    for (i_element=0,parr=arr; i_element<n_element; i_element++, parr+=usize){
        key = get_key(parr, other_param);
        if (key < min_key) min_key = key;
        if (key > max_key) max_key = key;
    }
    n_key = max_key - min_key + 1;

    uint64_t *counts=(uint64_t*)calloc(n_key, sizeof(uint64_t)), i_key;
    if (counts == NULL){
        swprintf(err_info, MADD_ERROR_INFO_LEN,
                 L"Sort_Counting: Failed to allocate %llu-count array (keys %llu-%llu)",
                 n_key, min_key, max_key);
        Madd_Error_Add(MADD_ERROR, err_info);
        return;
    }
    for (i_element=0,parr=arr; i_element<n_element; i_element++, parr+=usize){
        key = get_key(parr, other_param);
        counts[key - min_key] ++;
    }
    for (i_key=1; i_key<n_key; i_key++){
        counts[i_key] += counts[i_key-1];
    }

    unsigned char *arr_sorted=(unsigned char*)malloc(n_element*usize), *parr_sorted;
    if (arr_sorted == NULL){
        swprintf(err_info, MADD_ERROR_INFO_LEN, L"Sort_Counting: Failed to allocate %llu-element buffer (%zub each)",
                 n_element, usize);
        Madd_Error_Add(MADD_ERROR, err_info);
        free(counts);
        return;
    }
    for (i_element=0,parr=arr+(n_element-1)*usize; i_element<n_element; i_element++,parr-=usize){
        key = get_key(parr, other_param) - min_key;
        memcpy(arr_sorted+(counts[key]-1)*usize, parr, usize);
        counts[key] --;
    }

    memcpy(arr, arr_sorted, n_element*usize);

    free(counts);
    free(arr_sorted);
}