/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort_merge.c
*/
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include<stdbool.h>
#include"../basic/basic.h"

/* a general merge func for 2 arrays */
void Sort_Merge_merge_array(uint64_t n1, void *arr1_,
                            uint64_t n2, void *arr2_,
                            size_t usize, bool func_compare(void *a1, void *a2, void *other_param), void *other_param,
                            unsigned char *res)
{
    uint64_t i1=0, i2=0;
    unsigned char *p1=arr1_, *p2=arr2_, *pres=res;
    while (i1!=n1 && i2!=n2){
        if (func_compare((void*)p1, (void*)p2, other_param)){
            memcpy(pres, p1, usize);
            p1 += usize;
            i1 ++;
        }else{
            memcpy(pres, p2, usize);
            p2 += usize;
            i2 ++;
        }
        pres += usize;
    }
    if (i1 == n1) {
        memcpy(pres, p2, (n2-i2)*usize);
    } else {
        memcpy(pres, p1, (n1-i1)*usize);
    }
}

/*
Here we suppose arr2 is just after res in the memory
*/
static void Sort_Merge_merge(uint64_t n1, unsigned char *arr1,
                             uint64_t n2, unsigned char *arr2,
                             size_t usize, bool func_compare(void *a1, void *a2, void *other_param), void *other_param,
                             unsigned char *res)
{
    uint64_t i1=0, i2=0;
    unsigned char *p1=arr1, *p2=arr2, *pres=res;
    while (i1!=n1 && i2!=n2){
        if (func_compare((void*)p1, (void*)p2, other_param)){
            memcpy(pres, p1, usize);
            p1 += usize;
            i1 ++;
        }else{
            memcpy(pres, p2, usize);
            p2 += usize;
            i2 ++;
        }
        pres += usize;
    }
    if (i1 == n1) {
        memmove(pres, p2, (n2-i2)*usize);
    } else {
        memcpy(pres, p1, (n1-i1)*usize);
    }
}

void Sort_Merge(uint64_t n_element, size_t usize, void *arr_,
                bool func_compare(void *a1, void *a2, void *other_param), void *other_param)
{
    if (n_element < 2){
        Madd_Error_Add(MADD_WARNING, L"Sort_Merge: array length is less than 2, unnecessary to sort.");
        return;
    }
    if (usize == 0){
        Madd_Error_Add(MADD_ERROR, L"Sort_Merge: usize is 0.");
        return;
    }
    if (arr_ == NULL){
        Madd_Error_Add(MADD_ERROR, L"Sort_Merge: array pointer is NULL.");
        return;
    }

    unsigned char *arr=(unsigned char*)arr_, *arr_temp, *parr;
    uint64_t n1_element, log_n_element_low, log_n_element_high;
    Log2_Full(n_element, &log_n_element_low, &log_n_element_high);
    n1_element = (log_n_element_high==log_n_element_low) ? n_element >> 1 : 1 << log_n_element_low;

    arr_temp = (unsigned char*)malloc(n1_element*usize);
    if (arr_temp == NULL){
        wchar_t str_error[120];
        swprintf(str_error, 120, L"Sort_Merge: unable to allocate the temporary memory space -> %llux%llu.", n1_element, usize);
        Madd_Error_Add(MADD_ERROR, str_error);
        return;
    }

    uint64_t where_run=0, len_run=1, n_run, i_run, len2_run, len_rest;
    while (len_run < n_element){
        len_run <<= 1;
        n_run = n_element/len_run;
        len2_run = len_run >> 1;
        for (i_run=where_run=0,parr=arr; i_run<n_run; i_run++,where_run+=len_run,parr+=len_run*usize){
            memcpy(arr_temp, parr, len2_run*usize);
            Sort_Merge_merge(len2_run, arr_temp,
                             len2_run, parr + len2_run*usize,
                             usize, func_compare, other_param,
                             parr);
        }
        len_rest = n_element - where_run;
        if (len_rest > 0) {
            uint64_t first_len = len2_run;
            if (first_len > len_rest) {
                first_len = len_rest;
            }
            memcpy(arr_temp, parr, first_len * usize);
            Sort_Merge_merge(first_len, arr_temp,
                             len_rest - first_len, parr + first_len * usize,
                             usize, func_compare, other_param,
                             parr);
        }
    }

    free(arr_temp);
}