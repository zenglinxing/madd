/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort_heap.c
*/
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include"sort.h"
#include"../basic/basic.h"

static inline void swap_elements(void *a, void *b, size_t usize, void *temp) {
    if (a == b) return;
    memcpy(temp, a, usize);
    memcpy(a, b, usize);
    memcpy(b, temp, usize);
}

static void heapify(size_t usize, void *arr_,
                    uint64_t start, uint64_t end,
                    char func_compare(void*, void*, void*), void *other_param,
                    void *ptemp)
{
    unsigned char *arr = (unsigned char*)arr_;
    uint64_t parent = start, child = parent * 2 + 1;
    while (child < end){
        if (child + 1 < end && func_compare(arr+child*usize, arr+(child+1)*usize, other_param) == MADD_LESS){
            child ++;
        }
        if (func_compare(arr+child*usize, arr+parent*usize, other_param) == MADD_LESS){
            return;
        }else{
            swap_elements(arr+parent*usize, arr+child*usize, usize, ptemp);
            parent = child;
            child = parent * 2 + 1;
        }
    }
}

void Sort_Heap_Internal(uint64_t n, size_t usize, void *arr_,
                        char func_compare(void*, void*, void*), void *other_param,
                        void *ptemp)
{
    if (n < 2){
        Madd_Error_Add(MADD_WARNING, L"Sort_Heap_Internal: array length is less than 2, unnecessary to sort.");
        return;
    }
    if (usize == 0){
        Madd_Error_Add(MADD_ERROR, L"Sort_Heap_Internal: usize is 0.");
        return;
    }
    if (arr_ == NULL){
        Madd_Error_Add(MADD_ERROR, L"Sort_Heap_Internal: array pointer is NULL.");
        return;
    }

    unsigned char *arr = (unsigned char*)arr_;
    uint64_t i;
    for (i=n/2-1; /*i>=0*/1; i--){
        heapify(usize, arr, i, n, func_compare, other_param, ptemp);
        if (i==0) break;
    }

    for (i=n-1; i>0; i--){
        swap_elements(arr, arr+i*usize, usize, ptemp);
        heapify(usize, arr, 0, i, func_compare, other_param, ptemp);
    }

}

void Sort_Heap(uint64_t n, size_t usize, void *arr_,
               char func_compare(void*, void*, void*), void *other_param)
{
    if (n < 2){
        Madd_Error_Add(MADD_WARNING, L"Sort_Heap: array length is less than 2, unnecessary to sort.");
        return;
    }
    if (usize == 0){
        Madd_Error_Add(MADD_ERROR, L"Sort_Heap: usize is 0.");
        return;
    }
    if (arr_ == NULL){
        Madd_Error_Add(MADD_ERROR, L"Sort_Heap: array pointer is NULL.");
        return;
    }

    unsigned char /* *arr = (unsigned char*)arr_, */ *ptemp;
    unsigned char temp_element[1024];
    ptemp = (usize > 1024) ? (unsigned char*)malloc(usize) : temp_element;

    Sort_Heap_Internal(n, usize, arr_, func_compare, other_param, ptemp);

    if (usize > 1024){
        free(ptemp);
    }
}