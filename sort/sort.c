/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort.c
*/
#include<stdint.h>

#include"sort.h"
#include"../basic/basic.h"

void Sort(uint64_t n_element, size_t usize, void *arr,
          char func_compare(void *a1, void *a2, void *other_param), void *other_param)
{
    if (n_element < 2){
        Madd_Error_Add(MADD_WARNING, L"Sort_Merge: array length is less than 2, unnecessary to sort.");
        return;
    }
    if (usize == 0){
        Madd_Error_Add(MADD_ERROR, L"Sort_Merge: usize is 0.");
        return;
    }
    if (arr == NULL){
        Madd_Error_Add(MADD_ERROR, L"Sort_Merge: array pointer is NULL.");
        return;
    }

    if (n_element < 256){
        Sort_Insertion(n_element, usize, arr, func_compare, other_param);
    }else{
        Sort_Merge(n_element, usize, arr, func_compare, other_param);
    }
}