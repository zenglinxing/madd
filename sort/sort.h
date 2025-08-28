/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort.h
*/
#ifndef _SORT_H
#define _SORT_H

#include"binary_search.h"

typedef struct{
    uint64_t (*get_key_func)(void*, void*);
    void *other_param;
} Sort_Key_Func_to_Compare_Func_Param;

bool Sort_Key_Func_to_Compare_Func(void *a, void *b, void *input_param);

void Sort_Counting(uint64_t n_element, size_t usize, void *arr_,
                   uint64_t get_key(void *element, void *other_param), void *other_param);

void Sort_Insertion(uint64_t n_element, size_t usize, void *arr_,
                    bool func_compare(void *a1, void *a2, void *other_param), void *other_param);
void Sort_Merge_merge_array(uint64_t n1, void *arr1_,
                            uint64_t n2, void *arr2_,
                            size_t usize, bool func_compare(void *a1, void *a2, void *other_param), void *other_param,
                            unsigned char *res);
void Sort_Merge(uint64_t n_element, size_t usize, void *arr_,
                bool func_compare(void *a1, void *a2, void *other_param), void *other_param);
void Sort_Quicksort(uint64_t n_element, size_t usize, void *arr_,
                    bool func_compare(void *a, void *b, void *other_param), void *other_param);
void Sort__Merge_Left(uint64_t n_left, uint64_t n_right, size_t usize,
                      void *arr_, void *temp_,
                      bool func_compare(void *a, void *b, void *other_param), void *other_param);
void Sort__Merge_Right(uint64_t n_left, uint64_t n_right, size_t usize,
                       void *arr_, void *temp_,
                       bool func_compare(void *a, void *b, void *other_param), void *other_param);
void Sort_Shell(uint64_t n_element, size_t usize, void *arr_,
                bool func_compare(void*, void*, void*), void *other_param);
void Sort_Heap_Internal(uint64_t n, size_t usize, void *arr_,
                        bool func_compare(void*, void*, void*), void *other_param,
                        void *ptemp);
void Sort_Heap(uint64_t n, size_t usize, void *arr_,
                    bool func_compare(void*, void*, void*), void *other_param);

#endif /* _SORT_H */