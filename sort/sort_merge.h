/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort_merge.h
*/
#ifndef _SORT_MERGE_H
#define _SORT_MERGE_H

#include<stdint.h>
#include<stdlib.h>

/* a general merge func for 2 arrays */
void Sort_Merge_merge_array(uint64_t n1, void *arr1_,
                            uint64_t n2, void *arr2_,
                            size_t usize, bool func_compare(void *a1, void *a2, void *other_param), void *other_param,
                            unsigned char *res);
void Sort_Merge(uint64_t n_element, size_t usize, void *arr_, bool func_compare(void *a1, void *a2, void *other_param), void *other_param);

#endif /* _SORT_MERGE_H */