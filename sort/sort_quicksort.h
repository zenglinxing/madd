/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort_quicksort.h
*/
#ifndef _SORT_QUICKSORT_H
#define _SORT_QUICKSORT_H

#include<stdint.h>
#include<stdbool.h>

void Sort_Quicksort(uint64_t n_element, size_t usize, void *arr_, bool func_compare(void *a, void *b, void *other_param), void *other_param);

#endif /* _SORT_QUICKSORT_H */