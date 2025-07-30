/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort_counting.h
*/
#ifndef _SORT_COUNTING_H
#define _SORT_COUNTING_H

#include<wchar.h>
#include<stdint.h>
#include<stdlib.h>
#include<string.h>
#include<stdbool.h>
#include"../basic/basic.h"

void Sort_Counting(uint64_t n_element, size_t usize, void *arr_,
                   uint64_t get_key(void *element, void *other_param), void *other_param);

#endif /* _SORT_COUNTING_H */