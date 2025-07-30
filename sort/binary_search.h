/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/binary_search.h
*/
#ifndef _BINARY_SEARCH_H
#define _BINARY_SEARCH_H

#include<stdint.h>
#include<stdlib.h>

uint64_t Binary_Search(uint64_t n, size_t usize, void *arr_, void *element,
                       char func_compare(void *a, void *b, void *other_param), void *other_param);
uint64_t Binary_Search_Insert(uint64_t n, size_t usize, void *arr_, void *element,
                              bool func_compare(void *a, void *b, void *other_param), void *other_param);

#endif /* _BINARY_SEARCH_H */