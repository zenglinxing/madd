/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort.h
*/
#ifndef MADD_SORT_H
#define MADD_SORT_H

#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>

#include"binary_search.h"

typedef struct{
    uint64_t (*get_key_func)(void*, void*);
    void *other_param;
} Sort_Key_Func_to_Compare_Func_Param;

char Sort_Key_Func_to_Compare_Func(void *a, void *b, void *input_param);

void Sort_Counting(uint64_t n_element, size_t usize, void *arr_,
                   uint64_t get_key(void *element, void *other_param), void *other_param);

void Sort_Insertion(uint64_t n_element, size_t usize, void *arr_,
                    char func_compare(void *a1, void *a2, void *other_param), void *other_param);
void Sort_Merge_merge_array(uint64_t n1, void *arr1_,
                            uint64_t n2, void *arr2_,
                            size_t usize, char func_compare(void *a1, void *a2, void *other_param), void *other_param,
                            unsigned char *res);
void Sort_Merge(uint64_t n_element, size_t usize, void *arr_,
                char func_compare(void *a1, void *a2, void *other_param), void *other_param);
void Sort_Quicksort(uint64_t n_element, size_t usize, void *arr_,
                    char func_compare(void *a, void *b, void *other_param), void *other_param);
void Sort__Merge_Left(uint64_t n_left, uint64_t n_right, size_t usize,
                      void *arr_, void *temp_,
                      char func_compare(void *a, void *b, void *other_param), void *other_param);
void Sort__Merge_Right(uint64_t n_left, uint64_t n_right, size_t usize,
                       void *arr_, void *temp_,
                       char func_compare(void *a, void *b, void *other_param), void *other_param);
void Sort_Shell(uint64_t n_element, size_t usize, void *arr_,
                char func_compare(void*, void*, void*), void *other_param);
void Sort_Heap_Internal(uint64_t n, size_t usize, void *arr_,
                        char func_compare(void*, void*, void*), void *other_param,
                        void *ptemp);
void Sort_Heap(uint64_t n, size_t usize, void *arr_,
                    char func_compare(void*, void*, void*), void *other_param);

/* less or equal */
bool Sort_leq_f64(void *a, void *b, void *other_param);
bool Sort_leq_f32(void *a, void *b, void *other_param);
bool Sort_leq_fl(void *a, void *b, void *other_param);
bool Sort_leq_i8(void *a, void *b, void *other_param);
bool Sort_leq_u8(void *a, void *b, void *other_param);
bool Sort_leq_i16(void *a, void *b, void *other_param);
bool Sort_leq_u16(void *a, void *b, void *other_param);
bool Sort_leq_i32(void *a, void *b, void *other_param);
bool Sort_leq_u32(void *a, void *b, void *other_param);
bool Sort_leq_i64(void *a, void *b, void *other_param);
bool Sort_leq_u64(void *a, void *b, void *other_param);

/* less */
bool Sort_le_f64(void *a, void *b, void *other_param);
bool Sort_le_f32(void *a, void *b, void *other_param);
bool Sort_le_fl(void *a, void *b, void *other_param);
bool Sort_le_i8(void *a, void *b, void *other_param);
bool Sort_le_u8(void *a, void *b, void *other_param);
bool Sort_le_i16(void *a, void *b, void *other_param);
bool Sort_le_u16(void *a, void *b, void *other_param);
bool Sort_le_i32(void *a, void *b, void *other_param);
bool Sort_le_u32(void *a, void *b, void *other_param);
bool Sort_le_i64(void *a, void *b, void *other_param);
bool Sort_le_u64(void *a, void *b, void *other_param);

/* greater or equal */
bool Sort_geq_f64(void *a, void *b, void *other_param);
bool Sort_geq_f32(void *a, void *b, void *other_param);
bool Sort_geq_fl(void *a, void *b, void *other_param);
bool Sort_geq_i8(void *a, void *b, void *other_param);
bool Sort_geq_u8(void *a, void *b, void *other_param);
bool Sort_geq_i16(void *a, void *b, void *other_param);
bool Sort_geq_u16(void *a, void *b, void *other_param);
bool Sort_geq_i32(void *a, void *b, void *other_param);
bool Sort_geq_u32(void *a, void *b, void *other_param);
bool Sort_geq_i64(void *a, void *b, void *other_param);
bool Sort_geq_u64(void *a, void *b, void *other_param);

/* less */
bool Sort_ge_f64(void *a, void *b, void *other_param);
bool Sort_ge_f32(void *a, void *b, void *other_param);
bool Sort_ge_fl(void *a, void *b, void *other_param);
bool Sort_ge_i8(void *a, void *b, void *other_param);
bool Sort_ge_u8(void *a, void *b, void *other_param);
bool Sort_ge_i16(void *a, void *b, void *other_param);
bool Sort_ge_u16(void *a, void *b, void *other_param);
bool Sort_ge_i32(void *a, void *b, void *other_param);
bool Sort_ge_u32(void *a, void *b, void *other_param);
bool Sort_ge_i64(void *a, void *b, void *other_param);
bool Sort_ge_u64(void *a, void *b, void *other_param);

/* compare */
char Sort_Compare_Ascending_f64(void *a, void *b, void *other_param);
char Sort_Compare_Ascending_f32(void *a, void *b, void *other_param);
char Sort_Compare_Ascending_fl(void *a, void *b, void *other_param);
char Sort_Compare_Ascending_i8(void *a, void *b, void *other_param);
char Sort_Compare_Ascending_u8(void *a, void *b, void *other_param);
char Sort_Compare_Ascending_i16(void *a, void *b, void *other_param);
char Sort_Compare_Ascending_u16(void *a, void *b, void *other_param);
char Sort_Compare_Ascending_i32(void *a, void *b, void *other_param);
char Sort_Compare_Ascending_u32(void *a, void *b, void *other_param);
char Sort_Compare_Ascending_i64(void *a, void *b, void *other_param);
char Sort_Compare_Ascending_u64(void *a, void *b, void *other_param);

char Sort_Compare_Descending_f64(void *a, void *b, void *other_param);
char Sort_Compare_Descending_f32(void *a, void *b, void *other_param);
char Sort_Compare_Descending_fl(void *a, void *b, void *other_param);
char Sort_Compare_Descending_i8(void *a, void *b, void *other_param);
char Sort_Compare_Descending_u8(void *a, void *b, void *other_param);
char Sort_Compare_Descending_i16(void *a, void *b, void *other_param);
char Sort_Compare_Descending_u16(void *a, void *b, void *other_param);
char Sort_Compare_Descending_i32(void *a, void *b, void *other_param);
char Sort_Compare_Descending_u32(void *a, void *b, void *other_param);
char Sort_Compare_Descending_i64(void *a, void *b, void *other_param);
char Sort_Compare_Descending_u64(void *a, void *b, void *other_param);

#endif /* MADD_SORT_H */