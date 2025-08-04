/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort_quicksort.c
*/
#include<stdint.h>
#include<stdlib.h>
#include<string.h>
#include<stdbool.h>
#include"sort.h"
#include"../data_struct/data_struct.h"
#include"../basic/basic.h"

typedef bool (*func_compare_t)(void*, void*, void*);

typedef struct{
    uint64_t len;
    size_t usize;
    unsigned char *arr;
    func_compare_t func_compare;
    void *other_param;
} Quicksort_Param;

static bool Sort_Quicksort_Internal(Stack *stack, void *pivot)
{
    Quicksort_Param qp;
    bool res_pop = Stack_Pop(stack, &qp);
    if (!res_pop){
        Madd_Error_Add(MADD_ERROR, L"Sort_Quicksort_Internal: unable to get the next array to sort. See info from Stack_Pop.");
        return false;
    }
    if (qp.len < 2) return true;
    if (qp.len <= 16){
        Sort_Insertion(qp.len, qp.usize, qp.arr, qp.func_compare, qp.other_param);
        return true;
    }

    bool left_turn = false;
    unsigned char *left=qp.arr, *right=qp.arr+(qp.len-1)*qp.usize;
    uint64_t i_left = 0, i_right = qp.len - 1;
    memcpy(pivot, left, qp.usize);

    while (i_left != i_right){
        if (left_turn){
            if (qp.func_compare(left, pivot, qp.other_param)){
                left += qp.usize;
                i_left ++;
            }else{
                memcpy(right, left, qp.usize);
                left_turn = false;
            }
        }else{
            if (qp.func_compare(pivot, right, qp.other_param)){
                right -= qp.usize;
                i_right --;
            }else{
                memcpy(left, right, qp.usize);
                left_turn = true;
            }
        }
    }
    memcpy(left, pivot, qp.usize);

    uint64_t len_left = i_left, len_right = qp.len - i_left - 1;
    Quicksort_Param left_array={.len=len_left, .usize=qp.usize, .arr=qp.arr, .func_compare=qp.func_compare, .other_param=qp.other_param};
    Quicksort_Param right_array={.len=len_right, .usize=qp.usize, .arr=qp.arr+(i_left+1)*qp.usize, .func_compare=qp.func_compare, .other_param=qp.other_param};
    bool res_push;
    if (len_left > len_right){
        if (len_right >= 2){
            res_push = Stack_Push(stack, &right_array);
            if (!res_push){
                Madd_Error_Add(MADD_ERROR, L"Sort_Quicksort_Internal: unable to push param into stack. See info from Stack_Push.");
            }
            return false;
        }
        if (len_left >= 2){
            res_push = Stack_Push(stack, &left_array);
            if (!res_push){
                Madd_Error_Add(MADD_ERROR, L"Sort_Quicksort_Internal: unable to push param into stack. See info from Stack_Push.");
            }
            return false;
        }
    }else{
        if (len_left >= 2){
            res_push = Stack_Push(stack, &left_array);
            if (!res_push){
                Madd_Error_Add(MADD_ERROR, L"Sort_Quicksort_Internal: unable to push param into stack. See info from Stack_Push.");
            }
            return false;
        }
        if (len_right >= 2){
            res_push = Stack_Push(stack, &right_array);
            if (!res_push){
                Madd_Error_Add(MADD_ERROR, L"Sort_Quicksort_Internal: unable to push param into stack. See info from Stack_Push.");
            }
            return false;
        }
    }
    return true;
}

inline void Sort_Quicksort_Clean(size_t usize, void *pivot, Stack *stack)
{
    if (usize > 1024){
        free(pivot);
    }
    Stack_Destroy(stack);
}

void Sort_Quicksort(uint64_t n_element, size_t usize, void *arr_, bool func_compare(void *a, void *b, void *other_param), void *other_param)
{
    if (n_element < 2){
        Madd_Error_Add(MADD_WARNING, L"Sort_Quicksort: array length is less than 2, unnecessary to sort.");
        return;
    }
    unsigned char *arr = arr_;
    unsigned char arr_pivot[1024];
    void *pivot;
    if (usize > 1024){
        pivot = malloc(usize);
        if (pivot == NULL){
            Madd_Error_Add(MADD_ERROR, L"Sort_Quicksort: unable to allocate mem for pivot.");
            return;
        }
    }else{
        pivot = arr_pivot;
    }
    Stack stack;
    Stack_Init(&stack, 0, sizeof(Quicksort_Param));
    if (stack.capacity == 0){
        Madd_Error_Add(MADD_ERROR, L"Sort_Quicksort: unable to create a stack.");
        if (usize > 1024){
            free(pivot);
        }
        return;
    }
    Quicksort_Param qp = {.len=n_element, .usize=usize, .arr=arr_, .func_compare=func_compare, .other_param=other_param};
    bool res_push = Stack_Push(&stack, &qp);
    if (!res_push){
        Madd_Error_Add(MADD_ERROR, L"Sort_Quicksort: unable to push param into stack. See info from Stack_Push.");
        Sort_Quicksort_Clean(usize, pivot, &stack);
        return;
    }

    while (!Stack_Empty(&stack)){
        bool res_internal = Sort_Quicksort_Internal(&stack, pivot);
        if (!res_internal){
            Madd_Error_Add(MADD_ERROR, L"Sort_Quicksort: See info from Sort_Quicksort_Internal.");
            Sort_Quicksort_Clean(usize, pivot, &stack);
            return;
        }
    }
    
    Sort_Quicksort_Clean(usize, pivot, &stack);
}