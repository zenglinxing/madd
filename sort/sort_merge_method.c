/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort_merge_method.c
*/
#include<string.h>
#include<stdint.h>
#include<stdbool.h>
#include"binary_search.h"

uint64_t n_sort_galloping = 7;

void Sort__Merge_Left(uint64_t n_left, uint64_t n_right, size_t usize,
                      void *arr_, void *temp_,
                      bool func_compare(void *a, void *b, void *other_param), void *other_param)
{
    uint64_t i_left=0, i_right=0;
    unsigned char *pleft=temp_, *pright=((unsigned char*)arr_)+n_left*usize, *parr=arr_;
    memcpy(temp_, arr_, n_left*usize);

    while (i_left != n_left && i_right != n_right){
        if (func_compare(pleft, pright, other_param)){
            memcpy(parr, pleft, usize);
            pleft += usize;
            i_left ++;
        }else{
            memcpy(parr, pright, usize);
            pright += usize;
            i_right ++;
        }
        parr += usize;
    }
    if (i_left == n_left){
        /* no need to copy */
        /*memcpy(parr, pright, (n_right-i_right)*usize);*/
    }else{
        memcpy(parr, pleft, (n_left-i_left)*usize);
    }
}

void Sort__Merge_Right(uint64_t n_left, uint64_t n_right, size_t usize,
                       void *arr_, void *temp_,
                       bool func_compare(void *a, void *b, void *other_param), void *other_param)
{
    uint64_t i_left=0, i_right=0;
    unsigned char *arr=arr_, *temp=temp_, *pleft=arr+(n_left-1)*usize, *pright=temp+(n_right-1)*usize, *parr=arr + (n_left+n_right-1)*usize;
    memcpy(temp, pleft+usize, n_right*usize);

    while (i_left != n_left && i_right != n_right){
        if (func_compare(pleft, pright, other_param)){
            memcpy(parr, pright, usize);
            pright -= usize;
            i_right ++;
        }else{
            memcpy(parr, pleft, usize);
            pleft -= usize;
            i_left ++;
        }
        parr -= usize;
    }
    if (i_left == n_left){
        memcpy(arr_, temp, (n_right-i_right)*usize);
    }else{
        /* no need to copy */
    }
}