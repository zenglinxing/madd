/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/sort_shell.cpp
*/
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include<stdbool.h>
extern "C"{
#include"sort.h"
#include"../basic/basic.h"
}

#define MADD_SORT_SEDGEWICK_LENGTH 60

static uint64_t madd_shell_sort_sedgewick[MADD_SORT_SEDGEWICK_LENGTH];
static bool madd_shell_sort_sedgewick_init = false;

class CppSedgewick{
    public:
    CppSedgewick(){
        uint64_t s1[30], s2[30];
        int i, j;
        uint64_t i4, i2;
        for (i=0,i4=i2=1; i<30; i++,i4*=4,i2*=2){
            s1[i] = 9*(i4 - i2) + 1;
            //printf("i=%d\t%llxz\n", i, s1[i]);
        }
        for (i=2,i4=16,i2=4; i<32; i++,i4*=4,i2*=2){
            s2[i-2] = i4 - 3*i2 + 1;
            //printf("i=%d\t%llx\n", i, s2[i]);
        }
        /* merge */
        i = j = 0;
        while (i < 30 && j < 30){
            if (s1[i] < s2[j]){
                madd_shell_sort_sedgewick[i+j] = s1[i];
                i++;
            }else{
                madd_shell_sort_sedgewick[i+j] = s2[j];
                j++;
            }
        }
        if (i == 30){
            memcpy(madd_shell_sort_sedgewick+i+j, s2+j, (30-j)*sizeof(uint64_t));
        }
        if (j == 30){
            memcpy(madd_shell_sort_sedgewick+i+j, s1+i, (30-i)*sizeof(uint64_t));
        }
        madd_shell_sort_sedgewick_init = true;
    }
};

static CppSedgewick sedgewick_init;

extern "C" void Sort_Shell(uint64_t n_element, size_t usize, void *arr_,
                           bool func_compare(void*, void*, void*), void *other_param)
{
    if (n_element < 2){
        Madd_Error_Add(MADD_WARNING, L"Sort_Shell: array length is less than 2, unnecessary to sort.");
        return;
    }
    if (usize == 0){
        Madd_Error_Add(MADD_ERROR, L"Sort_Shell: usize is 0.");
        return;
    }
    if (arr_ == NULL){
        Madd_Error_Add(MADD_ERROR, L"Sort_Shell: array pointer is NULL.");
        return;
    }
    if (!madd_shell_sort_sedgewick_init){
        Madd_Error_Add(MADD_ERROR, L"Sort_Shell: Sedgewick augment array did not initialized. Check Madd compilation and make sure your C++ compiler & linker works");
        return;
    }

    unsigned char temp_element[1024], *ptemp, *arr=(unsigned char*)arr_;
    if (usize > 1024){
        ptemp = (unsigned char*)malloc(usize);
        if (ptemp == NULL){
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"Sort_Shell: cannot allocate %llu bytes for temporary mem.", usize);
            Madd_Error_Add(MADD_ERROR, error_info);
            return;
        }
    }else{
        ptemp = temp_element;
    }

    int gap_index = 0;
    while (gap_index < MADD_SORT_SEDGEWICK_LENGTH && 
           madd_shell_sort_sedgewick[gap_index] < n_element) {
        gap_index++;
    }
    
    for (int i = gap_index - 1; i >= 0; i--) {
        uint64_t gap = madd_shell_sort_sedgewick[i];
        
        for (uint64_t i_element = gap; i_element < n_element; i_element++) {
            memcpy(ptemp, arr + i_element * usize, usize);
            
            uint64_t j_element = i_element;
            while (j_element >= gap) {
                uint64_t prev_index = j_element - gap;
                void* prev_element = arr + prev_index * usize;
                
                if (func_compare(ptemp, prev_element, other_param)) {
                    memcpy(arr + j_element * usize, prev_element, usize);
                    j_element = prev_index;
                } else {
                    break;
                }
            }
            if (j_element != i_element) {
                memcpy(arr + j_element * usize, ptemp, usize);
            }
        }
    }

    if (usize > 1024){
        free(ptemp);
    }
}