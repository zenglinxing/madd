/* coding: utf-8 */
#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include<stdbool.h>
#include"madd.h"

bool func_compare(void *a, void *b, void *other_param)
{
    uint64_t *aa=a, *bb=b;
    return *aa <= *bb;
}

uint64_t get_key(void *a, void *other_param)
{
    uint64_t *aa=a;
    return *aa;
}

int main(int argc, char *argv[])
{
    uint64_t n = 1e5, seed=10, i;
    uint64_t *arr=(uint64_t*)malloc(n*sizeof(uint64_t)), *arr_sorted=(uint64_t*)malloc(n*sizeof(uint64_t));
    RNG_Xoshiro256_Param rng = RNG_Xoshiro256ss_Init(seed);
    for (i=0; i<n; i++){
        arr[i] = RNG_Xoshiro256ss_U64(&rng);
    }

    /* counting sort */
    memcpy(arr_sorted, arr, n*sizeof(uint64_t));
    clock_t clock_counting1=clock(), clock_counting2;
    Sort_Counting(n, sizeof(uint64_t), arr_sorted, get_key, NULL);
    clock_counting2 = clock();
    double time_counting = (clock_counting2 - clock_counting1) / (double)CLOCKS_PER_SEC;
    printf("counting sort:\t%f sec\n", time_counting);

    /* heap sort */
    memcpy(arr_sorted, arr, n*sizeof(uint64_t));
    clock_t clock_heap1=clock(), clock_heap2;
    Sort_Heap(n, sizeof(uint64_t), arr_sorted, func_compare, NULL);
    clock_heap2 = clock();
    double time_heap = (clock_heap2 - clock_heap1) / (double)CLOCKS_PER_SEC;
    printf("heap sort:\t%f sec\n", time_heap);

    /* insertion sort */
    memcpy(arr_sorted, arr, n*sizeof(uint64_t));
    clock_t clock_insertion1=clock(), clock_insertion2;
    Sort_Insertion(n, sizeof(uint64_t), arr_sorted, func_compare, NULL);
    clock_insertion2 = clock();
    double time_insertion = (clock_insertion2 - clock_insertion1) / (double)CLOCKS_PER_SEC;
    printf("insertion sort:\t%f sec\n", time_insertion);

    /* merge sort */
    memcpy(arr_sorted, arr, n*sizeof(uint64_t));
    clock_t clock_merge1=clock(), clock_merge2;
    Sort_Merge(n, sizeof(uint64_t), arr_sorted, func_compare, NULL);
    clock_merge2 = clock();
    double time_merge = (clock_merge2 - clock_merge1) / (double)CLOCKS_PER_SEC;
    printf("merge sort:\t%f sec\n", time_merge);

    /* quicksort */
    memcpy(arr_sorted, arr, n*sizeof(uint64_t));
    clock_t clock_quick1=clock(), clock_quick2;
    Sort_Quicksort(n, sizeof(uint64_t), arr_sorted, func_compare, NULL);
    clock_quick2 = clock();
    double time_quick = (clock_quick2 - clock_quick1) / (double)CLOCKS_PER_SEC;
    printf("quicksort:\t%f sec\n", time_quick);

    /* shell sort */
    memcpy(arr_sorted, arr, n*sizeof(uint64_t));
    clock_t clock_shell1=clock(), clock_shell2;
    Sort_Shell(n, sizeof(uint64_t), arr_sorted, func_compare, NULL);
    clock_shell2 = clock();
    double time_shell = (clock_shell2 - clock_shell1) / (double)CLOCKS_PER_SEC;
    printf("shell sort:\t%f sec\n", time_shell);

    free(arr);
    free(arr_sorted);
    return 0;
}