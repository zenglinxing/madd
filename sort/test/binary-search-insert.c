/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include"madd.h"

#define N 1000

char func_compare(void *p1, void *p2, void *other_param)
{
    double *a=p1, *b=p2;
    if (*a < *b) return MADD_LESS;
    else if (*a > *b) return MADD_GREATER;
    else return MADD_SAME;
}

int main(int argc, char *argv[])
{
    //Madd_Error_Enable_Logfile("test_sort_binary-search.log");
    double *arr=(double*)malloc(N*sizeof(double));
    if (arr==NULL){
        printf("cannot allocate memory.\n");
        exit(EXIT_FAILURE);
    }
    int i, j;
    for (i=0; i<N; i++){
        arr[i] = i;
    }

    bool flag_error = false;
    int len, res;
    double di;
    uint64_t madd_error_n_expect = 0;
    //flag_print = false;
    printf("insert same element test\n");
    for (len=0; len<N; len++){
        for (i=0; i<len; i++){
            di = i;
            res = Binary_Search_Insert(len, sizeof(double), arr, &di, func_compare, NULL);
            if (i!=res && i!=res+1){
                printf("error when:\n\tlen=%d\ti=%d\tres=%d\n", len, i, res);
                flag_error = true;
                madd_error_n_expect ++;
            }
        }
    }

    /* failure condition test */
    printf("insert different element test\n");

    madd_error_n_expect ++;
    for (len=0; len<N; len++){
        for (i=0; i<len+1; i++){
            di = i-.5;
            res = Binary_Search_Insert(len, sizeof(double), arr, &di, func_compare, NULL);
            if (res != i){
                printf("error when:\n\tlen=%d\tdi=%f\tres=%d\n", len, di, res);
                flag_error = true;
            }else{
                madd_error_n_expect ++;
            }
        }
    }

    free(arr);
    if (flag_error){
        printf("test failed\n");
        exit(EXIT_FAILURE);
    }else{
        printf("test passed\n");
    }
    return 0;
}