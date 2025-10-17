/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>
#include<string.h>
#include"madd.h"

double tolerance = 1e-6;
uint64_t nn = 11;

bool compare_array(uint64_t n, double *a, double *b)
{
    for (uint64_t i = 0; i < n; i++){
        if (fabs(a[i] - b[i])/fabs(a[i]) > tolerance){
            printf("Mismatch at index %llu: %f vs %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

bool Array_5_Reverse()
{
    printf("===\nArray of 5 elements before DCT-I:\n===\n");
    uint64_t n = 5, i;
    double *arr = (double*)malloc(sizeof(double) * n * 2), *arr_old = arr + n;
    if (arr == NULL){
        return false;
    }
    for (i = 0; i < n; i++){
        arr[i] = arr_old[i] = i + 1;
        printf("%f\t", arr[i]);
    }
    printf("\n");

    Discrete_Cosine_Transform_1(n, arr);
    //Discrete_Cosine_Transform_1_Naive(n, arr);
    printf("after DCT-I:\n");
    for (i = 0; i < n; i++){
        printf("%f\t", arr[i]);
    }
    printf("\n");
    Discrete_Cosine_Transform_1(n, arr);
    //Discrete_Cosine_Transform_1_Naive(n, arr);
    printf("after DCT-I:\n");
    for (i = 0; i < n; i++){
        printf("%f\t", arr[i]);
    }
    printf("\n");

    bool result = compare_array(n, arr, arr_old);
    if (result){
        printf("DCT-I reverse test passed for array of 5 elements.\n");
        free(arr);
        return true;
    }else{
        printf("*** DCT-I reverse test failed for array of 5 elements. ***\n");
        free(arr);
        return false;
    }
}

bool Array_N_Reverse()
{
    printf("===\nArray of %llu elements before DCT-I:\n===\n", nn);
    uint64_t n = nn, i;
    double *arr = (double*)malloc(sizeof(double) * n * 2), *arr_old = arr + n;
    if (arr == NULL){
        return false;
    }
    for (i = 0; i < n; i++){
        arr[i] = arr_old[i] = i + 1;
        printf("%f\t", arr[i]);
    }
    printf("\n");

    Discrete_Cosine_Transform_1(n, arr);
    //Discrete_Cosine_Transform_1_Naive(n, arr);
    printf("after DCT-I:\n");
    for (i = 0; i < n; i++){
        printf("%f\t", arr[i]);
    }
    printf("\n");
    Discrete_Cosine_Transform_1(n, arr);
    //Discrete_Cosine_Transform_1_Naive(n, arr);
    printf("after DCT-I:\n");
    for (i = 0; i < n; i++){
        printf("%f\t", arr[i]);
    }
    printf("\n");

    bool result = compare_array(n, arr, arr_old);
    if (result){
        printf("DCT-I reverse test passed for array of %llu elements.\n", n);
        free(arr);
        return true;
    }else{
        printf("*** DCT-I reverse test failed for array of %llu elements. ***\n", n);
        free(arr);
        return false;
    }
}

int main(int argc, char *argv[])
{
    if (argc >= 2){
        nn = atoi(argv[1]);
    }
    bool all_passed = true;

    if (!Array_5_Reverse()){
        all_passed = false;
    }

    printf("\n");

    if (!Array_N_Reverse()){
        all_passed = false;
    }

    if (all_passed){
        printf("All tests passed.\n");
        return EXIT_SUCCESS;
    }else{
        printf("*** Some tests failed. ***\n");
        return EXIT_FAILURE;
    }
}