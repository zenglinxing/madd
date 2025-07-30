/* coding: utf-8 */
/*
This test code is provided by Deepseek
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include"madd.h"

// 比较函数：整数
bool int_compare(void *a1, void *a2, void *other_param) {
    return (*(int*)a1) < (*(int*)a2);
}

// 比较函数：字符串
bool str_compare(void *a1, void *a2, void *other_param) {
    return strcmp(*(char**)a1, *(char**)a2) < 0;
}

// 验证数组是否有序
bool is_sorted(void *arr, uint64_t n, size_t usize, 
               bool (*compare)(void*, void*, void *other_param), void *other_param) {
    unsigned char *p = (unsigned char*)arr;
    for (uint64_t i = 0; i < n-1; i++) {
        if (!compare(p, p + usize, other_param) && 
            compare(p + usize, p, other_param)) { // 检查相邻元素顺序
            return false;
        }
        p += usize;
    }
    return true;
}

// 打印整型数组
void print_int_array(int *arr, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// 测试用例
void test_merge_sort() {
    // 测试1: 空数组
    {
        int *arr = NULL;
        Sort_Merge(0, sizeof(int), arr, int_compare, NULL);
        printf("Test 1 (empty array): Passed\n");
    }
    
    // 测试2: 单个元素
    {
        int arr[] = {5};
        Sort_Merge(1, sizeof(int), arr, int_compare, NULL);
        if (arr[0] == 5 && is_sorted(arr, 1, sizeof(int), int_compare, NULL)) {
            printf("Test 2 (single element): Passed\n");
        } else {
            printf("Test 2 (single element): Failed\n");
        }
    }
    
    // 测试3: 已排序数组
    {
        int arr[] = {1, 2, 3, 4, 5};
        uint64_t n = sizeof(arr)/sizeof(arr[0]);
        Sort_Merge(n, sizeof(int), arr, int_compare, NULL);
        if (is_sorted(arr, n, sizeof(int), int_compare, NULL)) {
            printf("Test 3 (sorted array): Passed\n");
        } else {
            printf("Test 3 (sorted array): Failed\n");
            print_int_array(arr, n);
        }
    }
    
    // 测试4: 逆序数组
    {
        int arr[] = {5, 4, 3, 2, 1};
        uint64_t n = sizeof(arr)/sizeof(arr[0]);
        Sort_Merge(n, sizeof(int), arr, int_compare, NULL);
        if (is_sorted(arr, n, sizeof(int), int_compare, NULL)) {
            printf("Test 4 (reversed array): Passed\n");
        } else {
            printf("Test 4 (reversed array): Failed\n");
            print_int_array(arr, n);
        }
    }
    
    // 测试5: 随机数组
    {
        int arr[] = {3, 1, 4, 2, 5};
        uint64_t n = sizeof(arr)/sizeof(arr[0]);
        Sort_Merge(n, sizeof(int), arr, int_compare, NULL);
        if (is_sorted(arr, n, sizeof(int), int_compare, NULL)) {
            printf("Test 5 (random array): Passed\n");
        } else {
            printf("Test 5 (random array): Failed\n");
            print_int_array(arr, n);
        }
    }
    
    // 测试6: 重复元素
    {
        int arr[] = {3, 1, 4, 2, 5, 2, 3};
        uint64_t n = sizeof(arr)/sizeof(arr[0]);
        Sort_Merge(n, sizeof(int), arr, int_compare, NULL);
        if (is_sorted(arr, n, sizeof(int), int_compare, NULL)) {
            printf("Test 6 (duplicates): Passed\n");
        } else {
            printf("Test 6 (duplicates): Failed\n");
            print_int_array(arr, n);
        }
    }
    
    // 测试7: 偶数元素数量
    {
        int arr[] = {4, 2, 7, 1, 5, 3};
        uint64_t n = sizeof(arr)/sizeof(arr[0]);
        Sort_Merge(n, sizeof(int), arr, int_compare, NULL);
        if (is_sorted(arr, n, sizeof(int), int_compare, NULL)) {
            printf("Test 7 (even elements): Passed\n");
        } else {
            printf("Test 7 (even elements): Failed\n");
            print_int_array(arr, n);
        }
    }
    
    // 测试8: 奇数元素数量
    {
        int arr[] = {4, 2, 7, 1, 5};
        uint64_t n = sizeof(arr)/sizeof(arr[0]);
        Sort_Merge(n, sizeof(int), arr, int_compare, NULL);
        if (is_sorted(arr, n, sizeof(int), int_compare, NULL)) {
            printf("Test 8 (odd elements): Passed\n");
        } else {
            printf("Test 8 (odd elements): Failed\n");
            print_int_array(arr, n);
        }
    }
    
    // 测试9: 大数组
    {
        #define LARGE_SIZE 1000
        int *arr = malloc(LARGE_SIZE * sizeof(int));
        if (!arr) {
            perror("malloc");
            return;
        }
        printf("malloced\n");
        
        // 随机初始化
        srand(time(NULL));
        for (int i = 0; i < LARGE_SIZE; i++) {
            arr[i] = rand() % 10000;
        }
        printf("initialized\n");
        
        Sort_Merge(LARGE_SIZE, sizeof(int), arr, int_compare, NULL);
        printf("sorted\n");
        if (is_sorted(arr, LARGE_SIZE, sizeof(int), int_compare, NULL)) {
            printf("Test 9 (large array): Passed\n");
        } else {
            printf("Test 9 (large array): Failed\n");
            // 打印部分结果以便调试
            print_int_array(arr, 10);
            printf("...\n");
            print_int_array(arr + LARGE_SIZE - 10, 10);
        }
        
        free(arr);
    }
    
    // 测试10: 字符串数组
    {
        char *words[] = {"apple", "banana", "cherry", "date", "blueberry"};
        uint64_t n = sizeof(words)/sizeof(words[0]);
        
        Sort_Merge(n, sizeof(char*), words, str_compare, NULL);
        
        bool sorted = true;
        for (uint64_t i = 0; i < n-1; i++) {
            if (strcmp(words[i], words[i+1]) > 0) {
                sorted = false;
                break;
            }
        }
        
        if (sorted) {
            printf("Test 10 (strings): Passed\n");
            for (uint64_t i = 0; i < n; i++) {
                printf("%s ", words[i]);
            }
            printf("\n");
        } else {
            printf("Test 10 (strings): Failed\n");
        }
    }
    
    // 测试11: 内存边界检查
    {
        // 分配三个连续的内存块
        int *block = malloc(3 * 3 * sizeof(int));
        if (!block) {
            perror("malloc");
            return;
        }
        
        // 初始化测试数据 (0-8)
        for (int i = 0; i < 9; i++) {
            block[i] = i;
        }
        
        // 对中间块排序 (3-5)
        Sort_Merge(3, sizeof(int), block + 3, int_compare, NULL);
        
        // 验证排序后的顺序应该是3,4,5
        bool passed = (block[3] == 3) && (block[4] == 4) && (block[5] == 5);
        
        // 验证相邻块未被修改
        passed &= (block[0] == 0) && (block[1] == 1) && (block[2] == 2);
        passed &= (block[6] == 6) && (block[7] == 7) && (block[8] == 8);
        
        if (passed) {
            printf("Test 11 (memory boundaries): Passed\n");
        } else {
            printf("Test 11 (memory boundaries): Failed\n");
            print_int_array(block, 9);
        }
        
        free(block);
    }
}

int main() {
    test_merge_sort();
    return 0;
}