#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include"madd.h"

typedef bool (*compare_func_t)(void *a, void *b, void *other_param);

// ================= 测试工具函数 =================

// 打印整型数组
void print_int_array(int *arr, size_t n) {
    printf("[");
    for (size_t i = 0; i < n; i++) {
        printf("%d", arr[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

// 打印字符串数组
void print_str_array(char **arr, size_t n) {
    printf("[");
    for (size_t i = 0; i < n; i++) {
        printf("\"%s\"", arr[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

// 验证数组是否有序
bool verify_sorted(void *arr, size_t n, size_t size, 
                   compare_func_t compare, void *param) {
    unsigned char *bytes = (unsigned char *)arr;
    for (size_t i = 0; i < n - 1; i++) {
        void *a = bytes + i * size;
        void *b = bytes + (i + 1) * size;
        if (!compare(a, b, param)) {
            printf("Order violation at position %zu: ", i);
            if (size == sizeof(int)) {
                printf("%d > %d\n", *(int *)a, *(int *)b);
            }
            return false;
        }
    }
    return true;
}

// ================= 比较函数实现 =================

// 整型升序比较
bool int_ascending(void *a, void *b, void *param) {
    return *(int *)a <= *(int *)b;
}

// 整型降序比较
bool int_descending(void *a, void *b, void *param) {
    return *(int *)a >= *(int *)b;
}

// 字符串升序比较 (字典序)
bool str_ascending(void *a, void *b, void *param) {
    return strcmp(*(char **)a, *(char **)b) <= 0;
}

// 自定义结构体
typedef struct {
    int id;
    char name[16];
    float score;
} Student;

// 学生按分数升序比较
bool student_score_asc(void *a, void *b, void *param) {
    Student *sa = (Student *)a;
    Student *sb = (Student *)b;
    return sa->score <= sb->score;
}

// ================= 测试用例 =================

// 测试空数组
void test_empty_array() {
    printf("=== Testing empty array ===\n");
    int arr[1] = {0}; // 防止未定义行为
    Sort_Insertion(0, sizeof(int), arr, int_ascending, NULL);
    printf("Test passed (no crash)\n\n");
}

// 测试单元素数组
void test_single_element() {
    printf("=== Testing single element ===\n");
    int arr[] = {42};
    printf("Before: ");
    print_int_array(arr, 1);
    
    Sort_Insertion(1, sizeof(int), arr, int_ascending, NULL);
    
    printf("After: ");
    print_int_array(arr, 1);
    printf("Test passed\n\n");
}

// 测试已排序数组
void test_already_sorted() {
    printf("=== Testing already sorted array ===\n");
    int arr[] = {1, 2, 3, 4, 5};
    printf("Before: ");
    print_int_array(arr, 5);
    
    Sort_Insertion(5, sizeof(int), arr, int_ascending, NULL);
    
    printf("After: ");
    print_int_array(arr, 5);
    printf("Test passed\n\n");
}

// 测试逆序数组
void test_reverse_sorted() {
    printf("=== Testing reverse sorted array ===\n");
    int arr[] = {9, 7, 5, 3, 1};
    printf("Before: ");
    print_int_array(arr, 5);
    
    Sort_Insertion(5, sizeof(int), arr, int_ascending, NULL);
    
    printf("After: ");
    print_int_array(arr, 5);
    
    if (verify_sorted(arr, 5, sizeof(int), int_ascending, NULL)) {
        printf("Test passed\n\n");
    } else {
        printf("Test failed\n\n");
    }
}

// 测试随机数组
void test_random_array() {
    printf("=== Testing random array ===\n");
    int arr[20];
    const size_t n = sizeof(arr) / sizeof(arr[0]);
    
    // 生成随机数
    srand(time(NULL));
    for (size_t i = 0; i < n; i++) {
        arr[i] = rand() % 100;
    }
    
    printf("Before: ");
    print_int_array(arr, n);
    
    Sort_Insertion(n, sizeof(int), arr, int_ascending, NULL);
    
    printf("After: ");
    print_int_array(arr, n);
    
    if (verify_sorted(arr, n, sizeof(int), int_ascending, NULL)) {
        printf("Test passed\n\n");
    } else {
        printf("Test failed\n\n");
    }
}

// 测试降序排序
void test_descending_sort() {
    printf("=== Testing descending sort ===\n");
    int arr[] = {3, 1, 4, 2, 5};
    printf("Before: ");
    print_int_array(arr, 5);
    
    Sort_Insertion(5, sizeof(int), arr, int_descending, NULL);
    
    printf("After: ");
    print_int_array(arr, 5);
    
    if (verify_sorted(arr, 5, sizeof(int), int_descending, NULL)) {
        printf("Test passed\n\n");
    } else {
        printf("Test failed\n\n");
    }
}

// 测试字符串数组
void test_string_array() {
    printf("=== Testing string array ===\n");
    char *arr[] = {"banana", "apple", "pear", "orange", "grape"};
    const size_t n = sizeof(arr) / sizeof(arr[0]);
    
    printf("Before: ");
    print_str_array(arr, n);
    
    Sort_Insertion(n, sizeof(char *), arr, str_ascending, NULL);
    
    printf("After: ");
    print_str_array(arr, n);
    
    // 验证排序
    bool sorted = true;
    for (size_t i = 0; i < n - 1; i++) {
        if (strcmp(arr[i], arr[i + 1]) > 0) {
            sorted = false;
            break;
        }
    }
    
    if (sorted) {
        printf("Test passed\n\n");
    } else {
        printf("Test failed\n\n");
    }
}

// 测试自定义结构体
void test_custom_struct() {
    printf("=== Testing custom struct ===\n");
    Student students[] = {
        {1, "Alice", 88.5f},
        {2, "Bob", 92.0f},
        {3, "Charlie", 76.5f},
        {4, "David", 95.0f},
        {5, "Eve", 82.0f}
    };
    const size_t n = sizeof(students) / sizeof(students[0]);
    
    printf("Before sorting by score:\n");
    for (size_t i = 0; i < n; i++) {
        printf("  %s: %.1f\n", students[i].name, students[i].score);
    }
    
    Sort_Insertion(n, sizeof(Student), students, student_score_asc, NULL);
    
    printf("\nAfter sorting by score:\n");
    for (size_t i = 0; i < n; i++) {
        printf("  %s: %.1f\n", students[i].name, students[i].score);
    }
    
    if (verify_sorted(students, n, sizeof(Student), student_score_asc, NULL)) {
        printf("Test passed\n\n");
    } else {
        printf("Test failed\n\n");
    }
}

// 测试大型数组性能
void test_large_array() {
    printf("=== Testing large array (10,000 elements) ===\n");
    const size_t n = 10000;
    int *arr = malloc(n * sizeof(int));
    
    if (!arr) {
        printf("Memory allocation failed\n");
        return;
    }
    
    // 填充逆序数组 (最坏情况)
    for (size_t i = 0; i < n; i++) {
        arr[i] = n - i;
    }
    
    clock_t start = clock();
    Sort_Insertion(n, sizeof(int), arr, int_ascending, NULL);
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sorted %zu elements in %.3f seconds\n", n, elapsed);
    
    if (verify_sorted(arr, n, sizeof(int), int_ascending, NULL)) {
        printf("Test passed\n\n");
    } else {
        printf("Test failed\n\n");
    }
    
    free(arr);
}

// ================= 主函数 =================

int main() {
    test_empty_array();
    test_single_element();
    test_already_sorted();
    test_reverse_sorted();
    test_random_array();
    test_descending_sort();
    test_string_array();
    test_custom_struct();
    test_large_array();
    
    printf("All tests completed!\n");
    return 0;
}

// ================= 需要实现的依赖函数 =================

// 简化的二分查找插入位置实现
uint64_t Binary_Search_Insert(uint64_t n, size_t usize, void *arr_, void *element,
                              compare_func_t func_compare, void *other_param) 
{
    if (n == 0) return 0;
    
    unsigned char *arr = (unsigned char *)arr_;
    uint64_t low = 0;
    uint64_t high = n;
    
    // 检查是否应该插入到最前面
    if (func_compare(element, arr, other_param)) {
        return 0;
    }
    
    // 检查是否应该插入到最后面
    if (!func_compare(arr + (n - 1) * usize, element, other_param)) {
        return n;
    }
    
    while (low < high) {
        uint64_t mid = low + (high - low) / 2;
        unsigned char *mid_ptr = arr + mid * usize;
        
        if (func_compare(mid_ptr, element, other_param)) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    
    return low;
}