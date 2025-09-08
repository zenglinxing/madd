/* coding: utf-8 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include"madd.h"

// 测试用例数据结构
typedef struct {
    uint64_t key;
    uint64_t id;  // 用于验证稳定性
    char data[16]; // 用于验证数据完整性
} TestElement;

// 获取键值的回调函数
uint64_t get_key(void *element, void *other_param) {
    return ((TestElement*)element)->key;
}

// 验证数组排序结果
bool is_sorted(TestElement *arr, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (arr[i].key < arr[i-1].key) {
            wprintf(L"Sort violation at index %zu: %llu > %llu\n", 
                   i, arr[i-1].key, arr[i].key);
            return false;
        }
    }
    return true;
}

// 验证稳定性 (相同键值保持原始顺序)
bool is_stable(TestElement *arr, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (arr[i].key == arr[i-1].key && arr[i].id < arr[i-1].id) {
            wprintf(L"Stability violation at index %zu: id %llu vs %llu (key %llu)\n", 
                   i, arr[i-1].id, arr[i].id, arr[i].key);
            return false;
        }
    }
    return true;
}

// 打印数组内容
void print_array(TestElement *arr, size_t n, const char *label) {
    printf("%s [%zu elements]:\n", label, n);
    for (size_t i = 0; i < n; i++) {
        printf("  [%zu] Key: %-4llu ID: %-4llu Data: %s\n", 
              i, arr[i].key, arr[i].id, arr[i].data);
    }
    printf("\n");
}

// 测试用例1: 空数组
void test_empty() {
    TestElement arr[1] = {{0}}; // 不需要实际元素
    printf("=== Test 1: Empty Array ===\n");
    Sort_Counting(0, sizeof(TestElement), arr, get_key, NULL);
    printf("Test passed (no crash)\n\n");
}

// 测试用例2: 单元素数组
void test_single_element() {
    TestElement arr[] = {{5, 1, "Single"}};
    printf("=== Test 2: Single Element ===\n");
    print_array(arr, 1, "Before");
    Sort_Counting(1, sizeof(TestElement), arr, get_key, NULL);
    print_array(arr, 1, "After");
    printf("Test passed\n\n");
}

// 测试用例3: 已排序数组
void test_sorted() {
    TestElement arr[] = {
        {1, 1, "A"}, {2, 2, "B"}, {3, 3, "C"}, {4, 4, "D"}, {5, 5, "E"}
    };
    printf("=== Test 3: Already Sorted ===\n");
    print_array(arr, 5, "Before");
    Sort_Counting(5, sizeof(TestElement), arr, get_key, NULL);
    print_array(arr, 5, "After");
    
    if (is_sorted(arr, 5) && is_stable(arr, 5)) {
        printf("Test passed\n\n");
    } else {
        printf("Test failed\n\n");
    }
}

// 测试用例4: 逆序数组
void test_reverse_sorted() {
    TestElement arr[] = {
        {5, 1, "E"}, {4, 2, "D"}, {3, 3, "C"}, {2, 4, "B"}, {1, 5, "A"}
    };
    printf("=== Test 4: Reverse Sorted ===\n");
    print_array(arr, 5, "Before");
    Sort_Counting(5, sizeof(TestElement), arr, get_key, NULL);
    print_array(arr, 5, "After");
    
    if (is_sorted(arr, 5) && is_stable(arr, 5)) {
        printf("Test passed\n\n");
    } else {
        printf("Test failed\n\n");
    }
}

// 测试用例5: 重复键值（稳定性测试）
void test_duplicate_keys() {
    TestElement arr[] = {
        {3, 1, "A"}, {2, 2, "B"}, {3, 3, "C"}, {1, 4, "D"}, {2, 5, "E"}
    };
    printf("=== Test 5: Duplicate Keys ===\n");
    print_array(arr, 5, "Before");
    Sort_Counting(5, sizeof(TestElement), arr, get_key, NULL);
    print_array(arr, 5, "After");
    
    if (is_sorted(arr, 5) && is_stable(arr, 5)) {
        printf("Test passed\n\n");
    } else {
        printf("Test failed\n\n");
    }
}

// 测试用例6: 随机数据
void test_random_data(size_t n) {
    printf("=== Test 6: Random Data (%zu elements) ===\n", n);
    
    TestElement *arr = malloc(n * sizeof(TestElement));
    if (!arr) {
        printf("Memory allocation failed\n");
        return;
    }
    
    // 初始化随机数据
    srand(time(NULL));
    for (size_t i = 0; i < n; i++) {
        arr[i].key = rand() % 1000;  // 0-999的随机键
        arr[i].id = i;               // ID用于验证稳定性
        sprintf(arr[i].data, "Data%zu", i);
    }
    
    print_array(arr, (n > 10 ? 10 : n), "First 10 elements before");
    Sort_Counting(n, sizeof(TestElement), arr, get_key, NULL);
    print_array(arr, (n > 10 ? 10 : n), "First 10 elements after");
    
    if (is_sorted(arr, n) && is_stable(arr, n)) {
        printf("Test passed\n\n");
    } else {
        printf("Test failed\n\n");
    }
    
    free(arr);
}

// 测试用例7: 大键值范围（稀疏数据）
void test_large_keys() {
    const size_t n = 5;
    TestElement arr[] = {
        {1000000, 1, "A"}, 
        {5000000, 2, "B"}, 
        {2000000, 3, "C"}, 
        {3000000, 4, "D"}, 
        {1, 5, "E"}
    };
    printf("=== Test 7: Large Key Range ===\n");
    print_array(arr, n, "Before");
    Sort_Counting(n, sizeof(TestElement), arr, get_key, NULL);
    print_array(arr, n, "After");
    
    if (is_sorted(arr, n) && is_stable(arr, n)) {
        printf("Test passed\n\n");
    } else {
        printf("Test failed\n\n");
    }
}

// 测试用例8: 内存分配失败模拟
void test_memory_failure() {
    TestElement arr[] = {
        {10000000000, 1, "A"}, // 非常大的键值，会触发大内存分配
        {90000000000, 2, "B"}
    };
    printf("=== Test 8: Memory Failure ===\n");
    Sort_Counting(2, sizeof(TestElement), arr, get_key, NULL);
    Madd_Error_Print_Last();
    printf("Expected error message should appear above\n");
    printf("Test passed (if error handled gracefully)\n\n");
}

int main() {
    // 运行测试套件
    Madd_Error_Enable_Logfile("test_sort-counting.log");
    test_empty();
    test_single_element();
    test_sorted();
    test_reverse_sorted();
    test_duplicate_keys();
    test_random_data(20);      // 中等规模测试
    test_random_data(100000);  // 大规模测试 (100K元素)
    test_large_keys();
    //test_memory_failure();
    
    printf("All tests completed.\n");
    return 0;
}