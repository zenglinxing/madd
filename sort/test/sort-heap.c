/* coding: utf-8 */
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include"madd.h"

/* ================== TEST FRAMEWORK ================== */
typedef struct {
    char name[20];
    int id;
    double score;
} Student;

// 比较函数：整数升序
bool compare_int_asc(void* a, void* b, void* unused) {
    (void)unused;
    return *(int*)a <= *(int*)b;
}

// 比较函数：整数降序
bool compare_int_desc(void* a, void* b, void* unused) {
    (void)unused;
    return *(int*)a >= *(int*)b;
}

// 比较函数：学生按分数升序
bool compare_student_score(void* a, void* b, void* unused) {
    (void)unused;
    return ((Student*)a)->score <= ((Student*)b)->score;
}

// 比较函数：学生按ID升序
bool compare_student_id(void* a, void* b, void* unused) {
    (void)unused;
    return ((Student*)a)->id <= ((Student*)b)->id;
}

// 验证数组是否有序
bool is_sorted(void* arr, uint64_t n, size_t usize, 
               bool func_compare(void*, void*, void*), void* param) {
    unsigned char* base = (unsigned char*)arr;
    for (uint64_t i = 0; i < n - 1; i++) {
        if (!func_compare(base + i*usize, base + (i+1)*usize, param)) {
            return false;
        }
    }
    return true;
}

// 打印整数数组
void print_int_array(int* arr, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// 打印学生数组
void print_student_array(Student* arr, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        printf("%s (ID:%d, Score:%.1f)\n", 
               arr[i].name, arr[i].id, arr[i].score);
    }
}

// 测试整数排序
void test_int_sort() {
    printf("\n===== Testing Integer Sorting =====\n");
    
    // Test 1: Small fixed array (ascending)
    int arr1[] = {5, 2, 8, 1, 9, 3};
    uint64_t n1 = sizeof(arr1)/sizeof(arr1[0]);
    printf("\nTest 1: Small fixed array (ascending)\n");
    printf("Original: "); print_int_array(arr1, n1);
    Sort_Heap(n1, sizeof(int), arr1, compare_int_asc, NULL);
    printf("Sorted:   "); print_int_array(arr1, n1);
    if (is_sorted(arr1, n1, sizeof(int), compare_int_asc, NULL)) {
        printf("PASS: Array is sorted in ascending order\n");
    } else {
        printf("FAIL: Array is not sorted correctly\n");
    }
    
    // Test 2: Small fixed array (descending)
    int arr2[] = {5, 2, 8, 1, 9, 3};
    uint64_t n2 = sizeof(arr2)/sizeof(arr2[0]);
    printf("\nTest 2: Small fixed array (descending)\n");
    printf("Original: "); print_int_array(arr2, n2);
    Sort_Heap(n2, sizeof(int), arr2, compare_int_desc, NULL);
    printf("Sorted:   "); print_int_array(arr2, n2);
    if (is_sorted(arr2, n2, sizeof(int), compare_int_desc, NULL)) {
        printf("PASS: Array is sorted in descending order\n");
    } else {
        printf("FAIL: Array is not sorted correctly\n");
    }
    
    // Test 3: Large random array (ascending)
    uint64_t n3 = 10000;
    int* arr3 = malloc(n3 * sizeof(int));
    srand(time(NULL));
    for (uint64_t i = 0; i < n3; i++) {
        arr3[i] = rand() % 10000;
    }
    printf("\nTest 3: Large random array (%lu elements, ascending)\n", n3);
    clock_t start = clock();
    Sort_Heap(n3, sizeof(int), arr3, compare_int_asc, NULL);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    if (is_sorted(arr3, n3, sizeof(int), compare_int_asc, NULL)) {
        printf("PASS: Array is sorted in ascending order (time: %.3f seconds)\n", time_taken);
    } else {
        printf("FAIL: Array is not sorted correctly\n");
    }
    free(arr3);
    
    // Test 4: Edge cases
    printf("\nTest 4: Edge cases\n");
    // Single element
    int single = 42;
    printf("Testing single element... ");
    Sort_Heap(1, sizeof(int), &single, compare_int_asc, NULL);
    printf("PASS (no crash)\n");
    
    // Empty array (shouldn't crash)
    printf("Testing empty array... ");
    Sort_Heap(0, sizeof(int), NULL, compare_int_asc, NULL);
    printf("PASS (no crash)\n");
    
    // Sorted array
    int sorted[] = {1, 2, 3, 4, 5};
    uint64_t n_sorted = 5;
    printf("Testing already sorted array... ");
    Sort_Heap(n_sorted, sizeof(int), sorted, compare_int_asc, NULL);
    if (is_sorted(sorted, n_sorted, sizeof(int), compare_int_asc, NULL)) {
        printf("PASS (remains sorted)\n");
    } else {
        printf("FAIL\n");
    }
    
    // Reverse sorted array
    int reverse[] = {5, 4, 3, 2, 1};
    uint64_t n_reverse = 5;
    printf("Testing reverse sorted array... ");
    Sort_Heap(n_reverse, sizeof(int), reverse, compare_int_asc, NULL);
    if (is_sorted(reverse, n_reverse, sizeof(int), compare_int_asc, NULL)) {
        printf("PASS (sorted correctly)\n");
    } else {
        printf("FAIL\n");
    }
}

// 测试结构体排序
void test_struct_sort() {
    printf("\n===== Testing Struct Sorting =====\n");
    
    // Test 1: Sort students by score (ascending)
    Student students1[] = {
        {"Alice", 101, 88.5},
        {"Bob", 102, 92.0},
        {"Charlie", 103, 76.0},
        {"David", 104, 95.5}
    };
    uint64_t n1 = sizeof(students1)/sizeof(students1[0]);
    
    printf("\nTest 1: Sorting students by score (ascending)\n");
    printf("Original:\n"); print_student_array(students1, n1);
    Sort_Heap(n1, sizeof(Student), students1, compare_student_score, NULL);
    printf("\nSorted by score:\n"); print_student_array(students1, n1);
    if (is_sorted(students1, n1, sizeof(Student), compare_student_score, NULL)) {
        printf("PASS: Students sorted by score\n");
    } else {
        printf("FAIL: Students not sorted by score\n");
    }
    
    // Test 2: Sort students by ID (ascending)
    Student students2[] = {
        {"Charlie", 103, 76.0},
        {"Alice", 101, 88.5},
        {"David", 104, 95.5},
        {"Bob", 102, 92.0}
    };
    uint64_t n2 = sizeof(students2)/sizeof(students2[0]);
    
    printf("\nTest 2: Sorting students by ID (ascending)\n");
    printf("Original:\n"); print_student_array(students2, n2);
    Sort_Heap(n2, sizeof(Student), students2, compare_student_id, NULL);
    printf("\nSorted by ID:\n"); print_student_array(students2, n2);
    if (is_sorted(students2, n2, sizeof(Student), compare_student_id, NULL)) {
        printf("PASS: Students sorted by ID\n");
    } else {
        printf("FAIL: Students not sorted by ID\n");
    }
    
    // Test 3: Large struct with big element size (>1024 bytes)
    typedef struct {
        char data[1025]; // 1025-byte struct
        int key;
    } BigStruct;
    
    uint64_t n3 = 10;
    BigStruct* big_arr = malloc(n3 * sizeof(BigStruct));
    srand(time(NULL));
    for (uint64_t i = 0; i < n3; i++) {
        big_arr[i].key = rand() % 100;
        // Fill with random data
        for (int j = 0; j < 1024; j++) {
            big_arr[i].data[j] = rand() % 256;
        }
        big_arr[i].data[1024] = '\0';
    }
    
    printf("\nTest 3: Sorting large structs (1025 bytes each)\n");
    bool compare_big_struct(void* a, void* b, void* unused) {
        (void)unused;
        return ((BigStruct*)a)->key <= ((BigStruct*)b)->key;
    }
    
    Sort_Heap(n3, sizeof(BigStruct), big_arr, compare_big_struct, NULL);
    
    bool sorted = true;
    for (uint64_t i = 0; i < n3 - 1; i++) {
        if (big_arr[i].key > big_arr[i+1].key) {
            sorted = false;
            break;
        }
    }
    
    if (sorted) {
        printf("PASS: Large structs sorted by key\n");
    } else {
        printf("FAIL: Large structs not sorted correctly\n");
    }
    
    free(big_arr);
}

int main() {
    printf("===== Starting Heap Sort Tests =====\n");
    
    test_int_sort();
    test_struct_sort();
    
    printf("\n===== All Tests Completed =====\n");
    return 0;
}