Sort Functions
===

```C
void Sort_Counting(uint64_t n_element, size_t usize, void *arr_,
                   uint64_t get_key(void *element, void *other_param), void *other_param);

void Sort_Insertion(uint64_t n_element, size_t usize, void *arr_,
                    char func_compare(void *a1, void *a2, void *other_param), void *other_param);
void Sort_Merge(uint64_t n_element, size_t usize, void *arr_,
                char func_compare(void *a1, void *a2, void *other_param), void *other_param);
void Sort_Quicksort(uint64_t n_element, size_t usize, void *arr_,
                    char func_compare(void *a, void *b, void *other_param), void *other_param);
void Sort_Shell(uint64_t n_element, size_t usize, void *arr_,
                char func_compare(void*, void*, void*), void *other_param);
void Sort_Heap_Internal(uint64_t n, size_t usize, void *arr_,
                        char func_compare(void*, void*, void*), void *other_param,
                        void *ptemp);
void Sort_Heap(uint64_t n, size_t usize, void *arr_,
                    char func_compare(void*, void*, void*), void *other_param);
```

`arr_` is the pointer to your array. `usize` is the size of element.

The function `Sort_Heap_Internal` has one more parameter than `Sort_Heap` `*ptemp`.
I suppose you had prepare memory space of `*ptemp` by `usize` bytes.

get-key function to compare function
---

You may be upset with transfering get-key function to compare function.
Madd has already implements a function to replace your get-key to compare.

```C
typedef struct{
    uint64_t (*get_key_func)(void*, void*);
    void *other_param;
} Sort_Key_Func_to_Compare_Func_Param;

char Sort_Key_Func_to_Compare_Func(void *a, void *b, void *input_param);
```

# Example

```C
#include<madd.h>
uint64_t get_key(void *a, void *other_param)
{
    uint64_t *aa = (uint64_t)a;
    return *aa;
}

int main(int argc, char *argv[])
{
    uint64_t arr[4] = {1, 2, 3, 4};
    Sort_Counting(arr, 4, sizeof(uint64_t), NULL);
    
    Sort_Key_Func_to_Compare_Func_Param *param = {.get_key_func=get_key, .other_param=NULL};
    Sort_Merge(arr, 4, sizeof(uint64_t), Sort_Key_Func_to_Compare_Func, param);
    return 0;
}
```

Binary Search
---

```C
uint64_t Binary_Search(uint64_t n, size_t usize, void *arr_, void *element,
                       char func_compare(void *a, void *b, void *other_param), void *other_param);
uint64_t Binary_Search_Insert(uint64_t n, size_t usize, void *arr_, void *element,
                              char func_compare(void *a, void *b, void *other_param), void *other_param);
```

The binary-search is an efficient method to search for element in a **sorted** array.
`Binary_Search` searches for the element in array, and returns where it is.
If the element is not found, it will returns the possible (maybe not accurate) place to insert the element and pops warning to Madd.
`Binary_Search_Insert` returns where to insert the element.