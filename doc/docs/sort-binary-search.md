Binary Search
===

```C
uint64_t Binary_Search(uint64_t n, size_t usize, void *arr_, void *element,
                       char func_compare(void *a, void *b, void *other_param), void *other_param);
uint64_t Binary_Search_Insert(uint64_t n, size_t usize, void *arr_, void *element,
                              char func_compare(void *a, void *b, void *other_param), void *other_param);
```

| parameter | explanation |
|:---------:|:----------- |
| `n` | number of elements in `arr_` |
| `usize` | size of element in `arr_` |
| `arr_` | array pointer, **The array is supposed to have been sorted** |
| `element` | the element to search in `Binary_Search`<br>the element to insert in `Binary_Search_Insert` |
| `func_compare` | compare function |
| `other_param` | the 3rd parameter for `func_compare` |

**return value**

* `Binary_Search`: returns the index where the `element` places. If the `element` is not found, the return value should be `0`, and a warning will be popped.
* `Binary_Search_Insert`: returns the index where the `element` should be inserted. For example, if return `0`, then `element` should be inserted before the first element of `arr_`. If return `1`, then `element` should be inserted between the first and the second element of `arr_`. If return `n`, then `element` should be inserted after the end of `arr_`.