Sort Algorithm Introduction
===

There are multiple prevalent sort algorithm implemented in Madd. As listed in Table

| sort type | algorithm | time complexity (best) | time complexity (worst) | space complexity | stability |
| --------- | --------- | ---------------------- | ----------------------- | ---------------- | --------- |
| integer | counting sort | $O(n+k)$ | $O(n+k)$ | $O(k)$ | Y |
| --------- | --------- | ---------------------- | ----------------------- | ---------------- | --------- |
| compare | heap sort | $O(n\log n)$ | $O(n\log n)$ | $O(1)$ | N |
|  | insertion sort | $O(n)$ | $O(n^{2})$ | $O(1)$ | Y |
|  | merge sort | $O(n\log n)$ | $O(n\log n)$ | $O(n)$ | Y |
|  | quick sort | $O(n\log n)$ | $O(n^{2})$ | $O(\log n)$ | Y |
|  | shell sort | $O(n^{7/6})$ | $O(n^{4/3})$ | $O(1)$ | N |

The sort type indicate the object that the algorithm to sort. Sort type of *integer* means the algorithm only sort the integer, thus you need a function to translate your element to an integer. While *compare* means you need a comparation function to judge less, equal or greater. **Before your further reading, please keep in mind that you shall prepare the following function according to your sorting problem (sort type).** The `compare_func` should return `true` if `element1` is less than / equal to `element2`.

```C
// if your sort type is integer
uint64_t get_key(void *element, void *other_param);

// if your sort type is compare
bool compare_func(void *element1, void *element2, void *other_param);
```