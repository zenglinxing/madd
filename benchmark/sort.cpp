// coding: utf-8
#include<stdlib.h>
#include<stdint.h>
#include<benchmark/benchmark.h>
#include"../madd.h"

#define N_LEN 4

int64_t lengths[N_LEN] = {1<<5, 1<<8, 1<<10, 1<<16/*, 1<<20*/};
uint64_t *arr[N_LEN], *arr_temp, seed = 10;
RNG_Param rng;

class Sort_Bench_Init{
    public:
        Sort_Bench_Init(uint64_t seed){
            rng = RNG_Init(seed, RNG_XOSHIRO256SS);
            uint64_t i, j, max_length = 0;
            for (i=0; i<N_LEN; i++){
                arr[i] = (uint64_t*)malloc(lengths[i]*sizeof(uint64_t));
                for (j=0; j<lengths[i]; j++){
                    arr[i][j] = Rand_Uint(&rng);
                }
                if (lengths[i] > max_length){
                    max_length = lengths[i];
                }
            }
            arr_temp = (uint64_t*)calloc(max_length, sizeof(uint64_t));
        }
};
static Sort_Bench_Init sort_bench_init(seed);

static void Custom_Sort_Input(benchmark::internal::Benchmark* b)
{
    for (int64_t id=0; id<N_LEN; id++){
        b->Args({id, lengths[id]});
    }
}

static char Compare_uint64(void *a, void *b, void *other_param)
{
    uint64_t *aa = (uint64_t*)a, *bb = (uint64_t*)b;
    if (*aa == *bb) return MADD_SAME;
    else if (*aa < *bb) return MADD_LESS;
    else return MADD_GREATER;
}

static uint64_t get_uint64(void *a, void *other_param)
{
    return *(uint64_t*)a;
}

#define SORT_BENCH(Sort_func, compare_func) \
{ \
    for (auto _ : state){ \
        state.PauseTiming(); \
        int64_t id_arr = state.range(0), len = state.range(1); \
        memcpy(arr_temp, arr[id_arr], len*sizeof(uint64_t)); \
        state.ResumeTiming(); \
        Sort_func(len, sizeof(uint64_t), arr_temp, compare_func, NULL); \
    } \
} \

// counting sort
static void counting_sort(benchmark::State& state)
SORT_BENCH(Sort_Counting, get_uint64)

BENCHMARK(counting_sort)->Apply(Custom_Sort_Input);

// heap sort
static void heap_sort(benchmark::State& state)
SORT_BENCH(Sort_Heap, Compare_uint64)

BENCHMARK(heap_sort)->Apply(Custom_Sort_Input);

// insertion sort
static void insertion_sort(benchmark::State& state)
SORT_BENCH(Sort_Insertion, Compare_uint64)

BENCHMARK(insertion_sort)->Apply(Custom_Sort_Input);

// merge sort
static void merge_sort(benchmark::State& state)
SORT_BENCH(Sort_Merge, Compare_uint64)

BENCHMARK(merge_sort)->Apply(Custom_Sort_Input);

// quick sort
static void quick_sort(benchmark::State& state)
SORT_BENCH(Sort_Quicksort, Compare_uint64)

BENCHMARK(quick_sort)->Apply(Custom_Sort_Input);

// shell sort
static void shell_sort(benchmark::State& state)
SORT_BENCH(Sort_Shell, Compare_uint64)

BENCHMARK(shell_sort)->Apply(Custom_Sort_Input);

BENCHMARK_MAIN();