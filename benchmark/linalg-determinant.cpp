// coding: utf-8
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<benchmark/benchmark.h>
#include"../madd.h"

#define N_LEN 4
int64_t lengths[N_LEN] = {(uint64_t)1e1, (uint64_t)5e1, (uint64_t)1e2, (uint64_t)1e3};

static void Custom_Determinant_Input(benchmark::internal::Benchmark* b)
{
    for (int64_t i=0; i<4; i++){
        b->Args({lengths[i]});
    }
}

#define DETERMINANT_BENCH(num_type, Determinant) \
{ \
    RNG_Param rng = RNG_Init(10, RNG_XOSHIRO256SS); \
    for (auto _ : state){ \
        state.PauseTiming(); \
        int64_t len = state.range(0), i; \
        num_type *arr = (num_type*)malloc(len * len * sizeof(num_type)), res; \
        for (i=0; i<len * len; i++){ \
            arr[i] = Rand(&rng); \
        } \
        state.ResumeTiming(); \
        Determinant(len, arr, &res); \
        state.PauseTiming(); \
        free(arr); \
        state.ResumeTiming(); \
    } \
} \

#define DETERMINANT_CNUM_BENCH(Cnum, Determinant_c64) \
{ \
    RNG_Param rng = RNG_Init(10, RNG_XOSHIRO256SS); \
    for (auto _ : state){ \
        state.PauseTiming(); \
        int64_t len = state.range(0), i; \
        Cnum *arr = (Cnum*)malloc(len * len * sizeof(Cnum)), res; \
        for (i=0; i<len * len; i++){ \
            arr[i].real = Rand(&rng); \
            arr[i].imag = Rand(&rng); \
        } \
        state.ResumeTiming(); \
        Determinant_c64(len, arr, &res); \
        state.PauseTiming(); \
        free(arr); \
        state.ResumeTiming(); \
    } \
} \

// double
static void determinant_f64(benchmark::State& state)
DETERMINANT_BENCH(double, Determinant)

BENCHMARK(determinant_f64)->Apply(Custom_Determinant_Input);

// float
static void determinant_f32(benchmark::State& state)
DETERMINANT_BENCH(float, Determinant_f32)

BENCHMARK(determinant_f32)->Apply(Custom_Determinant_Input);

// Cnum
static void determinant_c64(benchmark::State& state)
DETERMINANT_CNUM_BENCH(Cnum, Determinant_c64)

BENCHMARK(determinant_c64)->Apply(Custom_Determinant_Input);

// Cnum32
static void determinant_c32(benchmark::State& state)
DETERMINANT_CNUM_BENCH(Cnum32, Determinant_c32)

BENCHMARK(determinant_c32)->Apply(Custom_Determinant_Input);

#ifdef ENABLE_CUDA
// double
static void determinant_cuda_f64(benchmark::State& state)
DETERMINANT_BENCH(double, Determinant_cuda)

BENCHMARK(determinant_cuda_f64)->Apply(Custom_Determinant_Input);

// float
static void determinant_cuda_f32(benchmark::State& state)
DETERMINANT_BENCH(float, Determinant_cuda_f32)

BENCHMARK(determinant_cuda_f32)->Apply(Custom_Determinant_Input);

// Cnum
static void determinant_cuda_c64(benchmark::State& state)
DETERMINANT_CNUM_BENCH(Cnum, Determinant_cuda_c64)

BENCHMARK(determinant_cuda_c64)->Apply(Custom_Determinant_Input);

// Cnum32
static void determinant_cuda_c32(benchmark::State& state)
DETERMINANT_CNUM_BENCH(Cnum32, Determinant_cuda_c32)

BENCHMARK(determinant_cuda_c32)->Apply(Custom_Determinant_Input);
#endif

// main
BENCHMARK_MAIN();