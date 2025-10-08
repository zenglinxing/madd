// coding: utf-8
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<benchmark/benchmark.h>
#include"../madd.h"

#define N_LEN 4
int64_t lengths[N_LEN] = {(uint64_t)1e1, (uint64_t)5e1, (uint64_t)1e2, (uint64_t)1e3};

static void Custom_Matrix_Inverse_Input(benchmark::internal::Benchmark* b)
{
    for (int64_t i=0; i<4; i++){
        b->Args({lengths[i]});
    }
}

#define MATRIX_INVERSE_BENCH(num_type, Matrix_Inverse) \
{ \
    RNG_Param rng = RNG_Init(10, 0); \
    for (auto _ : state){ \
        state.PauseTiming(); \
        int64_t len = state.range(0), i; \
        num_type *arr = (num_type*)malloc(len * len * sizeof(num_type)); \
        RNG_Param rng = RNG_Init(10, RNG_XOSHIRO256SS); \
        for (i=0; i<len * len; i++){ \
            arr[i] = Rand(&rng); \
        } \
        state.ResumeTiming(); \
        Matrix_Inverse(len, arr); \
        state.PauseTiming(); \
        free(arr); \
        state.ResumeTiming(); \
    } \
} \

#define MATRIX_INVERSE_CNUM_BENCH(num_type, Matrix_Inverse) \
{ \
    RNG_Param rng = RNG_Init(10, 0); \
    for (auto _ : state){ \
        state.PauseTiming(); \
        int64_t len = state.range(0), i; \
        num_type *arr = (num_type*)malloc(len * len * sizeof(num_type)); \
        RNG_Param rng = RNG_Init(10, RNG_XOSHIRO256SS); \
        for (i=0; i<len * len; i++){ \
            arr[i].real = Rand(&rng); \
            arr[i].imag = Rand(&rng); \
        } \
        state.ResumeTiming(); \
        Matrix_Inverse(len, arr); \
        state.PauseTiming(); \
        free(arr); \
        state.ResumeTiming(); \
    } \
} \

static void inverse_f64(benchmark::State& state)
MATRIX_INVERSE_BENCH(double, Matrix_Inverse)

BENCHMARK(inverse_f64)->Apply(Custom_Matrix_Inverse_Input);

static void inverse_f32(benchmark::State& state)
MATRIX_INVERSE_BENCH(float, Matrix_Inverse_f32)

BENCHMARK(inverse_f32)->Apply(Custom_Matrix_Inverse_Input);

static void inverse_c64(benchmark::State& state)
MATRIX_INVERSE_CNUM_BENCH(Cnum, Matrix_Inverse_c64)

BENCHMARK(inverse_c64)->Apply(Custom_Matrix_Inverse_Input);

static void inverse_c32(benchmark::State& state)
MATRIX_INVERSE_CNUM_BENCH(Cnum32, Matrix_Inverse_c32)

BENCHMARK(inverse_c32)->Apply(Custom_Matrix_Inverse_Input);

BENCHMARK_MAIN();