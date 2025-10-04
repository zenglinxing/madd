// coding: utf-8
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<benchmark/benchmark.h>
#include"../madd.h"

#define N_LEN 4
int64_t lengths[N_LEN] = {1e1, 5e1, 1e2, 1e3};

static void Custom_Eigen_Input(benchmark::internal::Benchmark* b)
{
    for (int64_t i=0; i<4; i++){
        b->Args({lengths[i]});
    }
}

#define EIGEN_BENCH(num_type, Cnum, Eigen) \
{ \
    for (auto _ : state){ \
        state.PauseTiming(); \
        int64_t len = state.range(0), i; \
        num_type *arr = (num_type*)malloc(len * len * sizeof(num_type)); \
        RNG_Param rng = RNG_Init(10, RNG_XOSHIRO256SS); \
        for (i=0; i<len * len; i++){ \
            arr[i] = Rand(&rng); \
        } \
        Cnum *eigenvalue = (Cnum*)malloc(len * sizeof(Cnum)); \
        Cnum *eigenvector_left = (Cnum*)malloc(len * len * sizeof(Cnum)); \
        Cnum *eigenvector_right = (Cnum*)malloc(len * len * sizeof(Cnum)); \
        state.ResumeTiming(); \
        Eigen(len, arr, eigenvalue, true, eigenvector_left, true, eigenvector_right); \
        state.PauseTiming(); \
        free(arr); \
        free(eigenvalue); \
        free(eigenvector_left); \
        free(eigenvector_right); \
        state.ResumeTiming(); \
    } \
} \

#define EIGEN_CNUM_BENCH(Cnum, Eigen) \
{ \
    for (auto _ : state){ \
        state.PauseTiming(); \
        int64_t len = state.range(0), i; \
        Cnum *arr = (Cnum*)malloc(len * len * sizeof(Cnum)); \
        RNG_Param rng = RNG_Init(10, RNG_XOSHIRO256SS); \
        for (i=0; i<len * len; i++){ \
            arr[i].real = Rand(&rng); \
            arr[i].imag = Rand(&rng); \
        } \
        Cnum *eigenvalue = (Cnum*)malloc(len * sizeof(Cnum)); \
        Cnum *eigenvector_left = (Cnum*)malloc(len * len * sizeof(Cnum)); \
        Cnum *eigenvector_right = (Cnum*)malloc(len * len * sizeof(Cnum)); \
        state.ResumeTiming(); \
        Eigen(len, arr, eigenvalue, true, eigenvector_left, true, eigenvector_right); \
        state.PauseTiming(); \
        free(arr); \
        free(eigenvalue); \
        free(eigenvector_left); \
        free(eigenvector_right); \
        state.ResumeTiming(); \
    } \
} \

static void eigen_f64(benchmark::State& state)
EIGEN_BENCH(double, Cnum, Eigen)

BENCHMARK(eigen_f64)->Apply(Custom_Eigen_Input);

static void eigen_f32(benchmark::State& state)
EIGEN_BENCH(float, Cnum32, Eigen_f32)

BENCHMARK(eigen_f32)->Apply(Custom_Eigen_Input);

static void eigen_c64(benchmark::State& state)
EIGEN_CNUM_BENCH(Cnum, Eigen_c64)

BENCHMARK(eigen_c64)->Apply(Custom_Eigen_Input);

static void eigen_c32(benchmark::State& state)
EIGEN_CNUM_BENCH(Cnum32, Eigen_c32)

BENCHMARK(eigen_c32)->Apply(Custom_Eigen_Input);

#ifdef CUDA_12_6
static void eigen_cuda_f64(benchmark::State& state)
EIGEN_BENCH(double, Cnum, Eigen_cuda64)

BENCHMARK(eigen_cuda_f64)->Apply(Custom_Eigen_Input);

static void eigen_cuda_f32(benchmark::State& state)
EIGEN_BENCH(float, Cnum32, Eigen_cuda64_f32)

BENCHMARK(eigen_cuda_f32)->Apply(Custom_Eigen_Input);

static void eigen_cuda_c64(benchmark::State& state)
EIGEN_CNUM_BENCH(Cnum, Eigen_cuda64_c64)

BENCHMARK(eigen_cuda_c64)->Apply(Custom_Eigen_Input);

static void eigen_cuda_c32(benchmark::State& state)
EIGEN_CNUM_BENCH(Cnum32, Eigen_cuda64_c32)

BENCHMARK(eigen_cuda_c32)->Apply(Custom_Eigen_Input);
#endif /* CUDA_12_6 */

BENCHMARK_MAIN();