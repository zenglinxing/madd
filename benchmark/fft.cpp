// coding: utf-8
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<benchmark/benchmark.h>
#include"../madd.h"

static void Custom_FFT_Input(benchmark::internal::Benchmark* b)
{
    for (int64_t i=0; i<3; i++){
        int64_t len = (int64_t)pow(10, 2 + 2 * i);
        b->Args({len});
    }
    for (int64_t i=0; i<3; i++){
        int64_t len = 1 << (8 + 8 * i);
        b->Args({len});
    }
}

#define FFT_TEST(Cnum, Rand, Fast_Fourier_Transform) \
{ \
    for (auto _ : state){ \
        state.PauseTiming(); \
        int64_t len = state.range(0), i; \
        Cnum *arr = (Cnum*)malloc(len * sizeof(Cnum)); \
        RNG_Param rng = RNG_Init(10, RNG_XOSHIRO256SS); \
        for (i=0; i<len; i++){ \
            arr[i].real = Rand(&rng); \
            arr[i].imag = Rand(&rng); \
        } \
        state.ResumeTiming(); \
        Fast_Fourier_Transform(len, arr, MADD_FFT_FORWARD); \
        state.PauseTiming(); \
        free(arr); \
        state.ResumeTiming(); \
    } \
} \

static void fft_c64(benchmark::State& state)
FFT_TEST(Cnum, Rand, Fast_Fourier_Transform)

BENCHMARK(fft_c64)->Apply(Custom_FFT_Input);

static void fft_c32(benchmark::State& state)
FFT_TEST(Cnum32, Rand_f32, Fast_Fourier_Transform_c32)

BENCHMARK(fft_c32)->Apply(Custom_FFT_Input);

#ifdef ENABLE_CUDA
static void fft_cuda_c64(benchmark::State& state)
FFT_TEST(Cnum, Rand, Fast_Fourier_Transform_cuda)

BENCHMARK(fft_cuda_c64)->Apply(Custom_FFT_Input);

static void fft_cuda_c32(benchmark::State& state)
FFT_TEST(Cnum32, Rand_f32, Fast_Fourier_Transform_cuda_c32)

BENCHMARK(fft_cuda_c32)->Apply(Custom_FFT_Input);
#endif /* ENABLE_CUDA */

BENCHMARK_MAIN();