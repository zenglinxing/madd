// coding: utf-8
#include<stdint.h>
#include<benchmark/benchmark.h>
#include"../madd.h"

static void Custom_RNG_Input(benchmark::internal::Benchmark* b)
{
    for (int64_t i=0; i<3; i++){
        int64_t seed = i * 10;
        int64_t len = 1e4;
        b->Args({seed, len});
        /*for (uint64_t j=0; j<4; j++){
            uint64_t len = pow(10, 4 + j);
            b->Args({seed, len});
        }*/
    }
}

#define RAND_BENCH_FUNC(rng_type) \
{ \
    for (auto _ : state){ \
        RNG_Param rng = RNG_Init(state.range(0), rng_type); \
        for (uint64_t i=0; i<state.range(1); i++){ \
            double r = Rand(&rng); \
        } \
    } \
} \

// Xoshiro256**
static void Xoshiro256ss_rand(benchmark::State& state)
RAND_BENCH_FUNC(RNG_XOSHIRO256SS)

BENCHMARK(Xoshiro256ss_rand)->Apply(Custom_RNG_Input);

// MT
static void MT_rand(benchmark::State& state)
RAND_BENCH_FUNC(RNG_MT)

BENCHMARK(MT_rand)->Apply(Custom_RNG_Input);

// C library
static void Clib_rand(benchmark::State& state)
RAND_BENCH_FUNC(RNG_CLIB)

BENCHMARK(Clib_rand)->Apply(Custom_RNG_Input);

// Xorshift64
static void Xorshift64_rand(benchmark::State& state)
RAND_BENCH_FUNC(RNG_XORSHIFT64)

BENCHMARK(Xorshift64_rand)->Apply(Custom_RNG_Input);

// Xorshift64*
static void Xorshift64s_rand(benchmark::State& state)
RAND_BENCH_FUNC(RNG_XORSHIFT64S)

BENCHMARK(Xorshift64s_rand)->Apply(Custom_RNG_Input);

// Xorshift1024*
static void Xorshift1024s_rand(benchmark::State& state)
RAND_BENCH_FUNC(RNG_XORSHIFT1024S)

BENCHMARK(Xorshift1024s_rand)->Apply(Custom_RNG_Input);

// Xoshiro256+
static void Xoshiro256p_rand(benchmark::State& state)
RAND_BENCH_FUNC(RNG_XOSHIRO256P)

BENCHMARK(Xoshiro256p_rand)->Apply(Custom_RNG_Input);

// Xorwow
static void Xorwow_rand(benchmark::State& state)
RAND_BENCH_FUNC(RNG_XORWOW)

BENCHMARK(Xorwow_rand)->Apply(Custom_RNG_Input);

#if defined(__x86_64__) || defined(_M_X64)
// x86
static void x86_rand(benchmark::State& state)
RAND_BENCH_FUNC(RNG_X86)

BENCHMARK(x86_rand)->Apply(Custom_RNG_Input);
#endif

BENCHMARK_MAIN();