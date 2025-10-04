// coding: utf-8
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include<benchmark/benchmark.h>
#include"../madd.h"

#define N_LEN 3

int64_t lengths[N_LEN] = {10, 100, 1000};
uint64_t seed = 10, n_vec = 3;
Cnum *arr_sample[N_LEN], *vec_sample[N_LEN];
RNG_Param rng;
void *temp_space;

class Linear_Equations_Init{
    public:
        Linear_Equations_Init(uint64_t seed){
            rng = RNG_Init(seed, RNG_XOSHIRO256SS);
            uint64_t i, j, max_size = 0, max_len = 0;
            for (i=0; i<N_LEN; i++){
                int64_t len = lengths[i];
                arr_sample[i] = (Cnum*)malloc((len * len + len * n_vec) * sizeof(Cnum));
                for (j=0; j<len * len + len * n_vec; j++){
                    arr_sample[i][j].real = Rand(&rng);
                    arr_sample[i][j].imag = Rand(&rng);
                }
                if (len * len > max_size){
                    max_size = len * len;
                    max_len = len;
                }
            }
            temp_space = calloc(max_size + max_len * n_vec, sizeof(Cnum));
        }
};
static Linear_Equations_Init linear_equations_init(seed);

static void Custom_Linear_Equations_Input(benchmark::internal::Benchmark* b)
{
    for (int64_t id=0; id<N_LEN; id++){
        b->Args({id, lengths[id]});
    }
}

#define LINEAR_EQUATION_BENCH(num_type, Linear_Equations) \
{ \
    for (auto _ : state){ \
        /* init matrix & vectors */ \
        state.PauseTiming(); \
        int64_t id_arr = state.range(0), len = state.range(1); \
        num_type *arr = (num_type*)temp_space, *vec = arr + len * len; \
        for (int64_t i=0; i<len*len + len*n_vec; i++){ \
            arr[i] = arr_sample[id_arr][i].real; \
        } \
        state.ResumeTiming(); \
        /* linear equations */ \
        Linear_Equations(len, arr, n_vec, vec); \
    } \
} \

#define LINEAR_EQUATION_CNUM_BENCH(Cnum, Linear_Equations_c64) \
{ \
    for (auto _ : state){ \
        /* init matrix & vectors */ \
        state.PauseTiming(); \
        int64_t id_arr = state.range(0), len = state.range(1); \
        Cnum *arr = (Cnum*)temp_space, *vec = arr + len * len; \
        for (int64_t i=0; i<len*len + len*n_vec; i++){ \
            arr[i].real = arr_sample[id_arr][i].real; \
            arr[i].imag = arr_sample[id_arr][i].imag; \
        } \
        state.ResumeTiming(); \
        /* linear equations */ \
        Linear_Equations_c64(len, arr, n_vec, vec); \
    } \
} \

static void linear_equations_f64(benchmark::State& state)
LINEAR_EQUATION_BENCH(double, Linear_Equations)

BENCHMARK(linear_equations_f64)->Apply(Custom_Linear_Equations_Input);

static void linear_equations_f32(benchmark::State& state)
LINEAR_EQUATION_BENCH(float, Linear_Equations_f32)

BENCHMARK(linear_equations_f32)->Apply(Custom_Linear_Equations_Input);


static void linear_equations_c64(benchmark::State& state)
LINEAR_EQUATION_CNUM_BENCH(Cnum, Linear_Equations_c64)

BENCHMARK(linear_equations_c64)->Apply(Custom_Linear_Equations_Input);

static void linear_equations_c32(benchmark::State& state)
LINEAR_EQUATION_CNUM_BENCH(Cnum32, Linear_Equations_c32)

BENCHMARK(linear_equations_c32)->Apply(Custom_Linear_Equations_Input);

// CUDA
#ifdef ENABLE_CUDA
static void linear_equations_cuda_f64(benchmark::State& state)
LINEAR_EQUATION_BENCH(double, Linear_Equations_cuda)

BENCHMARK(linear_equations_cuda_f64)->Apply(Custom_Linear_Equations_Input);

static void linear_equations_cuda_f32(benchmark::State& state)
LINEAR_EQUATION_BENCH(float, Linear_Equations_cuda_f32)

BENCHMARK(linear_equations_cuda_f32)->Apply(Custom_Linear_Equations_Input);


static void linear_equations_cuda_c64(benchmark::State& state)
LINEAR_EQUATION_CNUM_BENCH(Cnum, Linear_Equations_cuda_c64)

BENCHMARK(linear_equations_cuda_c64)->Apply(Custom_Linear_Equations_Input);

static void linear_equations_cuda_c32(benchmark::State& state)
LINEAR_EQUATION_CNUM_BENCH(Cnum32, Linear_Equations_cuda_c32)

BENCHMARK(linear_equations_cuda_c32)->Apply(Custom_Linear_Equations_Input);

#ifdef CUDA_11_1
static void linear_equations_cuda64_f64(benchmark::State& state)
LINEAR_EQUATION_BENCH(double, Linear_Equations_cuda64)

BENCHMARK(linear_equations_cuda64_f64)->Apply(Custom_Linear_Equations_Input);

static void linear_equations_cuda64_f32(benchmark::State& state)
LINEAR_EQUATION_BENCH(float, Linear_Equations_cuda64_f32)

BENCHMARK(linear_equations_cuda64_f32)->Apply(Custom_Linear_Equations_Input);


static void linear_equations_cuda64_c64(benchmark::State& state)
LINEAR_EQUATION_CNUM_BENCH(Cnum, Linear_Equations_cuda64_c64)

BENCHMARK(linear_equations_cuda64_c64)->Apply(Custom_Linear_Equations_Input);

static void linear_equations_cuda64_c32(benchmark::State& state)
LINEAR_EQUATION_CNUM_BENCH(Cnum32, Linear_Equations_cuda64_c32)

BENCHMARK(linear_equations_cuda64_c32)->Apply(Custom_Linear_Equations_Input);
#endif /* CUDA_11_1 */
#endif /* ENABLE_CUDA */

BENCHMARK_MAIN();