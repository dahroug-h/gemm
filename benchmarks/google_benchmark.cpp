#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <omp.h>
#include "../src/kernel.h"
#include "dnnl.h"
#include <vector>
#include <ctime>

// ربط بارامترات الـ Autotune اللي طالعة من الـ YAML
#ifndef MC
extern __thread int MC, NC, KC;
#endif

// دالة تهيئة المصفوفات بنفس الرينج بتاعك [-28, 27]
void init_matrix(int8_t* A, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = (int8_t)((rand() % 56) - 28); 
    }
}

// ١. بنش مارك الكود بتاعك (ME)
static void BM_MyKernel(benchmark::State& state) {
    // القيم دي هيتم استبدالها وقت الـ Compile ببارامترات الـ Autotuner
    int N = state.range(0);
    int8_t* A = (int8_t*)_mm_malloc(N * N, 64);
    int8_t* B = (int8_t*)_mm_malloc(N * N, 64);
    int32_t* C = (int32_t*)_mm_malloc(N * N * sizeof(int32_t), 64);
    
    srand(time(NULL));
    init_matrix(A, N * N); 
    init_matrix(B, N * N);

    for (auto _ : state) {
        kernel(N, N, N, A, N, B, N, C, N);
        benchmark::DoNotOptimize(C);
    }

    // حساب الـ GFLOPS
    state.counters["GFLOPS"] = benchmark::Counter(
        (2.0 * N * N * N) / 1e9, benchmark::Counter::kIsIterationInvariantRate);

    _mm_free(A); _mm_free(B); _mm_free(C);
}

// ٢. بنش مارك إنتل (oneDNN)
static void BM_OneDNN(benchmark::State& state) {
    int N = state.range(0);
    int8_t* A = (int8_t*)_mm_malloc(N * N, 64);
    int8_t* B = (int8_t*)_mm_malloc(N * N, 64);
    int32_t* C = (int32_t*)_mm_malloc(N * N * sizeof(int32_t), 64);
    int8_t ao = 0, bo = 0; int32_t oc = 0;
    
    srand(time(NULL));
    init_matrix(A, N * N); 
    init_matrix(B, N * N);

    for (auto _ : state) {
        dnnl_gemm_s8s8s32('N', 'N', 'F', N, N, N, 1.0f, A, N, ao, B, N, bo, 0.0f, C, N, &oc);
        benchmark::DoNotOptimize(C);
    }

    state.counters["GFLOPS"] = benchmark::Counter(
        (2.0 * N * N * N) / 1e9, benchmark::Counter::kIsIterationInvariantRate);

    _mm_free(A); _mm_free(B); _mm_free(C);
}

// تجربة الأحجام من ١٠٠ لـ ٣٠٠٠ بنفس الـ Step بتاعك
BENCHMARK(BM_MyKernel)->DenseRange(100, 3000, 100)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_OneDNN)->DenseRange(100, 3000, 100)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();