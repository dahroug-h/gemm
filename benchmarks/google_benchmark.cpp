#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <omp.h>
#include "../src/kernel.h" // Ensure this path is correct based on your repo

// Link the autotuned parameters used in your YAML
#ifndef MC
extern __thread int MC, NC, KC;
#endif

// Replicate your matrix generation logic
void init_matrix(int8_t* A, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = (int8_t)((rand() % 56) - 28);
    }
}

static void BM_Custom_GEMM(benchmark::State& state) {
    int N = state.range(0);
    int m = N, n = N, k = N;

    // Align memory to 64 bytes for SIMD efficiency
    int8_t* A = (int8_t*)_mm_malloc(m * k, 64);
    int8_t* B = (int8_t*)_mm_malloc(k * n, 64);
    int32_t* C = (int32_t*)_mm_malloc(m * n * sizeof(int32_t), 64);

    init_matrix(A, m * k);
    init_matrix(B, k * n);

    // Warmup is handled automatically by Google Benchmark
    for (auto _ : state) {
        kernel(m, n, k, A, m, B, k, C, m);
        benchmark::DoNotOptimize(C);
    }

    // GFLOPS calculation logic from your matmul.c
    double flops = 2.0 * m * n * k;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops / 1e9, benchmark::Counter::kIsIterationInvariantRate);

    _mm_free(A); _mm_free(B); _mm_free(C);
}

// Sweep the sizes shown in your performance plots (100 to 3000)
BENCHMARK(BM_Custom_GEMM)->DenseRange(100, 3000, 100)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();