#include <omp.h>
#include "immintrin.h"
#include <stdlib.h>

#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

#ifndef MC
extern __thread int MC, NC, KC;
#endif

#ifndef NTHREADS
    #define NTHREADS omp_get_max_threads()
#endif

#ifndef OMP_SCHEDULE
    #define OMP_SCHEDULE static
#endif

#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(OMP_SCHEDULE) num_threads(NTHREADS)")

#define MC_PADDED(mc) (((mc + 5) / 6) * 6)
#define NC_PADDED(nc) (((nc + 15) / 16) * 16)

#define min(a,b) (((a)<(b))?(a):(b))

#ifndef PREFETCH_A_L1
    #define PREFETCH_A_L1  192
#endif
#ifndef PREFETCH_B_L1
    #define PREFETCH_B_L1  256
#endif
#ifndef PREFETCH_A_L2
    #define PREFETCH_A_L2  768
#endif
#ifndef PREFETCH_B_L2
    #define PREFETCH_B_L2  1024
#endif

#define micro_kernel_6x16 \
    _mm_prefetch((const char*)(pa + PREFETCH_A_L1), _MM_HINT_T0); \
    _mm_prefetch((const char*)(pa + PREFETCH_A_L2), _MM_HINT_T1); \
    _mm_prefetch((const char*)(pb + PREFETCH_B_L1), _MM_HINT_T0); \
    _mm_prefetch((const char*)(pb + PREFETCH_B_L2), _MM_HINT_T1); \
    __m256i b0 = _mm256_load_si256((__m256i*)pb); pb += 32; \
    __m256i b1 = _mm256_load_si256((__m256i*)pb); pb += 32; \
    \
    __m256i a0 = _mm256_set1_epi32(*(int32_t*)(pa + 0)); \
    c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), ones)); \
    c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b1), ones)); \
    \
    __m256i a1 = _mm256_set1_epi32(*(int32_t*)(pa + 4)); \
    c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(_mm256_maddubs_epi16(a1, b0), ones)); \
    c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(_mm256_maddubs_epi16(a1, b1), ones)); \
    \
    __m256i a2 = _mm256_set1_epi32(*(int32_t*)(pa + 8)); \
    c4 = _mm256_add_epi32(c4, _mm256_madd_epi16(_mm256_maddubs_epi16(a2, b0), ones)); \
    c5 = _mm256_add_epi32(c5, _mm256_madd_epi16(_mm256_maddubs_epi16(a2, b1), ones)); \
    \
    __m256i a3 = _mm256_set1_epi32(*(int32_t*)(pa + 12)); \
    c6 = _mm256_add_epi32(c6, _mm256_madd_epi16(_mm256_maddubs_epi16(a3, b0), ones)); \
    c7 = _mm256_add_epi32(c7, _mm256_madd_epi16(_mm256_maddubs_epi16(a3, b1), ones)); \
    \
    __m256i a4 = _mm256_set1_epi32(*(int32_t*)(pa + 16)); \
    c8 = _mm256_add_epi32(c8, _mm256_madd_epi16(_mm256_maddubs_epi16(a4, b0), ones)); \
    c9 = _mm256_add_epi32(c9, _mm256_madd_epi16(_mm256_maddubs_epi16(a4, b1), ones)); \
    \
    __m256i a5 = _mm256_set1_epi32(*(int32_t*)(pa + 20)); \
    c10 = _mm256_add_epi32(c10, _mm256_madd_epi16(_mm256_maddubs_epi16(a5, b0), ones)); \
    c11 = _mm256_add_epi32(c11, _mm256_madd_epi16(_mm256_maddubs_epi16(a5, b1), ones)); \
    \
    pa += 24; \
    k  += 4;

void pack_A(int8_t* __restrict A, int8_t* __restrict Buffer_A, int mc, int kc,
            int row_start, int col_start, int LDA)
{
    for (int i = 0; i < mc; i += 6) {
        int mr = min(6, mc - i);
        int p = 0;
        
        // MAIN LOOP: No ternary branches, pure memory speed
        for (; p + 3 < kc; p += 4) {
            for (int r = 0; r < mr; r++) {
                *Buffer_A++ = A(row_start+i+r, col_start+p+0);
                *Buffer_A++ = A(row_start+i+r, col_start+p+1);
                *Buffer_A++ = A(row_start+i+r, col_start+p+2);
                *Buffer_A++ = A(row_start+i+r, col_start+p+3);
            }
            for (int r = mr; r < 6; r++) {
                *((int32_t*)Buffer_A) = 0; Buffer_A += 4;
            }
        }
        
        // FRINGE LOOP: Handle the remainders with safe bounds checking
        if (p < kc) {
            for (int r = 0; r < mr; r++) {
                *Buffer_A++ = (p+0<kc) ? A(row_start+i+r, col_start+p+0) : 0;
                *Buffer_A++ = (p+1<kc) ? A(row_start+i+r, col_start+p+1) : 0;
                *Buffer_A++ = (p+2<kc) ? A(row_start+i+r, col_start+p+2) : 0;
                *Buffer_A++ = (p+3<kc) ? A(row_start+i+r, col_start+p+3) : 0;
            }
            for (int r = mr; r < 6; r++) {
                *((int32_t*)Buffer_A) = 0; Buffer_A += 4;
            }
        }
    }
}

void pack_B(int8_t* __restrict B, int8_t* __restrict Buffer_B, int nc, int kc,
            int col_start, int row_start, int LDB)
{
    for (int j = 0; j < nc; j += 16) {
        int nr = min(16, nc - j);
        int p = 0;
        
        // MAIN LOOP: No ternary branches, no correction math
        for (; p + 3 < kc; p += 4) {
            for (int i = 0; i < nr; i++) {
                *Buffer_B++ = B(row_start+p+0, col_start+j+i);
                *Buffer_B++ = B(row_start+p+1, col_start+j+i);
                *Buffer_B++ = B(row_start+p+2, col_start+j+i);
                *Buffer_B++ = B(row_start+p+3, col_start+j+i);
            }
            for (int i = nr; i < 16; i++) {
                *((int32_t*)Buffer_B) = 0; Buffer_B += 4;
            }
        }
        
        // FRINGE LOOP: Handle the remainders
        if (p < kc) {
            for (int i = 0; i < nr; i++) {
                *Buffer_B++ = (p+0<kc) ? B(row_start+p+0, col_start+j+i) : 0;
                *Buffer_B++ = (p+1<kc) ? B(row_start+p+1, col_start+j+i) : 0;
                *Buffer_B++ = (p+2<kc) ? B(row_start+p+2, col_start+j+i) : 0;
                *Buffer_B++ = (p+3<kc) ? B(row_start+p+3, col_start+j+i) : 0;
            }
            for (int i = nr; i < 16; i++) {
                *((int32_t*)Buffer_B) = 0; Buffer_B += 4;
            }
        }
    }
}

static inline __attribute__((always_inline))
void macro_kernel(int32_t M, int32_t N, int32_t K,
                  int8_t* __restrict A, int8_t* __restrict B, int32_t* __restrict C, int LDC)
{
    int k;
    __m256i ones = _mm256_set1_epi16(1);
    __m256i c0  = _mm256_setzero_si256();
    __m256i c1  = _mm256_setzero_si256();
    __m256i c2  = _mm256_setzero_si256();
    __m256i c3  = _mm256_setzero_si256();
    __m256i c4  = _mm256_setzero_si256();
    __m256i c5  = _mm256_setzero_si256();
    __m256i c6  = _mm256_setzero_si256();
    __m256i c7  = _mm256_setzero_si256();
    __m256i c8  = _mm256_setzero_si256();
    __m256i c9  = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256();
    __m256i c11 = _mm256_setzero_si256();

    int8_t* pa = A;
    int8_t* pb = B;
    int K_padded = (K + 3) & ~3;

    for (int j = 0; j < 16; j++)
        _mm_prefetch((const char*)&C[j * LDC], _MM_HINT_T0);

    for (k = 0; k < K_padded; ) {
        micro_kernel_6x16
    }

    int32_t tmp[6][16] __attribute__((aligned(32)));
    _mm256_storeu_si256((__m256i*)&tmp[0][0], c0);
    _mm256_storeu_si256((__m256i*)&tmp[0][8], c1);
    _mm256_storeu_si256((__m256i*)&tmp[1][0], c2);
    _mm256_storeu_si256((__m256i*)&tmp[1][8], c3);
    _mm256_storeu_si256((__m256i*)&tmp[2][0], c4);
    _mm256_storeu_si256((__m256i*)&tmp[2][8], c5);
    _mm256_storeu_si256((__m256i*)&tmp[3][0], c6);
    _mm256_storeu_si256((__m256i*)&tmp[3][8], c7);
    _mm256_storeu_si256((__m256i*)&tmp[4][0], c8);
    _mm256_storeu_si256((__m256i*)&tmp[4][8], c9);
    _mm256_storeu_si256((__m256i*)&tmp[5][0], c10);
    _mm256_storeu_si256((__m256i*)&tmp[5][8], c11);

    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            C(r, c) += tmp[r][c];
}

void kernel(int32_t M, int32_t N, int32_t K,
            int8_t* __restrict A, int LDA, int8_t* __restrict B, int LDB,
            int32_t* __restrict C, int LDC)
{
    int8_t* Local_Buffer_A = (int8_t*)_mm_malloc((MC_PADDED(MC)+6)*KC, 64);
    int8_t* Local_Buffer_B = (int8_t*)_mm_malloc((NC_PADDED(NC)+16)*KC, 64);

    if (!Local_Buffer_A || !Local_Buffer_B) {
        if (Local_Buffer_A) _mm_free(Local_Buffer_A);
        if (Local_Buffer_B) _mm_free(Local_Buffer_B);
        return;
    }

    for (int j = 0; j < N; j += NC) {
        int nc = min(N-j, NC);

        for (int p = 0; p < K; p += KC) {
            int kc = min(K-p, KC);

            pack_B(B, Local_Buffer_B, nc, kc, j, p, LDB); 
            int kc_padded = (kc+3) & ~3;

            for (int i = 0; i < M; i += MC) {
                int mc = min(M-i, MC);
                pack_A(A, Local_Buffer_A, mc, kc, i, p, LDA);

                PRAGMA_OMP_PARALLEL_FOR
                for (int jr = 0; jr < nc; jr += 16) {
                    int nr = min(nc-jr, 16);
                    for (int ir = 0; ir < mc; ir += 6) {
                        int mr = min(mc-ir, 6);
                        macro_kernel(mr, nr, kc,
                            &Local_Buffer_A[ir*kc_padded],
                            &Local_Buffer_B[jr*kc_padded],
                            &C(i+ir, j+jr), LDC); 
                    }
                }
            }
        }
    }

    _mm_free(Local_Buffer_A);
    _mm_free(Local_Buffer_B);
}