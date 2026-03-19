#include "immintrin.h"

#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

#ifndef MC
extern int MC, NC, KC;
#endif

#ifndef NTHREADS
    #define NTHREADS 8
#endif

#ifndef OMP_SCHEDULE
    #define OMP_SCHEDULE auto
#endif

#define PRAGMA_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(OMP_SCHEDULE) num_threads(NTHREADS)")

#define MC_PADDED(mc) (((mc + 5) / 6) * 6)
#define NC_PADDED(nc) (((nc + 15) / 16) * 16)

#define min(a,b) (((a)<(b))?(a):(b))

#define micro_kernel_6x16 \
   _mm_prefetch((const char*)(pa + 192), _MM_HINT_T1); \
    _mm_prefetch((const char*)(pb + 256), _MM_HINT_T1); \
    __m256i b0 = _mm256_loadu_si256((__m256i*)pb); pb += 32; \
    __m256i b1 = _mm256_loadu_si256((__m256i*)pb); pb += 32; \
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
    k += 4; 


/*static int8_t Buffer_A[(MC_PADDED+6) * KC] __attribute__((aligned(64)));
#static int8_t Buffer_B[(NC_PADDED+16) * KC] __attribute__((aligned(64)));*/

void pack_A(int8_t* A, int8_t* Buffer_A, int mc, int kc, int row_start, int col_start, int LDA) {
    for (int i = 0; i < mc; i += 6) {
        int mr = min(6, mc - i);
        for (int p = 0; p < kc; p += 4) { 
            for (int r = 0; r < mr; r++) {
                *Buffer_A++ = (p + 0 < kc) ? A(row_start + i + r, col_start + p + 0) + 128 : 128;
                *Buffer_A++ = (p + 1 < kc) ? A(row_start + i + r, col_start + p + 1) + 128 : 128;
                *Buffer_A++ = (p + 2 < kc) ? A(row_start + i + r, col_start + p + 2) + 128 : 128;
                *Buffer_A++ = (p + 3 < kc) ? A(row_start + i + r, col_start + p + 3) + 128 : 128;
            }
            for (int r = mr; r < 6; r++) {
                *Buffer_A++ = 0; *Buffer_A++ = 0; *Buffer_A++ = 0; *Buffer_A++ = 0;
            }
        }
    }
}

void pack_B(int8_t* B, int8_t* Buffer_B, int nc, int kc, int col_start, int row_start, int LDB, int32_t* B_col_correction) {
    for (int x = 0; x < nc; x++) {
        B_col_correction[col_start + x] = 0;
    }
    
    for (int j = 0; j < nc; j += 16) {
        int nr = min(16, nc - j);
        for (int p = 0; p < kc; p += 4) { 
            for (int i = 0; i < nr; i++) {
                int8_t v0 = (p + 0 < kc) ? B(row_start + p + 0, col_start + j + i) : 0;
                int8_t v1 = (p + 1 < kc) ? B(row_start + p + 1, col_start + j + i) : 0;
                int8_t v2 = (p + 2 < kc) ? B(row_start + p + 2, col_start + j + i) : 0;
                int8_t v3 = (p + 3 < kc) ? B(row_start + p + 3, col_start + j + i) : 0;
                
                *Buffer_B++ = v0; *Buffer_B++ = v1; *Buffer_B++ = v2; *Buffer_B++ = v3;
                
                B_col_correction[col_start + j + i] += (int32_t)v0 * 128 + (int32_t)v1 * 128 + (int32_t)v2 * 128 + (int32_t)v3 * 128;
            }
            for (int i = nr; i < 16; i++) {
                *Buffer_B++ = 0; *Buffer_B++ = 0; *Buffer_B++ = 0; *Buffer_B++ = 0;
            }
        }
    }
}

void macro_kernel(int32_t M, int32_t N, int32_t K, int8_t* A, int8_t* B, int32_t* C, int LDC) {
    int k;
    __m256i ones = _mm256_set1_epi16(1);

    __m256i c0 = _mm256_setzero_si256(); __m256i c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256(); __m256i c3 = _mm256_setzero_si256();
    __m256i c4 = _mm256_setzero_si256(); __m256i c5 = _mm256_setzero_si256();
    __m256i c6 = _mm256_setzero_si256(); __m256i c7 = _mm256_setzero_si256();
    __m256i c8 = _mm256_setzero_si256(); __m256i c9 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256(); __m256i c11 = _mm256_setzero_si256();

    int8_t* pa = A; 
    int8_t* pb = B;
    int K_padded = (K + 3) & ~3;
    for (int j = 0; j < 16; j++) {
        _mm_prefetch((const char*)&C[j * LDC], _MM_HINT_T0);
    }
    for (k = 0; k < K_padded; ) {
        micro_kernel_6x16
    }

    int32_t tmp[6][16] __attribute__((aligned(32)));
    _mm256_storeu_si256((__m256i*)&tmp[0][0], c0); _mm256_storeu_si256((__m256i*)&tmp[0][8], c1);
    _mm256_storeu_si256((__m256i*)&tmp[1][0], c2); _mm256_storeu_si256((__m256i*)&tmp[1][8], c3);
    _mm256_storeu_si256((__m256i*)&tmp[2][0], c4); _mm256_storeu_si256((__m256i*)&tmp[2][8], c5);
    _mm256_storeu_si256((__m256i*)&tmp[3][0], c6); _mm256_storeu_si256((__m256i*)&tmp[3][8], c7);
    _mm256_storeu_si256((__m256i*)&tmp[4][0], c8); _mm256_storeu_si256((__m256i*)&tmp[4][8], c9);
    _mm256_storeu_si256((__m256i*)&tmp[5][0], c10); _mm256_storeu_si256((__m256i*)&tmp[5][8], c11);

    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            C(r, c) += tmp[r][c];
        }
    }
}

void kernel(int32_t M, int32_t N, int32_t K, int8_t* A, int LDA, int8_t* B, int LDB, int32_t* C, int LDC) {
    

 

    int N_safe = ((N + 15) / 16) * 16;
    

    int32_t* B_col_correction = (int32_t*)malloc(N_safe * sizeof(int32_t));
    
    int8_t* Local_Buffer_A = (int8_t*)_mm_malloc((MC_PADDED(MC) + 6) * KC, 64);
    int8_t* Local_Buffer_B = (int8_t*)_mm_malloc((NC_PADDED(NC) + 16) * KC, 64);
    
    if (!Local_Buffer_A || !Local_Buffer_B || !B_col_correction) return;
    for (int j = 0; j < N; j += NC) {
        int nc = min(N-j, NC);
  
        for(int p = 0; p < K; p += KC) {
            int kc = min(K-p, KC);

            PRAGMA_OMP_PARALLEL_FOR
            for(int x = 0; x < nc; x++) B_col_correction[j + x] = 0;

            pack_B(B, Local_Buffer_B, nc, kc, j, p, LDB, B_col_correction);
            int kc_padded = (kc + 3) & ~3; 

            for(int i = 0; i < M; i += MC) {
                int mc = min(M-i, MC);
                pack_A(A, Local_Buffer_A, mc, kc, i, p, LDA);
               PRAGMA_OMP_PARALLEL_FOR
                for(int jr = 0; jr < nc; jr += 16) {
                    int nr = min(nc-jr, 16);
                    for(int ir = 0; ir < mc; ir += 6) {
                        int mr = min(mc-ir, 6);

                        // === التصحيح الجوهري هنا: نستخدم Local_Buffer وليس Buffer ===
                        macro_kernel(mr, nr, kc, 
                                     &Local_Buffer_A[ir * kc_padded], 
                                     &Local_Buffer_B[jr * kc_padded], 
                                     &C(i+ir, j+jr), LDC);

                        for (int r = 0; r < mr; r++) {
                            for (int c = 0; c < nr; c++) {
                                C(i + ir + r, j + jr + c) -= B_col_correction[j + jr + c];
                            }
                        }
                    }
                }
            }
        }
    }
    _mm_free(Local_Buffer_A);
    _mm_free(Local_Buffer_B);
    free(B_col_correction);
}