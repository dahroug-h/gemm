#include "immintrin.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

#define MC 192
#define NC 2048
#define KC 384
#define min(a,b) (((a)<(b))?(a):(b))
#define  micro_kernel_8x8\
    a8  = _mm_loadl_epi64((__m128i*) &A(i,k));\
    a   = _mm256_cvtepi8_epi32(a8);\
    b0 = _mm256_set1_epi32((int32_t)B(k,j));\
    b1 = _mm256_set1_epi32((int32_t)B(k,j+1));\
    b2 = _mm256_set1_epi32((int32_t)B(k,j+2));\
    b3 = _mm256_set1_epi32((int32_t)B(k,j+3));\
    b4 = _mm256_set1_epi32((int32_t)B(k,j+4));\
    b5 = _mm256_set1_epi32((int32_t)B(k,j+5));\
    b6 = _mm256_set1_epi32((int32_t)B(k,j+6));\
    b7 = _mm256_set1_epi32((int32_t)B(k,j+7));\
    c0 = _mm256_add_epi32(c0,_mm256_mullo_epi32(a, b0));\
    c1 = _mm256_add_epi32(c1,_mm256_mullo_epi32(a, b1));\
    c2 = _mm256_add_epi32(c2,_mm256_mullo_epi32(a, b2));\
    c3 = _mm256_add_epi32(c3,_mm256_mullo_epi32(a, b3));\
    c4 = _mm256_add_epi32(c4,_mm256_mullo_epi32(a, b4));\
    c5 = _mm256_add_epi32(c5,_mm256_mullo_epi32(a, b5));\
    c6 = _mm256_add_epi32(c6,_mm256_mullo_epi32(a, b6));\
    c7 = _mm256_add_epi32(c7,_mm256_mullo_epi32(a, b7));\
    k++;
   

void padding(int32_t M, int32_t N, int32_t K, int8_t *A, int LDA, int8_t *B, int LDB, int32_t *C, int LDC){
    int i,j,k;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            int32_t acc = C(i,j);
            for (k=0;k<K;k++){
                acc+= A(i,k)*B(k,j);
            }
            C(i,j) = acc;
        }
    }
}


 void macro_kernel(int32_t M,int32_t N,int32_t K,int8_t* A,int LDA, int8_t* B,int LDB, int32_t* C,int LDC){
    int i, j, k;
    __m128i a8;
    __m256i a, b0, b1, b2, b3, b4, b5, b6, b7, c0, c1, c2, c3, c4, c5, c6, c7;
    int M8=M&-8;
    int N8=N&-8;
    int K8=K&-8;

    for (j = 0; j < N8; j+= 8) {
        for (i = 0; i < M8; i+= 8) {
            c0 = _mm256_setzero_si256();
            c1 = _mm256_setzero_si256();
            c2 = _mm256_setzero_si256();
            c3 = _mm256_setzero_si256();
            c4 = _mm256_setzero_si256();
            c5 = _mm256_setzero_si256();
            c6 = _mm256_setzero_si256();
            c7 = _mm256_setzero_si256();
            for (k = 0; k < K8; ) {
                micro_kernel_8x8
                micro_kernel_8x8
                micro_kernel_8x8
                micro_kernel_8x8
                micro_kernel_8x8
                micro_kernel_8x8
                micro_kernel_8x8
                micro_kernel_8x8
            }
            for (k=K8;k<K;){
                micro_kernel_8x8
            }
            _mm256_storeu_si256((__m256i_u*)&C(i,j), _mm256_add_epi32(c0, _mm256_loadu_si256((__m256i*)&C(i,j))));
            _mm256_storeu_si256((__m256i_u*)&C(i,j+1), _mm256_add_epi32(c1, _mm256_loadu_si256((__m256i*)&C(i,j+1))));
            _mm256_storeu_si256((__m256i_u*)&C(i,j+2), _mm256_add_epi32(c2, _mm256_loadu_si256((__m256i*)&C(i,j+2))));
            _mm256_storeu_si256((__m256i_u*)&C(i,j+3), _mm256_add_epi32(c3, _mm256_loadu_si256((__m256i*)&C(i,j+3))));
            _mm256_storeu_si256((__m256i_u*)&C(i,j+4), _mm256_add_epi32(c4, _mm256_loadu_si256((__m256i*)&C(i,j+4))));
            _mm256_storeu_si256((__m256i_u*)&C(i,j+5), _mm256_add_epi32(c5, _mm256_loadu_si256((__m256i*)&C(i,j+5))));
            _mm256_storeu_si256((__m256i_u*)&C(i,j+6), _mm256_add_epi32(c6, _mm256_loadu_si256((__m256i*)&C(i,j+6))));
            _mm256_storeu_si256((__m256i_u*)&C(i,j+7), _mm256_add_epi32(c7, _mm256_loadu_si256((__m256i*)&C(i,j+7))));
        }
    }   
    if(M!=M8) padding(M-M8,N,K,&A(M8,0),LDA,B,LDB,&C(M8,0),LDC);
    if(N!=N8) padding(M8,N-N8,K,A,LDA,&B(0,N8),LDB,&C(0,N8),LDC);
 }

 void kernel(int32_t M,int32_t N,int32_t K,int8_t* A,int LDA, int8_t* B,int LDB, int32_t* C,int LDC){
    
    
    
    for (int j = 0; j < N; j += NC) {
        int nc = min(N-j,NC);
        for(int p = 0; p < K; p += KC) {
            int kc = min(K-p,KC);
            for(int i = 0; i < M; i += MC) {
                int mc = min(M-i,MC);
                for(int jr = 0; jr < nc; jr += 8) {
                    int nr = min(nc-jr,8);
                    for(int ir = 0; ir < mc; ir += 8) {
                        int mr = min(mc-ir,8);
                        macro_kernel(mr,nr,kc,&A(i+ir,p),LDA,&B(p,j+jr),LDB,&C(i+ir,j+jr),LDC);
                    }
                }
            }
        }
    }
}