#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"
#include "kernel.h"
//#include "kernel_packed_8x8.h"
#include <stdbool.h>
#include "dnnl.h"
#include <string.h>
#include <omp.h>

void create_matrix_63(int8_t* A, int m, int n) {
    srand(time(NULL));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            A[j * m + i] = (int8_t)((rand() % 56) - 28);  
            
}
void create_matrix_127(int8_t* A, int m, int n) {
    srand(time(NULL));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            A[j * m + i] = (int8_t)((rand() % 256) - 127); 
}
void naive_matmul(int8_t* A, int8_t* B, int32_t* C, int m, int n, int k) {
    int overflow = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int32_t acc = 0;
            for (int v = 0; v < k; v++) {
                acc += A[v * m + i] * B[j * k + v];  
            }
            C[j * m + i] = acc; 
            }                      
        }
    }

bool Check_matrix(int32_t *A, int32_t *B, int n){
    int32_t diff = 0;
    int i;
    for (i = 0; B + i && A + i && i < n; i++){
        diff = abs(A[i] - B[i]);
        if (diff > 0) {
            printf("error. %5.2d,%5.2d,%d\n", A[i], B[i], i);
            return false;
        }
    }
    return true;
}

void main(int LDB,int LDC,int LDA) {
    

    FILE *fp = fopen("benchmark_results.csv", "w");
fprintf(fp, "Size,DNNL_GFLOPS,ME_GFLOPS\n");

    for (int i = 100; i <= 4000; i+=100) { 
        
        int m = i, n = i, k = i;
        int8_t ao = 0, bo = 0;
        int32_t oc = 0;
        int8_t* A = (int8_t*)malloc(m * k * sizeof(int8_t));
        int8_t* B = (int8_t*)malloc(k * n * sizeof(int8_t));
        int32_t* C_dnnl = (int32_t*)malloc(m * n * sizeof(int32_t));
        int32_t* C_naive = (int32_t*)malloc(m * n * sizeof(int32_t));
        int32_t* C_kernel = (int32_t*)malloc(m * n * sizeof(int32_t));

    create_matrix_63(A, m, k);
    create_matrix_63(B, k, n);
    memset(C_dnnl, 0, m * n * sizeof(int32_t));
    memset(C_naive, 0, m * n * sizeof(int32_t));
    memset(C_kernel, 0, m * n * sizeof(int32_t));

    clock_t start_dnnl = clock();

  

    dnnl_status_t status = dnnl_gemm_s8s8s32(
    'N', 'N', 'F',//transA, transB, offsetC
    (dnnl_dim_t)m, (dnnl_dim_t)n, (dnnl_dim_t)k,
    1.0f,//C := alpha * (op(A) - A_offset) * (op(B) - B_offset) + beta * C + C_offset
    A, m, ao,
    B, k, bo,
    0.0f,
    C_dnnl, m, &oc);

    //naive_matmul(A, B, C_naive, m, n, k);
    clock_t end_dnnl = clock();

    clock_t start_kernel = clock();

    kernel(m, n, k, A, m, B, k, C_kernel, m);

    clock_t end_kernel = clock();
    
   

    if (!Check_matrix((int32_t*)C_dnnl, (int32_t*)C_kernel, m * n)) {
        printf("====Failed====\n");
    } else {
        printf("====Passed====\n");
    }
    
    double time_dnnl = (double)(end_dnnl - start_dnnl) / CLOCKS_PER_SEC;
    double time_kernel = (double)(end_kernel - start_kernel) / CLOCKS_PER_SEC;
    double flops = 2.0 * m * n * k;
    double gflops = flops / (time_dnnl * 1e9);

    double gflops_k = flops / (time_kernel * 1e9);





printf("[[[[[ %d x %d x %d ]]]]]\n", m, n, k);
printf("DNNL: %f GFLOPS\n", gflops);
printf("ME: %f GFLOPS\n", gflops_k);


fprintf(fp, "%d,%f,%f\n", m, gflops, gflops_k);



    /*
    printf("[[[[[ %d x %d x %d ]]]]]\n", m, n, k);
    printf("DNNL: %f GFLOPS\n", gflops);
    printf("ME: %f GFLOPS\n", gflops_k);
*/
    free(A);
    free(B);
    free(C_dnnl);
    free(C_naive);
    free(C_kernel);  
}
fclose(fp);
printf("\nResults saved to benchmark_results.csv\n");
}






