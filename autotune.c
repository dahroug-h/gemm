/*
 * autotune.c  —  Fast parallel in-process auto-tuner (~2 min)
 *
 * Build:
 *   gcc autotune.c -O3 -march=core-avx2 -mavx2 -mno-avx512f \
 *       -fopenmp -DNTHREADS=$(nproc) -o autotune -lm -lpthread
 *   ./autotune
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

__thread int MC = 80;
__thread int NC = 1800;
__thread int KC = 4072;

#define MR 6
#define NR 16

#undef min
#include "kernel.h"

/* ── tuning config — kept minimal for speed ── */
#define COARSE_SIZE     1024
#define COARSE_REPEATS  1
#define COARSE_WARMUP   1

/* only 3 fine sizes — enough to capture the harmonic mean shape */
static const int TUNE_SIZES[] = { 512, 1024, 2048 };
static const int N_TUNE_SIZES = 3;
#define FINE_REPEATS  2
#define FINE_WARMUP   1

/* cap at 4 — more causes shared-cache contention on 2-core machines */
static int NPAR = 2;

static int read_cache_kb(int index)
{
    char path[128], buf[32];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return -1; }
    fclose(f);
    return atoi(buf);
}

static int round_down(int x, int m) { return (x / m) * m; }
static int clamp_val(int x, int lo, int hi){ return x<lo?lo:x>hi?hi:x; }

typedef struct {
    int mc, nc, kc, size, warmup, repeats, cpu_id;
    double gops;
} BenchTask;

static void *bench_thread(void *arg)
{
    BenchTask *t = (BenchTask*)arg;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(t->cpu_id % (int)sysconf(_SC_NPROCESSORS_ONLN), &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    MC = t->mc; NC = t->nc; KC = t->kc;
    int N = t->size;
    int8_t  *A = (int8_t* )malloc((size_t)N * N);
    int8_t  *B = (int8_t* )malloc((size_t)N * N);
    int32_t *C = (int32_t*)calloc((size_t)N * N, sizeof(int32_t));
    if (!A || !B || !C) { free(A); free(B); free(C); t->gops=-1.0; return NULL; }
    for (int i = 0; i < N*N; i++) {
        A[i]=(int8_t)((i*7+3)&63); B[i]=(int8_t)((i*5+1)&63);
    }
    for (int r = 0; r < t->warmup; r++) kernel(N,N,N,A,N,B,N,C,N);
    memset(C, 0, (size_t)N*N*sizeof(int32_t));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < t->repeats; r++) kernel(N,N,N,A,N,B,N,C,N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    t->gops = 2.0*(double)N*(double)N*(double)N*t->repeats/sec/1e9;
    free(A); free(B); free(C);
    return NULL;
}

static void run_batch(int (*combos)[3], double *results, int n,
                      int size, int warmup, int repeats)
{
    int i = 0;
    while (i < n) {
        int batch = (n-i)<NPAR?(n-i):NPAR;
        pthread_t threads[64]; BenchTask tasks[64];
        for (int b = 0; b < batch; b++) {
            tasks[b].mc=combos[i+b][0]; tasks[b].nc=combos[i+b][1];
            tasks[b].kc=combos[i+b][2]; tasks[b].size=size;
            tasks[b].warmup=warmup; tasks[b].repeats=repeats;
            tasks[b].cpu_id=b; tasks[b].gops=-1.0;
            pthread_create(&threads[b],NULL,bench_thread,&tasks[b]);
        }
        for (int b=0;b<batch;b++) {
            pthread_join(threads[b],NULL); results[i+b]=tasks[b].gops;
        }
        i += batch;
    }
}

static void score_fine_batch(int (*combos)[3], double *scores, int n)
{
    double *inv=(double*)calloc(n,sizeof(double));
    double *tmp=(double*)malloc(n*sizeof(double));
    for (int s=0;s<N_TUNE_SIZES;s++) {
        run_batch(combos,tmp,n,TUNE_SIZES[s],FINE_WARMUP,FINE_REPEATS);
        for (int i=0;i<n;i++) {
            if(tmp[i]<=0.0) inv[i]=1e18;
            else            inv[i]+=1.0/tmp[i];
        }
    }
    for (int i=0;i<n;i++)
        scores[i]=(inv[i]>=1e17)?-1.0:(double)N_TUNE_SIZES/inv[i];
    free(inv); free(tmp);
}

static void gen_kc(int l1d_kb,int l2_kb,int*out,int*n,int maxn){
    int ideal=(l1d_kb*1024)/NR,max_kc=(l2_kb*1024)/NR;
    int pct[]={25,60,100,175,300}; *n=0;
    for(int i=0;i<5;i++){
        int kc=round_down((int)((long long)ideal*pct[i]/100),MR);
        if(kc<MR)continue; if(kc>max_kc)kc=round_down(max_kc,MR);
        int dup=0; for(int j=0;j<*n;j++)if(out[j]==kc){dup=1;break;}
        if(!dup&&*n<maxn)out[(*n)++]=kc;
    }
}
static void gen_mc(int l2_kb,int l3_kb,int kc,int*out,int*n,int maxn){
    int ideal=(l2_kb*1024)/kc,max_mc=(l3_kb*1024)/kc/2;
    if(max_mc<ideal)max_mc=ideal*2;
    ideal=clamp_val(ideal,MR,max_mc);
    int pct[]={25,60,100,175}; *n=0;
    for(int i=0;i<4;i++){
        int mc=round_down((int)((long long)ideal*pct[i]/100),MR);
        if(mc<MR)continue; if(mc>max_mc)mc=round_down(max_mc,MR);
        int dup=0; for(int j=0;j<*n;j++)if(out[j]==mc){dup=1;break;}
        if(!dup&&*n<maxn)out[(*n)++]=mc;
    }
}
static void gen_nc(int l3_kb,int kc,int*out,int*n,int maxn){
    int ideal=(l3_kb*1024)/kc,max_nc=ideal*2;
    int pct[]={25,60,100,175}; *n=0;
    for(int i=0;i<4;i++){
        int nc=round_down((int)((long long)ideal*pct[i]/100),NR);
        if(nc<NR)continue; if(nc>max_nc)nc=round_down(max_nc,NR);
        int dup=0; for(int j=0;j<*n;j++)if(out[j]==nc){dup=1;break;}
        if(!dup&&*n<maxn)out[(*n)++]=nc;
    }
}
static void fine_neighbours(int center,int align,int lo,int hi,int*out,int*n){
    int step=round_down((int)((long long)center*125/1000),align);
    if(step<align)step=align; *n=0;
    for(int d=-2;d<=2;d++){
        int v=round_down(clamp_val(center+d*step,lo,hi),align);
        int dup=0; for(int j=0;j<*n;j++)if(out[j]==v){dup=1;break;}
        if(!dup&&*n<16)out[(*n)++]=v;
    }
}

int main(void)
{
    int ncpus=(int)sysconf(_SC_NPROCESSORS_ONLN);
    /* cap at 4 — sweet spot for parallel tuning without cache thrash */
    NPAR = ncpus < 4 ? ncpus : 4;

    printf("==============================================\n");
    printf("  int8 GEMM Auto-Tuner (fast, ~2 min)\n");
    printf("  MR=%d NR=%d  CPUs=%d  NPAR=%d\n", MR, NR, ncpus, NPAR);
    printf("  Tune sizes:");
    for(int s=0;s<N_TUNE_SIZES;s++) printf(" %d",TUNE_SIZES[s]);
    printf("\n==============================================\n\n");

    int l1d_kb=read_cache_kb(1),l2_kb=read_cache_kb(2),l3_kb=read_cache_kb(3);
    if(l1d_kb<=0){puts("L1D→48KB");l1d_kb=48;}
    if(l2_kb<=0){puts("L2→512KB");l2_kb=512;}
    if(l3_kb<=0){puts("L3→8192KB");l3_kb=8192;}
    printf("Cache: L1D=%dKB L2=%dKB L3=%dKB\n\n",l1d_kb,l2_kb,l3_kb);

    /* ── build coarse combo list ── */
    int kc_c[16],mc_c[16],nc_c[16],n_kc,n_mc,n_nc;
    gen_kc(l1d_kb,l2_kb,kc_c,&n_kc,16);
    int total_c=0;
    for(int ki=0;ki<n_kc;ki++){
        gen_mc(l2_kb,l3_kb,kc_c[ki],mc_c,&n_mc,16);
        gen_nc(l3_kb,kc_c[ki],nc_c,&n_nc,16);
        total_c+=n_mc*n_nc;
    }
    int (*cc)[3]=malloc(total_c*sizeof(*cc));
    double *cs=malloc(total_c*sizeof(double));
    int ci=0;
    for(int ki=0;ki<n_kc;ki++){
        int kc=kc_c[ki];
        gen_mc(l2_kb,l3_kb,kc,mc_c,&n_mc,16);
        gen_nc(l3_kb,kc,nc_c,&n_nc,16);
        for(int mi=0;mi<n_mc;mi++)
            for(int ni=0;ni<n_nc;ni++){
                cc[ci][0]=mc_c[mi];cc[ci][1]=nc_c[ni];cc[ci][2]=kc;ci++;
            }
    }

    /* ═══ PHASE 1 ═══ */
    printf("══ PHASE 1: %d combos at N=%d (NPAR=%d) ══\n",total_c,COARSE_SIZE,NPAR);
    run_batch(cc,cs,total_c,COARSE_SIZE,COARSE_WARMUP,COARSE_REPEATS);
    int c_best_mc=80,c_best_nc=1800,c_best_kc=4072; double c_best=-1.0;
    printf("%-7s %-7s %-7s  GOPS\n","MC","NC","KC");
    printf("----------------------------------------\n");
    for(int i=0;i<total_c;i++){
        printf("%-7d %-7d %-7d  ",cc[i][0],cc[i][1],cc[i][2]);
        if(cs[i]<0) printf("FAIL\n");
        else printf("%.2f%s\n",cs[i],cs[i]>c_best?" <- best":"");
        if(cs[i]>c_best){c_best=cs[i];c_best_mc=cc[i][0];c_best_nc=cc[i][1];c_best_kc=cc[i][2];}
    }
    printf("\nPhase 1 winner: MC=%d NC=%d KC=%d (%.2f GOPS)\n\n",
           c_best_mc,c_best_nc,c_best_kc,c_best);
    free(cc); free(cs);

    /* ═══ PHASE 2 ═══ */
    int kc_max=(l2_kb*1024)/NR;
    int mc_max=(l3_kb*1024)/c_best_kc/2;
    int nc_max=(l3_kb*1024)/c_best_kc*2;
    int kc_f[16],mc_f[16],nc_f[16],n_kc_f,n_mc_f,n_nc_f;
    fine_neighbours(c_best_kc,MR,MR,kc_max,kc_f,&n_kc_f);
    fine_neighbours(c_best_mc,MR,MR,mc_max,mc_f,&n_mc_f);
    fine_neighbours(c_best_nc,NR,NR,nc_max,nc_f,&n_nc_f);
    int total_f=n_kc_f*n_mc_f*n_nc_f;
    int (*fc)[3]=malloc(total_f*sizeof(*fc));
    double *fs=malloc(total_f*sizeof(double));
    int fi=0;
    for(int ki=0;ki<n_kc_f;ki++)
        for(int mi=0;mi<n_mc_f;mi++)
            for(int ni=0;ni<n_nc_f;ni++){
                fc[fi][0]=mc_f[mi];fc[fi][1]=nc_f[ni];fc[fi][2]=kc_f[ki];fi++;
            }
    printf("══ PHASE 2: %d fine combos (NPAR=%d) ══\n",total_f,NPAR);
    score_fine_batch(fc,fs,total_f);
    int best_mc=c_best_mc,best_nc=c_best_nc,best_kc=c_best_kc;
    double best=-1.0;
    {int tc[1][3]={{c_best_mc,c_best_nc,c_best_kc}};double ts[1];score_fine_batch(tc,ts,1);best=ts[0];}
    printf("%-7s %-7s %-7s  SCORE\n","MC","NC","KC");
    printf("----------------------------------------\n");
    printf("%-7d %-7d %-7d  %.2f [phase1 winner]\n",best_mc,best_nc,best_kc,best);
    for(int i=0;i<total_f;i++){
        printf("%-7d %-7d %-7d  ",fc[i][0],fc[i][1],fc[i][2]);
        if(fs[i]<0) printf("FAIL\n");
        else printf("%.2f%s\n",fs[i],fs[i]>best?" <- best":"");
        if(fs[i]>best){best=fs[i];best_mc=fc[i][0];best_nc=fc[i][1];best_kc=fc[i][2];}
    }
    free(fc); free(fs);

    printf("\n==============================================\n");
    printf("  FINAL BEST  (N=%d..%d)\n",TUNE_SIZES[0],TUNE_SIZES[N_TUNE_SIZES-1]);
    printf("==============================================\n");
    printf("  #define MC  %d\n",best_mc);
    printf("  #define NC  %d\n",best_nc);
    printf("  #define KC  %d\n",best_kc);
    printf("  Score = %.2f GOPS\n",best);
    printf("==============================================\n");

    FILE *out=fopen("/tmp/autotune_result.txt","w");
    if(out){fprintf(out,"%d %d %d\n",best_mc,best_nc,best_kc);fclose(out);
        printf("  Written to /tmp/autotune_result.txt\n");}
    return 0;
}