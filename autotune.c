/*
 * autotune.c  —  Fast parallel in-process auto-tuner (~4 min)
 *
 * Phase 1: coarse MC/NC/KC grid at N=1024
 * Phase 2: fine MC/NC/KC around winner, harmonic mean N=512-2048
 * Phase 3: prefetch distance sweep with best MC/NC/KC fixed
 *          25 combos (5 PA × 5 PB), harmonic mean N=512-2048
 *
 * Writes to /tmp/autotune_result.txt:
 *   MC NC KC PA_L1 PB_L1 PA_L2 PB_L2
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

/* prefetch distances — thread-local for parallel sweep */
__thread int PFETCH_A_L1 = 192;
__thread int PFETCH_B_L1 = 512;
__thread int PFETCH_A_L2 = 768;
__thread int PFETCH_B_L2 = 2048;

#define MR 6
#define NR 16

/* override kernel.h prefetch defines with thread-local values */
#define PREFETCH_A_L1 PFETCH_A_L1
#define PREFETCH_B_L1 PFETCH_B_L1
#define PREFETCH_A_L2 PFETCH_A_L2
#define PREFETCH_B_L2 PFETCH_B_L2

#undef min
#include "kernel.h"

/* ── config ── */
#define COARSE_SIZE     1024
#define COARSE_REPEATS  1
#define COARSE_WARMUP   1

static const int TUNE_SIZES[] = { 512, 1024, 2048 };
static const int N_TUNE_SIZES = 3;
#define FINE_REPEATS  2
#define FINE_WARMUP   1

/* prefetch search space */
static const int PA_L1_VALS[] = { 96, 192, 384, 768, 1536 };
static const int PB_L1_VALS[] = { 128, 256, 512, 1024, 2048 };
static const int N_PA = 5;
static const int N_PB = 5;

static int NPAR = 2;

/* ─────────────────────────────────────────────────────────────────
 * Cache detection
 * ──────────────────────────────────────────────────────────────── */
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

static int round_down(int x, int m) { return (x/m)*m; }
static int clamp_val(int x, int lo, int hi){ return x<lo?lo:x>hi?hi:x; }

/* ─────────────────────────────────────────────────────────────────
 * Per-thread task — supports both MC/NC/KC and prefetch tuning
 * ──────────────────────────────────────────────────────────────── */
typedef struct {
    int mc, nc, kc;
    int pa_l1, pb_l1, pa_l2, pb_l2;
    int size, warmup, repeats, cpu_id;
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
    PFETCH_A_L1 = t->pa_l1; PFETCH_B_L1 = t->pb_l1;
    PFETCH_A_L2 = t->pa_l2; PFETCH_B_L2 = t->pb_l2;

    int N = t->size;
    int8_t  *A = (int8_t* )malloc((size_t)N*N);
    int8_t  *B = (int8_t* )malloc((size_t)N*N);
    int32_t *C = (int32_t*)calloc((size_t)N*N, sizeof(int32_t));
    if (!A||!B||!C){free(A);free(B);free(C);t->gops=-1.0;return NULL;}
    for (int i=0;i<N*N;i++){
        A[i]=(int8_t)((i*7+3)&63); B[i]=(int8_t)((i*5+1)&63);
    }
    for (int r=0;r<t->warmup;r++) kernel(N,N,N,A,N,B,N,C,N);
    memset(C,0,(size_t)N*N*sizeof(int32_t));
    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for (int r=0;r<t->repeats;r++) kernel(N,N,N,A,N,B,N,C,N);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double sec=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    t->gops=2.0*(double)N*(double)N*(double)N*t->repeats/sec/1e9;
    free(A);free(B);free(C);
    return NULL;
}

static void run_batch_full(BenchTask *tasks, int n)
{
    int i=0;
    while(i<n){
        int batch=(n-i)<NPAR?(n-i):NPAR;
        pthread_t threads[64];
        for(int b=0;b<batch;b++){
            tasks[i+b].cpu_id=b;
            pthread_create(&threads[b],NULL,bench_thread,&tasks[i+b]);
        }
        for(int b=0;b<batch;b++) pthread_join(threads[b],NULL);
        i+=batch;
    }
}

/* ─────────────────────────────────────────────────────────────────
 * MC/NC/KC scoring helpers
 * ──────────────────────────────────────────────────────────────── */
static double score_mc_nc_kc_coarse(int mc, int nc, int kc,
    int pa, int pb, int pa2, int pb2)
{
    BenchTask t={mc,nc,kc,pa,pb,pa2,pb2,
                 COARSE_SIZE,COARSE_WARMUP,COARSE_REPEATS,0,-1.0};
    bench_thread(&t);
    return t.gops;
}

static double score_mc_nc_kc_fine(int mc, int nc, int kc,
    int pa, int pb, int pa2, int pb2)
{
    double inv=0.0;
    for(int s=0;s<N_TUNE_SIZES;s++){
        BenchTask t={mc,nc,kc,pa,pb,pa2,pb2,
                     TUNE_SIZES[s],FINE_WARMUP,FINE_REPEATS,0,-1.0};
        bench_thread(&t);
        if(t.gops<=0.0) return -1.0;
        inv+=1.0/t.gops;
    }
    return (double)N_TUNE_SIZES/inv;
}

/* ─────────────────────────────────────────────────────────────────
 * Candidate generators
 * ──────────────────────────────────────────────────────────────── */
static void gen_kc(int l1d_kb,int l2_kb,int*out,int*n,int maxn){
    int ideal=(l1d_kb*1024)/NR,max_kc=(l2_kb*1024)/NR;
    int pct[]={25,60,100,175,300};*n=0;
    for(int i=0;i<5;i++){
        int kc=round_down((int)((long long)ideal*pct[i]/100),MR);
        if(kc<MR)continue;if(kc>max_kc)kc=round_down(max_kc,MR);
        int dup=0;for(int j=0;j<*n;j++)if(out[j]==kc){dup=1;break;}
        if(!dup&&*n<maxn)out[(*n)++]=kc;
    }
}
static void gen_mc(int l2_kb,int l3_kb,int kc,int*out,int*n,int maxn){
    int ideal=(l2_kb*1024)/kc,max_mc=(l3_kb*1024)/kc/2;
    if(max_mc<ideal)max_mc=ideal*2;
    ideal=clamp_val(ideal,MR,max_mc);
    int pct[]={25,60,100,175};*n=0;
    for(int i=0;i<4;i++){
        int mc=round_down((int)((long long)ideal*pct[i]/100),MR);
        if(mc<MR)continue;if(mc>max_mc)mc=round_down(max_mc,MR);
        int dup=0;for(int j=0;j<*n;j++)if(out[j]==mc){dup=1;break;}
        if(!dup&&*n<maxn)out[(*n)++]=mc;
    }
}
static void gen_nc(int l3_kb,int kc,int*out,int*n,int maxn){
    int ideal=(l3_kb*1024)/kc,max_nc=ideal*2;
    int pct[]={25,60,100,175};*n=0;
    for(int i=0;i<4;i++){
        int nc=round_down((int)((long long)ideal*pct[i]/100),NR);
        if(nc<NR)continue;if(nc>max_nc)nc=round_down(max_nc,NR);
        int dup=0;for(int j=0;j<*n;j++)if(out[j]==nc){dup=1;break;}
        if(!dup&&*n<maxn)out[(*n)++]=nc;
    }
}
static void fine_neighbours(int center,int align,int lo,int hi,int*out,int*n){
    int step=round_down((int)((long long)center*125/1000),align);
    if(step<align)step=align;*n=0;
    for(int d=-2;d<=2;d++){
        int v=round_down(clamp_val(center+d*step,lo,hi),align);
        int dup=0;for(int j=0;j<*n;j++)if(out[j]==v){dup=1;break;}
        if(!dup&&*n<16)out[(*n)++]=v;
    }
}

/* ═══════════════════════════════════════════════════════════════ */
int main(void)
{
    int ncpus=(int)sysconf(_SC_NPROCESSORS_ONLN);
    NPAR=ncpus<4?ncpus:4;

    printf("==============================================\n");
    printf("  int8 GEMM Auto-Tuner (MC/NC/KC + Prefetch)\n");
    printf("  MR=%d NR=%d  CPUs=%d  NPAR=%d\n",MR,NR,ncpus,NPAR);
    printf("  Tune sizes:");
    for(int s=0;s<N_TUNE_SIZES;s++) printf(" %d",TUNE_SIZES[s]);
    printf("\n==============================================\n\n");

    int l1d_kb=read_cache_kb(1),l2_kb=read_cache_kb(2),l3_kb=read_cache_kb(3);
    if(l1d_kb<=0){puts("L1D→48KB");l1d_kb=48;}
    if(l2_kb<=0){puts("L2→512KB");l2_kb=512;}
    if(l3_kb<=0){puts("L3→8192KB");l3_kb=8192;}
    printf("Cache: L1D=%dKB L2=%dKB L3=%dKB\n\n",l1d_kb,l2_kb,l3_kb);

    /* default prefetch values for phases 1+2 */
    int def_pa1=192,def_pb1=512,def_pa2=768,def_pb2=2048;

    /* ════ PHASE 1: coarse MC/NC/KC ════ */
    int kc_c[16],mc_c[16],nc_c[16],n_kc,n_mc,n_nc;
    gen_kc(l1d_kb,l2_kb,kc_c,&n_kc,16);
    int total_c=0;
    for(int ki=0;ki<n_kc;ki++){
        gen_mc(l2_kb,l3_kb,kc_c[ki],mc_c,&n_mc,16);
        gen_nc(l3_kb,kc_c[ki],nc_c,&n_nc,16);
        total_c+=n_mc*n_nc;
    }

    printf("══ PHASE 1: %d combos at N=%d ══\n",total_c,COARSE_SIZE);
    printf("%-7s %-7s %-7s  GOPS\n","MC","NC","KC");
    printf("----------------------------------------\n");

    int c_best_mc=80,c_best_nc=1800,c_best_kc=4072;
    double c_best=-1.0;

    for(int ki=0;ki<n_kc;ki++){
        int kc=kc_c[ki];
        gen_mc(l2_kb,l3_kb,kc,mc_c,&n_mc,16);
        gen_nc(l3_kb,kc,nc_c,&n_nc,16);
        /* run NPAR combos in parallel */
        int total_mi_ni=n_mc*n_nc;
        BenchTask *tasks=malloc(total_mi_ni*sizeof(BenchTask));
        int ti=0;
        for(int mi=0;mi<n_mc;mi++)
            for(int ni=0;ni<n_nc;ni++){
                tasks[ti].mc=mc_c[mi];tasks[ti].nc=nc_c[ni];tasks[ti].kc=kc;
                tasks[ti].pa_l1=def_pa1;tasks[ti].pb_l1=def_pb1;
                tasks[ti].pa_l2=def_pa2;tasks[ti].pb_l2=def_pb2;
                tasks[ti].size=COARSE_SIZE;
                tasks[ti].warmup=COARSE_WARMUP;tasks[ti].repeats=COARSE_REPEATS;
                tasks[ti].gops=-1.0;ti++;
            }
        run_batch_full(tasks,total_mi_ni);
        for(int ti2=0;ti2<total_mi_ni;ti2++){
            BenchTask *t=&tasks[ti2];
            printf("%-7d %-7d %-7d  ",t->mc,t->nc,t->kc);
            if(t->gops<0) printf("FAIL\n");
            else printf("%.2f%s\n",t->gops,t->gops>c_best?" <- best":"");
            if(t->gops>c_best){
                c_best=t->gops;
                c_best_mc=t->mc;c_best_nc=t->nc;c_best_kc=t->kc;
            }
        }
        free(tasks);
    }
    printf("\nPhase 1 winner: MC=%d NC=%d KC=%d (%.2f)\n\n",
           c_best_mc,c_best_nc,c_best_kc,c_best);

    /* ════ PHASE 2: fine MC/NC/KC ════ */
    int kc_max=(l2_kb*1024)/NR;
    int mc_max=(l3_kb*1024)/c_best_kc/2;
    int nc_max=(l3_kb*1024)/c_best_kc*2;
    int kf[16],mf[16],nf[16],nkf,nmf,nnf;
    fine_neighbours(c_best_kc,MR,MR,kc_max,kf,&nkf);
    fine_neighbours(c_best_mc,MR,MR,mc_max,mf,&nmf);
    fine_neighbours(c_best_nc,NR,NR,nc_max,nf,&nnf);
    int total_f=nkf*nmf*nnf;

    printf("══ PHASE 2: %d fine combos ══\n",total_f);
    printf("%-7s %-7s %-7s  SCORE(hmean)\n","MC","NC","KC");
    printf("----------------------------------------\n");

    int best_mc=c_best_mc,best_nc=c_best_nc,best_kc=c_best_kc;
    double best=score_mc_nc_kc_fine(best_mc,best_nc,best_kc,
                                    def_pa1,def_pb1,def_pa2,def_pb2);
    printf("%-7d %-7d %-7d  %.2f [phase1 winner]\n",
           best_mc,best_nc,best_kc,best);

    BenchTask *ftasks=malloc(total_f*N_TUNE_SIZES*sizeof(BenchTask));
    int fi=0;
    for(int ki=0;ki<nkf;ki++)
        for(int mi=0;mi<nmf;mi++)
            for(int ni=0;ni<nnf;ni++){
                for(int s=0;s<N_TUNE_SIZES;s++){
                    ftasks[fi].mc=mf[mi];ftasks[fi].nc=nf[ni];ftasks[fi].kc=kf[ki];
                    ftasks[fi].pa_l1=def_pa1;ftasks[fi].pb_l1=def_pb1;
                    ftasks[fi].pa_l2=def_pa2;ftasks[fi].pb_l2=def_pb2;
                    ftasks[fi].size=TUNE_SIZES[s];
                    ftasks[fi].warmup=FINE_WARMUP;ftasks[fi].repeats=FINE_REPEATS;
                    ftasks[fi].gops=-1.0;fi++;
                }
            }
    run_batch_full(ftasks,fi);

    /* compute harmonic mean per combo */
    fi=0;
    for(int ki=0;ki<nkf;ki++)
        for(int mi=0;mi<nmf;mi++)
            for(int ni=0;ni<nnf;ni++){
                int mc=mf[mi],nc=nf[ni],kc=kf[ki];
                double inv=0.0;int ok=1;
                for(int s=0;s<N_TUNE_SIZES;s++){
                    if(ftasks[fi+s].gops<=0.0){ok=0;break;}
                    inv+=1.0/ftasks[fi+s].gops;
                }
                double score=ok?(double)N_TUNE_SIZES/inv:-1.0;
                printf("%-7d %-7d %-7d  ",mc,nc,kc);
                if(score<0) printf("FAIL\n");
                else printf("%.2f%s\n",score,score>best?" <- best":"");
                if(score>best){best=score;best_mc=mc;best_nc=nc;best_kc=kc;}
                fi+=N_TUNE_SIZES;
            }
    free(ftasks);
    printf("\nPhase 2 winner: MC=%d NC=%d KC=%d (%.2f)\n\n",
           best_mc,best_nc,best_kc,best);

    /* ════ PHASE 3: prefetch distance sweep ════ */
    int total_pf=N_PA*N_PB;
    printf("══ PHASE 3: %d prefetch combos (PA_L1×PB_L1) ══\n",total_pf);
    printf("%-7s %-7s  SCORE(hmean)\n","PA_L1","PB_L1");
    printf("----------------------------------------\n");

    int best_pa1=def_pa1,best_pb1=def_pb1;
    int best_pa2=def_pa2,best_pb2=def_pb2;
    double best_pf=-1.0;

    /* score default first */
    best_pf=score_mc_nc_kc_fine(best_mc,best_nc,best_kc,
                                 def_pa1,def_pb1,def_pa2,def_pb2);
    printf("%-7d %-7d  %.2f [default]\n",def_pa1,def_pb1,best_pf);

    BenchTask *ptasks=malloc(total_pf*N_TUNE_SIZES*sizeof(BenchTask));
    int pi=0;
    for(int ai=0;ai<N_PA;ai++)
        for(int bi=0;bi<N_PB;bi++){
            int pa1=PA_L1_VALS[ai],pb1=PB_L1_VALS[bi];
            int pa2=pa1*4,pb2=pb1*4;
            for(int s=0;s<N_TUNE_SIZES;s++){
                ptasks[pi].mc=best_mc;ptasks[pi].nc=best_nc;ptasks[pi].kc=best_kc;
                ptasks[pi].pa_l1=pa1;ptasks[pi].pb_l1=pb1;
                ptasks[pi].pa_l2=pa2;ptasks[pi].pb_l2=pb2;
                ptasks[pi].size=TUNE_SIZES[s];
                ptasks[pi].warmup=FINE_WARMUP;ptasks[pi].repeats=FINE_REPEATS;
                ptasks[pi].gops=-1.0;pi++;
            }
        }
    run_batch_full(ptasks,pi);

    pi=0;
    for(int ai=0;ai<N_PA;ai++)
        for(int bi=0;bi<N_PB;bi++){
            int pa1=PA_L1_VALS[ai],pb1=PB_L1_VALS[bi];
            int pa2=pa1*4,pb2=pb1*4;
            double inv=0.0;int ok=1;
            for(int s=0;s<N_TUNE_SIZES;s++){
                if(ptasks[pi+s].gops<=0.0){ok=0;break;}
                inv+=1.0/ptasks[pi+s].gops;
            }
            double score=ok?(double)N_TUNE_SIZES/inv:-1.0;
            printf("%-7d %-7d  ",pa1,pb1);
            if(score<0) printf("FAIL\n");
            else printf("%.2f%s\n",score,score>best_pf?" <- best":"");
            if(score>best_pf){
                best_pf=score;
                best_pa1=pa1;best_pb1=pb1;
                best_pa2=pa2;best_pb2=pb2;
            }
            pi+=N_TUNE_SIZES;
        }
    free(ptasks);

    /* ════ FINAL RESULT ════ */
    printf("\n==============================================\n");
    printf("  FINAL BEST\n");
    printf("==============================================\n");
    printf("  MC=%d  NC=%d  KC=%d\n",best_mc,best_nc,best_kc);
    printf("  PREFETCH_A_L1=%d  PREFETCH_B_L1=%d\n",best_pa1,best_pb1);
    printf("  PREFETCH_A_L2=%d  PREFETCH_B_L2=%d\n",best_pa2,best_pb2);
    printf("  Score = %.2f GOPS\n",best_pf);
    printf("==============================================\n");

    FILE *out=fopen("/tmp/autotune_result.txt","w");
    if(out){
        fprintf(out,"%d %d %d %d %d %d %d\n",
                best_mc,best_nc,best_kc,
                best_pa1,best_pb1,best_pa2,best_pb2);
        fclose(out);
        printf("  Written to /tmp/autotune_result.txt\n");
    }
    return 0;
}