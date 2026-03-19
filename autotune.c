/*
 * autotune.c  —  Parallel in-process auto-tuner (~1 min)
 *
 * SPEED TRICK:
 *   Each combo runs kernel() with 1 OMP thread pinned to one core.
 *   We run NCORES combos in parallel using pthreads — each thread
 *   owns one core, measures one combo, reports back.
 *   No timing interference between threads since they work on
 *   separate data and separate cores.
 *
 * REQUIRES kernel.h change:
 *   Replace:
 *     #define MC 80
 *     #define NC 1800
 *     #define KC 4072
 *     static int8_t Buffer_A[...]  <- remove this line
 *     static int8_t Buffer_B[...]  <- remove this line
 *   With:
 *     #ifndef MC
 *     extern int MC, NC, KC;
 *     #endif
 *
 * Build:
 *   gcc autotune.c -O3 -march=core-avx2 -mavx2 -mno-avx512f \
 *       -fopenmp -o autotune -lm -lpthread
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

/* ── define MC/NC/KC as thread-local so each thread has its own ── */
__thread int MC = 80;
__thread int NC = 1800;
__thread int KC = 4072;

/* ── must match kernel.h register tile ── */
#define MR 6
#define NR 16

/* ── include kernel — uses thread-local MC/NC/KC ── */
#undef min
#include "kernel.h"

/* ─────────────────────────────────────────────────────────────────
 * Config
 * ──────────────────────────────────────────────────────────────── */
#define COARSE_SIZE     400
#define COARSE_REPEATS  2
#define COARSE_WARMUP   1

static const int TUNE_SIZES[] = { 100, 200, 300, 400, 500, 600, 700, 800 };
static const int N_TUNE_SIZES = 8;
#define FINE_REPEATS  3
#define FINE_WARMUP   1

/* number of combos to run in parallel = logical CPUs, capped at 8 */
static int NPAR = 4;

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

static int round_down(int x, int m) { return (x / m) * m; }
static int clamp_val(int x, int lo, int hi){ return x<lo?lo:x>hi?hi:x; }

/* ─────────────────────────────────────────────────────────────────
 * Per-thread benchmark state
 * ──────────────────────────────────────────────────────────────── */
typedef struct {
    /* input */
    int mc, nc, kc;
    int size;
    int warmup;
    int repeats;
    int cpu_id;     /* pin to this CPU */
    /* output */
    double gops;    /* -1 on failure */
} BenchTask;

static void *bench_thread(void *arg)
{
    BenchTask *t = (BenchTask*)arg;

    /* pin thread to cpu_id */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(t->cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    /* set this thread's MC/NC/KC */
    MC = t->mc;
    NC = t->nc;
    KC = t->kc;

    int N = t->size;
    int8_t  *A = (int8_t* )malloc((size_t)N * N);
    int8_t  *B = (int8_t* )malloc((size_t)N * N);
    int32_t *C = (int32_t*)calloc((size_t)N * N, sizeof(int32_t));

    if (!A || !B || !C) {
        free(A); free(B); free(C);
        t->gops = -1.0;
        return NULL;
    }

    for (int i = 0; i < N*N; i++) {
        A[i] = (int8_t)((i * 7 + 3) & 63);
        B[i] = (int8_t)((i * 5 + 1) & 63);
    }

    /* warmup */
    for (int r = 0; r < t->warmup; r++)
        kernel(N, N, N, A, N, B, N, C, N);
    memset(C, 0, (size_t)N * N * sizeof(int32_t));

    /* timed */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < t->repeats; r++)
        kernel(N, N, N, A, N, B, N, C, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double sec = (t1.tv_sec - t0.tv_sec)
               + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    t->gops = 2.0 * (double)N*(double)N*(double)N * t->repeats / sec / 1e9;

    free(A); free(B); free(C);
    return NULL;
}

/* ─────────────────────────────────────────────────────────────────
 * Run a batch of combos in parallel.
 * combos[i] = {mc, nc, kc}, results[i] = GOPS
 * All combos benchmarked at the same size/warmup/repeats.
 * ──────────────────────────────────────────────────────────────── */
static void run_batch(int (*combos)[3], double *results, int n,
                      int size, int warmup, int repeats)
{
    int i = 0;
    while (i < n) {
        int batch = (n - i) < NPAR ? (n - i) : NPAR;

        pthread_t   threads[8];
        BenchTask   tasks[8];

        for (int b = 0; b < batch; b++) {
            tasks[b].mc      = combos[i+b][0];
            tasks[b].nc      = combos[i+b][1];
            tasks[b].kc      = combos[i+b][2];
            tasks[b].size    = size;
            tasks[b].warmup  = warmup;
            tasks[b].repeats = repeats;
            tasks[b].cpu_id  = b % NPAR;
            tasks[b].gops    = -1.0;
            pthread_create(&threads[b], NULL, bench_thread, &tasks[b]);
        }

        for (int b = 0; b < batch; b++) {
            pthread_join(threads[b], NULL);
            results[i+b] = tasks[b].gops;
        }

        i += batch;
    }
}

/* ─────────────────────────────────────────────────────────────────
 * Score fine: harmonic mean across TUNE_SIZES (sequential per size,
 * but combos within each size run in parallel)
 * ──────────────────────────────────────────────────────────────── */
static void score_fine_batch(int (*combos)[3], double *scores, int n)
{
    double *inv = (double*)calloc(n, sizeof(double));
    double *tmp = (double*)malloc(n * sizeof(double));

    for (int s = 0; s < N_TUNE_SIZES; s++) {
        run_batch(combos, tmp, n, TUNE_SIZES[s], FINE_WARMUP, FINE_REPEATS);
        for (int i = 0; i < n; i++) {
            if (tmp[i] <= 0.0) inv[i] = 1e18;  /* mark as bad */
            else               inv[i] += 1.0 / tmp[i];
        }
    }

    for (int i = 0; i < n; i++) {
        if (inv[i] >= 1e17) scores[i] = -1.0;
        else                scores[i] = (double)N_TUNE_SIZES / inv[i];
    }

    free(inv); free(tmp);
}

/* ─────────────────────────────────────────────────────────────────
 * Candidate generators
 * ──────────────────────────────────────────────────────────────── */
static void gen_kc(int l1d_kb, int l2_kb, int *out, int *n, int maxn)
{
    int ideal  = (l1d_kb * 1024) / NR;
    int max_kc = (l2_kb  * 1024) / NR;
    int pct[]  = { 25, 60, 100, 175, 300 };
    *n = 0;
    for (int i = 0; i < 5; i++) {
        int kc = round_down((int)((long long)ideal * pct[i] / 100), MR);
        if (kc < MR) continue;
        if (kc > max_kc) kc = round_down(max_kc, MR);
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j]==kc){dup=1;break;}
        if (!dup && *n < maxn) out[(*n)++] = kc;
    }
}

static void gen_mc(int l2_kb, int l3_kb, int kc,
                   int *out, int *n, int maxn)
{
    int ideal  = (l2_kb * 1024) / kc;
    int max_mc = (l3_kb * 1024) / kc / 2;
    if (max_mc < ideal) max_mc = ideal * 2;
    ideal = clamp_val(ideal, MR, max_mc);
    int pct[] = { 25, 60, 100, 175 };
    *n = 0;
    for (int i = 0; i < 4; i++) {
        int mc = round_down((int)((long long)ideal * pct[i] / 100), MR);
        if (mc < MR) continue;
        if (mc > max_mc) mc = round_down(max_mc, MR);
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j]==mc){dup=1;break;}
        if (!dup && *n < maxn) out[(*n)++] = mc;
    }
}

static void gen_nc(int l3_kb, int kc, int *out, int *n, int maxn)
{
    int ideal  = (l3_kb * 1024) / kc;
    int max_nc = ideal * 2;
    int pct[]  = { 25, 60, 100, 175 };
    *n = 0;
    for (int i = 0; i < 4; i++) {
        int nc = round_down((int)((long long)ideal * pct[i] / 100), NR);
        if (nc < NR) continue;
        if (nc > max_nc) nc = round_down(max_nc, NR);
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j]==nc){dup=1;break;}
        if (!dup && *n < maxn) out[(*n)++] = nc;
    }
}

static void fine_neighbours(int center, int align,
                            int lo, int hi, int *out, int *n)
{
    int step = round_down((int)((long long)center * 125 / 1000), align);
    if (step < align) step = align;
    *n = 0;
    for (int d = -2; d <= 2; d++) {
        int v = round_down(clamp_val(center + d*step, lo, hi), align);
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j]==v){dup=1;break;}
        if (!dup && *n < 16) out[(*n)++] = v;
    }
}

/* ═══════════════════════════════════════════════════════════════ */
int main(void)
{
    /* detect parallelism */
    int ncpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    NPAR = ncpus < 8 ? ncpus : 8;

    printf("==============================================\n");
    printf("  int8 GEMM Auto-Tuner  (parallel in-process)\n");
    printf("  MR=%d  NR=%d\n", MR, NR);
    printf("  CPUs: %d  Parallel combos: %d\n", ncpus, NPAR);
    printf("==============================================\n\n");

    int l1d_kb = read_cache_kb(1);
    int l2_kb  = read_cache_kb(2);
    int l3_kb  = read_cache_kb(3);
    if (l1d_kb <= 0) { puts("L1D → 48 KB");   l1d_kb = 48;   }
    if (l2_kb  <= 0) { puts("L2  → 512 KB");  l2_kb  = 512;  }
    if (l3_kb  <= 0) { puts("L3  → 8192 KB"); l3_kb  = 8192; }
    printf("Cache: L1D=%dKB  L2=%dKB  L3=%dKB\n\n",
           l1d_kb, l2_kb, l3_kb);

    /* ── build coarse combo list ── */
    int kc_c[16], mc_c[16], nc_c[16];
    int n_kc, n_mc, n_nc;
    gen_kc(l1d_kb, l2_kb, kc_c, &n_kc, 16);

    /* count */
    int total_c = 0;
    for (int ki = 0; ki < n_kc; ki++) {
        gen_mc(l2_kb, l3_kb, kc_c[ki], mc_c, &n_mc, 16);
        gen_nc(l3_kb,        kc_c[ki], nc_c, &n_nc, 16);
        total_c += n_mc * n_nc;
    }

    /* build flat array */
    int (*coarse_combos)[3] = malloc(total_c * sizeof(*coarse_combos));
    double *coarse_scores   = malloc(total_c * sizeof(double));
    int ci = 0;
    for (int ki = 0; ki < n_kc; ki++) {
        int kc = kc_c[ki];
        gen_mc(l2_kb, l3_kb, kc, mc_c, &n_mc, 16);
        gen_nc(l3_kb,        kc, nc_c, &n_nc, 16);
        for (int mi = 0; mi < n_mc; mi++)
            for (int ni = 0; ni < n_nc; ni++) {
                coarse_combos[ci][0] = mc_c[mi];
                coarse_combos[ci][1] = nc_c[ni];
                coarse_combos[ci][2] = kc;
                ci++;
            }
    }

    /* ════════════════════════════════════════════
     * PHASE 1 — all coarse combos in parallel
     * ════════════════════════════════════════════ */
    printf("══ PHASE 1: %d combos at N=%d (parallel=%d) ══\n",
           total_c, COARSE_SIZE, NPAR);

    run_batch(coarse_combos, coarse_scores, total_c,
              COARSE_SIZE, COARSE_WARMUP, COARSE_REPEATS);

    /* find winner */
    int    c_best_mc=80, c_best_nc=1800, c_best_kc=4072;
    double c_best = -1.0;
    printf("%-7s %-7s %-7s  GOPS\n", "MC","NC","KC");
    printf("----------------------------------------\n");
    for (int i = 0; i < total_c; i++) {
        printf("%-7d %-7d %-7d  ",
               coarse_combos[i][0],
               coarse_combos[i][1],
               coarse_combos[i][2]);
        if (coarse_scores[i] < 0) printf("FAIL\n");
        else {
            printf("%.2f%s\n", coarse_scores[i],
                   coarse_scores[i] > c_best ? " <- best" : "");
        }
        if (coarse_scores[i] > c_best) {
            c_best      = coarse_scores[i];
            c_best_mc   = coarse_combos[i][0];
            c_best_nc   = coarse_combos[i][1];
            c_best_kc   = coarse_combos[i][2];
        }
    }

    printf("\nPhase 1 winner: MC=%d NC=%d KC=%d (%.2f GOPS)\n\n",
           c_best_mc, c_best_nc, c_best_kc, c_best);
    free(coarse_combos); free(coarse_scores);

    /* ════════════════════════════════════════════
     * PHASE 2 — fine combos in parallel
     * ════════════════════════════════════════════ */
    int kc_max = (l2_kb * 1024) / NR;
    int mc_max = (l3_kb * 1024) / c_best_kc / 2;
    int nc_max = (l3_kb * 1024) / c_best_kc * 2;

    int kc_f[16], mc_f[16], nc_f[16];
    int n_kc_f, n_mc_f, n_nc_f;
    fine_neighbours(c_best_kc, MR, MR, kc_max, kc_f, &n_kc_f);
    fine_neighbours(c_best_mc, MR, MR, mc_max, mc_f, &n_mc_f);
    fine_neighbours(c_best_nc, NR, NR, nc_max, nc_f, &n_nc_f);

    int total_f = n_kc_f * n_mc_f * n_nc_f;
    int (*fine_combos)[3] = malloc(total_f * sizeof(*fine_combos));
    double *fine_scores   = malloc(total_f * sizeof(double));
    int fi = 0;
    for (int ki = 0; ki < n_kc_f; ki++)
        for (int mi = 0; mi < n_mc_f; mi++)
            for (int ni = 0; ni < n_nc_f; ni++) {
                fine_combos[fi][0] = mc_f[mi];
                fine_combos[fi][1] = nc_f[ni];
                fine_combos[fi][2] = kc_f[ki];
                fi++;
            }

    printf("══ PHASE 2: %d fine combos (parallel=%d, harmonic mean:",
           total_f, NPAR);
    for (int s=0;s<N_TUNE_SIZES;s++) printf(" %d",TUNE_SIZES[s]);
    printf(") ══\n");

    score_fine_batch(fine_combos, fine_scores, total_f);

    int    best_mc=c_best_mc, best_nc=c_best_nc, best_kc=c_best_kc;
    double best = -1.0;

    /* include phase1 winner in fine scoring */
    {
        int tmp_combo[1][3] = {{c_best_mc, c_best_nc, c_best_kc}};
        double tmp_score[1];
        score_fine_batch(tmp_combo, tmp_score, 1);
        best = tmp_score[0];
    }

    printf("%-7s %-7s %-7s  SCORE\n","MC","NC","KC");
    printf("----------------------------------------\n");
    printf("%-7d %-7d %-7d  %.2f [phase1 winner]\n",
           best_mc, best_nc, best_kc, best);

    for (int i = 0; i < total_f; i++) {
        printf("%-7d %-7d %-7d  ",
               fine_combos[i][0],
               fine_combos[i][1],
               fine_combos[i][2]);
        if (fine_scores[i] < 0) printf("FAIL\n");
        else {
            printf("%.2f%s\n", fine_scores[i],
                   fine_scores[i] > best ? " <- best" : "");
        }
        if (fine_scores[i] > best) {
            best    = fine_scores[i];
            best_mc = fine_combos[i][0];
            best_nc = fine_combos[i][1];
            best_kc = fine_combos[i][2];
        }
    }

    free(fine_combos); free(fine_scores);

    printf("\n==============================================\n");
    printf("  FINAL BEST\n");
    printf("==============================================\n");
    printf("  #define MC  %d\n", best_mc);
    printf("  #define NC  %d\n", best_nc);
    printf("  #define KC  %d\n", best_kc);
    printf("  Score = %.2f GOPS\n", best);
    printf("==============================================\n");

    FILE *out = fopen("/tmp/autotune_result.txt", "w");
    if (out) {
        fprintf(out, "%d %d %d\n", best_mc, best_nc, best_kc);
        fclose(out);
        printf("  Written to /tmp/autotune_result.txt\n");
    }
    return 0;
}