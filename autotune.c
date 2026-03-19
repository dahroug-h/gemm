/*
 * autotune.c  —  Single-compile in-process auto-tuner (~2 min)
 *
 * REQUIRES one change to kernel.h:
 *   Replace:
 *     #define MC 80
 *     #define NC 1800
 *     #define KC 4072
 *   With:
 *     #ifndef MC
 *     extern int MC, NC, KC;
 *     #endif
 *
 * Then in this file we define MC/NC/KC as globals and change them
 * between runs — zero recompilation, all combos run in one process.
 *
 * Build:
 *   gcc autotune.c -O3 -march=core-avx2 -mavx2 -mno-avx512f \
 *       -fopenmp -o autotune -lm -lpthread
 *   ./autotune
 *
 * Time estimate: ~200 combos × ~0.5s avg = ~100s ≈ 2 min
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>

/* ── define MC/NC/KC as global ints before including kernel.h ── */
int MC = 80;
int NC = 1800;
int KC = 4072;

/* ── must match kernel.h register tile ── */
#define MR 8
#define NR 8

/* ── include kernel directly — it uses MC/NC/KC as variables now ── */
#undef min
#include "kernel.h"

/* ─────────────────────────────────────────────────────────────────
 * Timing
 * ──────────────────────────────────────────────────────────────── */
static double now_sec(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* ─────────────────────────────────────────────────────────────────
 * Run kernel() at size N, return GOPS. Returns -1 on error.
 * ──────────────────────────────────────────────────────────────── */
static double bench_size(int N, int warmup, int repeats)
{
    int8_t  *A = (int8_t* )malloc((size_t)N * N);
    int8_t  *B = (int8_t* )malloc((size_t)N * N);
    int32_t *C = (int32_t*)calloc((size_t)N * N, sizeof(int32_t));
    if (!A || !B || !C) { free(A); free(B); free(C); return -1.0; }

    for (int i = 0; i < N*N; i++) {
        A[i] = (int8_t)((i * 7 + 3) & 63);
        B[i] = (int8_t)((i * 5 + 1) & 63);
    }

    /* warmup */
    for (int r = 0; r < warmup; r++)
        kernel(N, N, N, A, N, B, N, C, N);
    memset(C, 0, (size_t)N * N * sizeof(int32_t));

    double t0 = now_sec();
    for (int r = 0; r < repeats; r++)
        kernel(N, N, N, A, N, B, N, C, N);
    double elapsed = now_sec() - t0;

    free(A); free(B); free(C);
    return 2.0 * (double)N * (double)N * (double)N * repeats / elapsed / 1e9;
}

/* ─────────────────────────────────────────────────────────────────
 * Score functions
 * ──────────────────────────────────────────────────────────────── */
#define COARSE_SIZE     1024
#define COARSE_REPEATS  2
#define COARSE_WARMUP   1

static const int FINE_SIZES[]  = { 512, 1024, 2048, 4000 };
static const int N_FINE_SIZES  = 4;
#define FINE_REPEATS  3
#define FINE_WARMUP   1

static double score_coarse(int mc, int nc, int kc)
{
    MC = mc; NC = nc; KC = kc;
    return bench_size(COARSE_SIZE, COARSE_WARMUP, COARSE_REPEATS);
}

static double score_fine(int mc, int nc, int kc)
{
    MC = mc; NC = nc; KC = kc;
    double inv = 0.0;
    for (int s = 0; s < N_FINE_SIZES; s++) {
        double g = bench_size(FINE_SIZES[s], FINE_WARMUP, FINE_REPEATS);
        if (g <= 0.0) return -1.0;
        inv += 1.0 / g;
    }
    return (double)N_FINE_SIZES / inv;
}

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
    printf("==============================================\n");
    printf("  int8 GEMM Auto-Tuner  (in-process, ~2 min)\n");
    printf("  MR=%d  NR=%d\n", MR, NR);
    printf("==============================================\n\n");

    int l1d_kb = read_cache_kb(1);
    int l2_kb  = read_cache_kb(2);
    int l3_kb  = read_cache_kb(3);
    if (l1d_kb <= 0) { puts("L1D → 48 KB");   l1d_kb = 48;   }
    if (l2_kb  <= 0) { puts("L2  → 512 KB");  l2_kb  = 512;  }
    if (l3_kb  <= 0) { puts("L3  → 8192 KB"); l3_kb  = 8192; }
    printf("Cache: L1D=%dKB  L2=%dKB  L3=%dKB\n\n",
           l1d_kb, l2_kb, l3_kb);

    /* smoke test */
    printf("Smoke test (MC=80 NC=1800 KC=4072)... ");
    fflush(stdout);
    double smoke = score_fine(80, 1800, 4072);
    if (smoke < 0) { printf("FAILED\n"); return 1; }
    printf("%.2f GOPS — OK\n\n", smoke);

    /* ════════════════════════════════════════════
     * PHASE 1 — coarse, single size
     * ════════════════════════════════════════════ */
    int kc_c[16], mc_c[16], nc_c[16];
    int n_kc, n_mc, n_nc;

    gen_kc(l1d_kb, l2_kb, kc_c, &n_kc, 16);

    int total_c = 0;
    for (int ki = 0; ki < n_kc; ki++) {
        gen_mc(l2_kb, l3_kb, kc_c[ki], mc_c, &n_mc, 16);
        gen_nc(l3_kb,        kc_c[ki], nc_c, &n_nc, 16);
        total_c += n_mc * n_nc;
    }

    printf("══ PHASE 1: %d combos at N=%d ══\n", total_c, COARSE_SIZE);
    printf("%-7s %-7s %-7s  GOPS\n", "MC","NC","KC");
    printf("----------------------------------------\n");

    int    c_best_mc=80, c_best_nc=1800, c_best_kc=4072;
    double c_best = -1.0;
    int    done = 0;

    for (int ki = 0; ki < n_kc; ki++) {
        int kc = kc_c[ki];
        gen_mc(l2_kb, l3_kb, kc, mc_c, &n_mc, 16);
        gen_nc(l3_kb,        kc, nc_c, &n_nc, 16);
        for (int mi = 0; mi < n_mc; mi++) {
            for (int ni = 0; ni < n_nc; ni++) {
                int mc = mc_c[mi], nc = nc_c[ni];
                double g = score_coarse(mc, nc, kc);
                done++;
                if (g < 0)
                    printf("%-7d %-7d %-7d  FAIL [%d/%d]\n",
                           mc, nc, kc, done, total_c);
                else
                    printf("%-7d %-7d %-7d  %.2f [%d/%d]%s\n",
                           mc, nc, kc, g, done, total_c,
                           g > c_best ? " <- best" : "");
                fflush(stdout);
                if (g > c_best) {
                    c_best = g;
                    c_best_mc=mc; c_best_nc=nc; c_best_kc=kc;
                }
            }
        }
    }

    printf("\nPhase 1 winner: MC=%d NC=%d KC=%d (%.2f GOPS)\n\n",
           c_best_mc, c_best_nc, c_best_kc, c_best);

    /* ════════════════════════════════════════════
     * PHASE 2 — fine grid around winner
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
    printf("══ PHASE 2: %d fine combos (harmonic mean:", total_f);
    for (int s=0;s<N_FINE_SIZES;s++) printf(" %d",FINE_SIZES[s]);
    printf(") ══\n");
    printf("%-7s %-7s %-7s  SCORE\n","MC","NC","KC");
    printf("----------------------------------------\n");

    int    best_mc=c_best_mc, best_nc=c_best_nc, best_kc=c_best_kc;
    double best = score_fine(best_mc, best_nc, best_kc);
    printf("%-7d %-7d %-7d  %.2f [baseline]\n",
           best_mc, best_nc, best_kc, best);
    done = 0;

    for (int ki = 0; ki < n_kc_f; ki++) {
        for (int mi = 0; mi < n_mc_f; mi++) {
            for (int ni = 0; ni < n_nc_f; ni++) {
                int kc=kc_f[ki], mc=mc_f[mi], nc=nc_f[ni];
                if (mc==best_mc && nc==best_nc && kc==best_kc) continue;
                double g = score_fine(mc, nc, kc);
                done++;
                if (g < 0)
                    printf("%-7d %-7d %-7d  FAIL [%d/%d]\n",
                           mc, nc, kc, done, total_f);
                else
                    printf("%-7d %-7d %-7d  %.2f [%d/%d]%s\n",
                           mc, nc, kc, g, done, total_f,
                           g > best ? " <- best" : "");
                fflush(stdout);
                if (g > best) {
                    best=g;
                    best_mc=mc; best_nc=nc; best_kc=kc;
                }
            }
        }
    }

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