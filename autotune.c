/*
 * autotune.c  —  Fast two-phase auto-tuner for int8→int32 GEMM
 *
 * Phase 1 COARSE: ~5 KC × 4 MC × 4 NC = ~80 combos, single mid size
 * Phase 2 FINE:   ±2 steps around winner, full 6-size harmonic mean
 * Total wall time: ~3-5 min on GitHub Actions
 *
 * Build:
 *   gcc autotune.c -O2 -o autotune -lm
 *   ./autotune
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* ── must match kernel.h ── */
#define MR  8
#define NR  8

/* ── phase 1: single representative size, fewer reps ── */
#define COARSE_SIZE     1024
#define COARSE_REPEATS  3
#define COARSE_WARMUP   1

/* ── phase 2: full sweep, more reps ── */
static const int FINE_SIZES[]   = { 256, 512, 1024, 2048, 3072, 4000 };
static const int N_FINE_SIZES   = 6;
#define FINE_REPEATS    6
#define FINE_WARMUP     2

/* ── phase 1 grid: only 5 KC × 4 MC × 4 NC = 80 combos ── */
#define N_COARSE_KC  5
#define N_COARSE_MC  4
#define N_COARSE_NC  4

#define TMP_DRIVER  "/tmp/at_driver.c"
#define TMP_BIN     "/tmp/at_run"

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
static int clamp(int x, int lo, int hi) { return x < lo ? lo : x > hi ? hi : x; }

/* ─────────────────────────────────────────────────────────────────
 * PHASE 1 — coarse grid
 * Pick 5 KC values log-spaced between 25% and 300% of ideal.
 * For each KC pick 4 MC and 4 NC values (25/50/100/150% of ideal).
 * Total: 5 × 4 × 4 = 80 combos, each benchmarked at ONE size.
 * ──────────────────────────────────────────────────────────────── */
static void coarse_kc(int l1d_kb, int l2_kb, int *out, int *n)
{
    int ideal  = (l1d_kb * 1024) / NR;
    int max_kc = (l2_kb  * 1024) / NR;
    /* 5 points: 25% 60% 100% 175% 300% */
    int pct[] = { 25, 60, 100, 175, 300 };
    *n = 0;
    for (int i = 0; i < N_COARSE_KC; i++) {
        int kc = round_down((int)((long long)ideal * pct[i] / 100), MR);
        if (kc < MR)     continue;
        if (kc > max_kc) kc = round_down(max_kc, MR);
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j] == kc) { dup=1; break; }
        if (!dup && *n < 16) out[(*n)++] = kc;
    }
}

static void coarse_mc(int l2_kb, int l3_kb, int kc, int *out, int *n)
{
    int ideal  = (l2_kb * 1024) / kc;
    int max_mc = (l3_kb * 1024) / kc / 2;
    if (max_mc < ideal) max_mc = ideal * 2;
    ideal  = clamp(ideal,  MR, max_mc);
    /* 4 points: 25% 60% 100% 175% */
    int pct[] = { 25, 60, 100, 175 };
    *n = 0;
    for (int i = 0; i < N_COARSE_MC; i++) {
        int mc = round_down((int)((long long)ideal * pct[i] / 100), MR);
        if (mc < MR)     continue;
        if (mc > max_mc) mc = round_down(max_mc, MR);
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j] == mc) { dup=1; break; }
        if (!dup && *n < 16) out[(*n)++] = mc;
    }
}

static void coarse_nc(int l3_kb, int kc, int *out, int *n)
{
    int ideal  = (l3_kb * 1024) / kc;
    int max_nc = ideal * 2;
    /* 4 points: 25% 60% 100% 175% */
    int pct[] = { 25, 60, 100, 175 };
    *n = 0;
    for (int i = 0; i < N_COARSE_NC; i++) {
        int nc = round_down((int)((long long)ideal * pct[i] / 100), NR);
        if (nc < NR)     continue;
        if (nc > max_nc) nc = round_down(max_nc, NR);
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j] == nc) { dup=1; break; }
        if (!dup && *n < 16) out[(*n)++] = nc;
    }
}

/* ─────────────────────────────────────────────────────────────────
 * PHASE 2 — fine grid around the coarse winner
 * For each dimension generate ±2 neighbours with step = 12.5% of
 * the winner value, aligned to MR/NR. ~5×5×5 = 125 combos max,
 * each scored with harmonic mean across all 6 FINE_SIZES.
 * ──────────────────────────────────────────────────────────────── */
static void fine_neighbours(int center, int align,
                            int lo_bound, int hi_bound,
                            int *out, int *n)
{
    /* step = 12.5% of center, at least one alignment unit */
    int step = round_down((int)((long long)center * 125 / 1000), align);
    if (step < align) step = align;

    *n = 0;
    for (int delta = -2; delta <= 2; delta++) {
        int v = round_down(center + delta * step, align);
        v = clamp(v, lo_bound, hi_bound);
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j] == v) { dup=1; break; }
        if (!dup && *n < 16) out[(*n)++] = v;
    }
}

/* ─────────────────────────────────────────────────────────────────
 * Driver writer — N passed as argv[1]
 * ──────────────────────────────────────────────────────────────── */
static void write_driver(const char *kernel_h_path)
{
    FILE *f = fopen(TMP_DRIVER, "w");
    if (!f) { perror("cannot write " TMP_DRIVER); exit(1); }
    fprintf(f,
"#include <stdlib.h>\n"
"#include <stdio.h>\n"
"#include <string.h>\n"
"#include <time.h>\n"
"#include <stdint.h>\n"
"#include <immintrin.h>\n"
"#undef MC\n#undef NC\n#undef KC\n#undef min\n"
"#include \"%s\"\n"
"int main(int argc, char **argv) {\n"
"    int N = (argc > 1) ? atoi(argv[1]) : 512;\n"
"    int warmup  = (argc > 2) ? atoi(argv[2]) : 2;\n"
"    int repeats = (argc > 3) ? atoi(argv[3]) : 6;\n"
"    int8_t  *A = (int8_t* )malloc((size_t)N*N);\n"
"    int8_t  *B = (int8_t* )malloc((size_t)N*N);\n"
"    int32_t *C = (int32_t*)calloc((size_t)N*N, sizeof(int32_t));\n"
"    if (!A||!B||!C){fputs(\"OOM\\n\",stderr);return 1;}\n"
"    for(int i=0;i<N*N;i++){\n"
"        A[i]=(int8_t)((i*7+3)&63); B[i]=(int8_t)((i*5+1)&63);}\n"
"    for(int r=0;r<warmup;r++)\n"
"        kernel(N,N,N,A,N,B,N,C,N);\n"
"    memset(C,0,(size_t)N*N*sizeof(int32_t));\n"
"    struct timespec t0,t1;\n"
"    clock_gettime(CLOCK_MONOTONIC,&t0);\n"
"    for(int r=0;r<repeats;r++)\n"
"        kernel(N,N,N,A,N,B,N,C,N);\n"
"    clock_gettime(CLOCK_MONOTONIC,&t1);\n"
"    double sec=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;\n"
"    double gops=2.0*(double)N*(double)N*(double)N*repeats/sec/1e9;\n"
"    printf(\"%%.3f\\n\",gops);\n"
"    free(A);free(B);free(C);\n"
"    return 0;\n}\n",
        kernel_h_path);
    fclose(f);
}

/* ─────────────────────────────────────────────────────────────────
 * Compile + run helpers
 * ──────────────────────────────────────────────────────────────── */
static int compile_combo(int mc, int nc, int kc)
{
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "gcc -O3 -march=core-avx2 -mavx2 -mno-avx512f "
        "-DMC=%d -DNC=%d -DKC=%d "
        "%s -o %s -lm -lpthread 2>/tmp/at_err.txt",
        mc, nc, kc, TMP_DRIVER, TMP_BIN);
    return system(cmd);
}

static double run_once(int N, int warmup, int repeats)
{
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "%s %d %d %d", TMP_BIN, N, warmup, repeats);
    FILE *p = popen(cmd, "r");
    if (!p) return -1.0;
    double g = -1.0;
    if (fscanf(p, "%lf", &g) != 1) g = -1.0;
    pclose(p);
    return g;
}

/* single-size score (coarse phase) */
static double bench_coarse(int mc, int nc, int kc)
{
    if (compile_combo(mc, nc, kc) != 0) return -1.0;
    return run_once(COARSE_SIZE, COARSE_WARMUP, COARSE_REPEATS);
}

/* harmonic-mean score across all fine sizes */
static double bench_fine(int mc, int nc, int kc)
{
    if (compile_combo(mc, nc, kc) != 0) return -1.0;
    double inv = 0.0;
    for (int s = 0; s < N_FINE_SIZES; s++) {
        double g = run_once(FINE_SIZES[s], FINE_WARMUP, FINE_REPEATS);
        if (g <= 0.0) return -1.0;
        inv += 1.0 / g;
    }
    return (double)N_FINE_SIZES / inv;
}

/* ═══════════════════════════════════════════════════════════════ */
int main(void)
{
    char cwd[512], kpath[600];
    if (!getcwd(cwd, sizeof(cwd))) { perror("getcwd"); return 1; }
    snprintf(kpath, sizeof(kpath), "%s/kernel.h", cwd);

    printf("==============================================\n");
    printf("  int8 GEMM Auto-Tuner  (two-phase search)\n");
    printf("  MR=%d  NR=%d\n", MR, NR);
    printf("  Flags: -march=core-avx2 -mavx2 -mno-avx512f\n");
    printf("  kernel.h: %s\n", kpath);
    printf("==============================================\n\n");

    int l1d_kb = read_cache_kb(1);
    int l2_kb  = read_cache_kb(2);
    int l3_kb  = read_cache_kb(3);
    if (l1d_kb <= 0) { puts("L1D unreadable → 48 KB");   l1d_kb = 48;   }
    if (l2_kb  <= 0) { puts("L2  unreadable → 512 KB");  l2_kb  = 512;  }
    if (l3_kb  <= 0) { puts("L3  unreadable → 8192 KB"); l3_kb  = 8192; }

    printf("Cache: L1D=%dKB  L2=%dKB  L3=%dKB\n\n", l1d_kb, l2_kb, l3_kb);

    write_driver(kpath);

    /* ── smoke test ── */
    printf("Smoke test (MC=80 NC=1800 KC=4072)... ");
    fflush(stdout);
    double smoke = bench_fine(80, 1800, 4072);
    if (smoke < 0) {
        printf("FAILED\n");
        system("cat /tmp/at_err.txt");
        return 1;
    }
    printf("%.2f GOPS — OK\n\n", smoke);

    /* ════════════════════════════════════════════
     * PHASE 1 — coarse grid, single size N=1024
     * ════════════════════════════════════════════ */
    int kc_c[16], mc_c[16], nc_c[16];
    int n_kc, n_mc, n_nc;

    coarse_kc(l1d_kb, l2_kb, kc_c, &n_kc);

    int total_coarse = 0;
    for (int ki = 0; ki < n_kc; ki++) {
        coarse_mc(l2_kb, l3_kb, kc_c[ki], mc_c, &n_mc);
        coarse_nc(l3_kb, kc_c[ki], nc_c, &n_nc);
        total_coarse += n_mc * n_nc;
    }

    printf("══ PHASE 1: coarse grid  (N=%d, %d combos) ══\n",
           COARSE_SIZE, total_coarse);
    printf("%-7s %-7s %-7s  GOPS@%d\n", "MC","NC","KC", COARSE_SIZE);
    printf("----------------------------------------------\n");

    int    c_best_mc = 80, c_best_nc = 1800, c_best_kc = 4072;
    double c_best    = -1.0;
    int    done      = 0;

    for (int ki = 0; ki < n_kc; ki++) {
        int kc = kc_c[ki];
        coarse_mc(l2_kb, l3_kb, kc, mc_c, &n_mc);
        coarse_nc(l3_kb, kc, nc_c, &n_nc);
        for (int mi = 0; mi < n_mc; mi++) {
            for (int ni = 0; ni < n_nc; ni++) {
                int mc = mc_c[mi], nc = nc_c[ni];
                double g = bench_coarse(mc, nc, kc);
                done++;
                if (g < 0)
                    printf("%-7d %-7d %-7d  FAIL  [%d/%d]\n",
                           mc, nc, kc, done, total_coarse);
                else
                    printf("%-7d %-7d %-7d  %.2f  [%d/%d]%s\n",
                           mc, nc, kc, g, done, total_coarse,
                           g > c_best ? "  <- best" : "");
                fflush(stdout);
                if (g > c_best) {
                    c_best = g;
                    c_best_mc = mc; c_best_nc = nc; c_best_kc = kc;
                }
            }
        }
    }

    printf("\nPhase 1 winner: MC=%d NC=%d KC=%d  (%.2f GOPS@%d)\n\n",
           c_best_mc, c_best_nc, c_best_kc, c_best, COARSE_SIZE);

    /* ════════════════════════════════════════════
     * PHASE 2 — fine grid around winner
     * ════════════════════════════════════════════ */
    int kc_max = (l2_kb * 1024) / NR;
    int mc_max = (l3_kb * 1024) / c_best_kc / 2;
    int nc_max = (l3_kb * 1024) / c_best_kc * 2;

    int kc_f[16], mc_f[16], nc_f[16];
    int n_kc_f, n_mc_f, n_nc_f;

    fine_neighbours(c_best_kc, MR, MR,    kc_max, kc_f, &n_kc_f);
    fine_neighbours(c_best_mc, MR, MR,    mc_max, mc_f, &n_mc_f);
    fine_neighbours(c_best_nc, NR, NR,    nc_max, nc_f, &n_nc_f);

    int total_fine = n_kc_f * n_mc_f * n_nc_f;

    printf("══ PHASE 2: fine grid around winner (%d combos, harmonic mean) ══\n",
           total_fine);
    printf("   Fine sizes:");
    for (int s = 0; s < N_FINE_SIZES; s++) printf(" %d", FINE_SIZES[s]);
    printf("\n");
    printf("%-7s %-7s %-7s  SCORE(hmean)\n", "MC","NC","KC");
    printf("----------------------------------------------\n");

    int    best_mc = c_best_mc, best_nc = c_best_nc, best_kc = c_best_kc;
    double best    = bench_fine(best_mc, best_nc, best_kc);
    printf("%-7d %-7d %-7d  %.2f  [baseline]\n",
           best_mc, best_nc, best_kc, best);
    done = 0;

    for (int ki = 0; ki < n_kc_f; ki++) {
        for (int mi = 0; mi < n_mc_f; mi++) {
            for (int ni = 0; ni < n_nc_f; ni++) {
                int kc = kc_f[ki], mc = mc_f[mi], nc = nc_f[ni];
                /* skip baseline — already scored */
                if (mc==best_mc && nc==best_nc && kc==best_kc) continue;
                double g = bench_fine(mc, nc, kc);
                done++;
                if (g < 0)
                    printf("%-7d %-7d %-7d  FAIL  [%d/%d]\n",
                           mc, nc, kc, done, total_fine);
                else
                    printf("%-7d %-7d %-7d  %.2f  [%d/%d]%s\n",
                           mc, nc, kc, g, done, total_fine,
                           g > best ? "  <- best" : "");
                fflush(stdout);
                if (g > best) {
                    best = g;
                    best_mc = mc; best_nc = nc; best_kc = kc;
                }
            }
        }
    }

    printf("\n==============================================\n");
    printf("  FINAL BEST  (harmonic mean %d…%d)\n",
           FINE_SIZES[0], FINE_SIZES[N_FINE_SIZES-1]);
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