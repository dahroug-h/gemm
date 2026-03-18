/*
 * autotune.c  —  Auto-tuner for the int8→int32 GEMM in kernel.h
 *
 * Based on the BLIS cache-blocking model:
 *   KC : kc × nR  fills L1 cache   →  KC = L1  / (NR × 1)
 *   MC : mc × kc  fills L2 cache   →  MC = L2  / (KC × 1)
 *   NC : kc × nc  fills L3 cache   →  NC = L3  / (KC × 1)
 *   (sizeof int8 = 1 byte)
 *
 * Build (run from the same directory as kernel.h):
 *   gcc autotune.c -O2 -o autotune -lm
 *   ./autotune
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* ── register-tile size — must match kernel.h ── */
#define MR  8
#define NR  8
#define NTHREADS 4   /* cores sharing L3 */

/* ── tuning knobs ──
 * We sweep multiple representative sizes so the chosen blocking
 * is good across the FULL range 64 … 4000, not just one point.
 * Sizes chosen to stress L1/L2/L3 and memory-bound regimes.       */
static const int TUNE_SIZES[]  = { 256, 512, 1024, 2048, 3072, 4000 };
static const int N_TUNE_SIZES  = 6;
#define REPEATS    8      /* timed repetitions per trial (more = less noise) */
#define WARMUP     2      /* un-timed warm-up reps */

#define TMP_DRIVER  "/tmp/at_driver.c"
#define TMP_BIN     "/tmp/at_run"

/* ─────────────────────────────────────────────────────────────────
 * Read Linux sysfs cache size (KB).
 * index1=L1d  index2=L2  index3=L3
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

/* ─────────────────────────────────────────────────────────────────
 * Candidate generators — FIXED
 *
 * Problem with original: used only 40-100% of BLIS ideal.
 * Reality: the best KC (4072) is far above the BLIS formula because
 * modern CPUs have large L2 and the panel of B can safely be bigger.
 * We now sweep 25% … 400% of the BLIS ideal so we don't miss the
 * real optimum.  Candidates are deduplicated and clamped to sane
 * maximums to keep the search tractable.
 *
 *  KC  max = 8 × L1_bytes / NR   (stay below L2)
 *  MC  max = 2 × L2_bytes / KC   (allow spilling slightly into L3)
 *  NC  max = L3_bytes / KC       (full L3, no per-thread division —
 *                                  threads share L3, don't each need
 *                                  their own copy of the B panel)
 * ──────────────────────────────────────────────────────────────── */
static void gen_kc(int l1d_kb, int l2_kb, int *out, int *n)
{
    /* kc × nR × 1 byte <= L1D  →  ideal = L1D_bytes / NR
     * But extend search up to L2 boundary:
     *   max_kc = L2_bytes / NR  (so A micro-panel still fits in L2) */
    int ideal   = (l1d_kb * 1024) / NR;
    int max_kc  = (l2_kb  * 1024) / NR;   /* upper bound */

    /* percentage steps: 25 40 55 70 85 100 120 150 175 200 250 300 350 400 */
    int pct[] = { 25, 40, 55, 70, 85, 100, 120, 150, 175, 200, 250, 300, 350, 400 };
    int npct  = (int)(sizeof pct / sizeof pct[0]);

    *n = 0;
    for (int i = 0; i < npct; i++) {
        int kc = round_down((int)((long long)ideal * pct[i] / 100), MR);
        if (kc < MR)     continue;
        if (kc > max_kc) kc = round_down(max_kc, MR);  /* clamp */
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j] == kc) { dup = 1; break; }
        if (!dup && *n < 32) out[(*n)++] = kc;
    }
}

static void gen_mc(int l2_kb, int l3_kb, int kc, int *out, int *n)
{
    /* mc × kc × 1 byte <= L2
     * ideal = L2_bytes / kc
     * allow up to 200% (slight L3 spill is OK for MC panel of A) */
    int ideal  = (l2_kb  * 1024) / kc;
    int max_mc = (l3_kb  * 1024) / kc / 2;  /* don't blow entire L3 */
    if (max_mc < ideal) max_mc = ideal * 2;

    int pct[] = { 25, 40, 55, 70, 85, 100, 120, 150, 175, 200 };
    int npct  = (int)(sizeof pct / sizeof pct[0]);

    *n = 0;
    for (int i = 0; i < npct; i++) {
        int mc = round_down((int)((long long)ideal * pct[i] / 100), MR);
        if (mc < MR)     continue;
        if (mc > max_mc) mc = round_down(max_mc, MR);
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j] == mc) { dup = 1; break; }
        if (!dup && *n < 32) out[(*n)++] = mc;
    }
}

static void gen_nc(int l3_kb, int kc, int *out, int *n)
{
    /* kc × nc × 1 byte <= L3
     * FIX: original divided by NTHREADS — wrong.
     * Threads share L3; the NC panel of B is shared, not replicated.
     * ideal = L3_bytes / kc  (full L3 for the B panel)
     * We also search above ideal up to 2×, because NC that's larger
     * than L3 still works (just memory-bound) and can win at large N. */
    int ideal  = (l3_kb * 1024) / kc;
    int max_nc = ideal * 2;

    int pct[] = { 25, 40, 55, 70, 85, 100, 120, 150, 175, 200 };
    int npct  = (int)(sizeof pct / sizeof pct[0]);

    *n = 0;
    for (int i = 0; i < npct; i++) {
        int nc = round_down((int)((long long)ideal * pct[i] / 100), NR);
        if (nc < NR)     continue;
        if (nc > max_nc) nc = round_down(max_nc, NR);
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j] == nc) { dup = 1; break; }
        if (!dup && *n < 32) out[(*n)++] = nc;
    }
}

/* ─────────────────────────────────────────────────────────────────
 * Write benchmark driver — includes kernel.h directly.
 * MC/NC/KC are injected via -D flags at compile time.
 * Driver now accepts the matrix size N as argv[1] so we can reuse
 * the same binary for all TUNE_SIZES without recompiling.
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
"\n"
"#undef MC\n"
"#undef NC\n"
"#undef KC\n"
"#undef min\n"
"#include \"%s\"\n"
"\n"
"int main(int argc, char **argv) {\n"
"    int N = (argc > 1) ? atoi(argv[1]) : 512;\n"
"    int8_t  *A = (int8_t* )malloc((size_t)N * N);\n"
"    int8_t  *B = (int8_t* )malloc((size_t)N * N);\n"
"    int32_t *C = (int32_t*)calloc((size_t)N * N, sizeof(int32_t));\n"
"    if (!A || !B || !C) { fputs(\"OOM\\n\", stderr); return 1; }\n"
"    for (int i = 0; i < N*N; i++) {\n"
"        A[i] = (int8_t)((i * 7 + 3) & 63);\n"
"        B[i] = (int8_t)((i * 5 + 1) & 63);\n"
"    }\n"
"    /* warm-up — not timed */\n"
"    for (int r = 0; r < %d; r++)\n"
"        kernel(N, N, N, A, N, B, N, C, N);\n"
"    memset(C, 0, (size_t)N * N * sizeof(int32_t));\n"
"\n"
"    struct timespec t0, t1;\n"
"    clock_gettime(CLOCK_MONOTONIC, &t0);\n"
"    for (int r = 0; r < %d; r++)\n"
"        kernel(N, N, N, A, N, B, N, C, N);\n"
"    clock_gettime(CLOCK_MONOTONIC, &t1);\n"
"\n"
"    double sec = (t1.tv_sec  - t0.tv_sec)\n"
"               + (t1.tv_nsec - t0.tv_nsec) * 1e-9;\n"
"    double gops = 2.0 * (double)N * (double)N * (double)N\n"
"                  * %d / sec / 1e9;\n"
"    printf(\"%%.3f\\n\", gops);\n"
"    free(A); free(B); free(C);\n"
"    return 0;\n"
"}\n",
        kernel_h_path,
        WARMUP,
        REPEATS,
        REPEATS);

    fclose(f);
}

/* ─────────────────────────────────────────────────────────────────
 * Compile one combination. Returns 0 on success.
 * ──────────────────────────────────────────────────────────────── */
static int compile_combo(int mc, int nc, int kc)
{
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "gcc -O3 -march=native -mavx2 "
        "-DMC=%d -DNC=%d -DKC=%d "
        "%s -o %s "
        "-lm -lpthread "
        "2>/tmp/at_err.txt",
        mc, nc, kc,
        TMP_DRIVER, TMP_BIN);
    return system(cmd);
}

/* ─────────────────────────────────────────────────────────────────
 * Run benchmark for one size N. Returns GOPS or -1.0.
 * ──────────────────────────────────────────────────────────────── */
static double run_size(int N)
{
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "%s %d", TMP_BIN, N);
    FILE *p = popen(cmd, "r");
    if (!p) return -1.0;
    double gops = -1.0;
    if (fscanf(p, "%lf", &gops) != 1) gops = -1.0;
    pclose(p);
    return gops;
}

/* ─────────────────────────────────────────────────────────────────
 * Compile + run across ALL TUNE_SIZES.
 * Score = harmonic mean of GOPS (penalises any size that's slow).
 * Returns score or -1.0 on compile failure.
 * ──────────────────────────────────────────────────────────────── */
static double benchmark(int mc, int nc, int kc)
{
    if (compile_combo(mc, nc, kc) != 0) return -1.0;

    double inv_sum = 0.0;
    int    valid   = 0;
    for (int s = 0; s < N_TUNE_SIZES; s++) {
        double g = run_size(TUNE_SIZES[s]);
        if (g <= 0.0) return -1.0;   /* any failure → reject */
        inv_sum += 1.0 / g;
        valid++;
    }
    if (valid == 0) return -1.0;
    return (double)valid / inv_sum;   /* harmonic mean */
}

/* ═══════════════════════════════════════════════════════════════ */
int main(void)
{
    char cwd[512], kernel_h_path[600];
    if (!getcwd(cwd, sizeof(cwd))) { perror("getcwd"); return 1; }
    snprintf(kernel_h_path, sizeof(kernel_h_path), "%s/kernel.h", cwd);

    printf("==============================================\n");
    printf("  int8 GEMM Auto-Tuner  (BLIS blocking model)\n");
    printf("  MR=%d  NR=%d\n", MR, NR);
    printf("  kernel.h: %s\n", kernel_h_path);
    printf("  Tuning sizes:");
    for (int s = 0; s < N_TUNE_SIZES; s++) printf(" %d", TUNE_SIZES[s]);
    printf("\n");
    printf("  Score = harmonic mean GOPS across all sizes\n");
    printf("==============================================\n\n");

    /* ── cache detection ── */
    int l1d_kb = read_cache_kb(1);
    int l2_kb  = read_cache_kb(2);
    int l3_kb  = read_cache_kb(3);

    if (l1d_kb <= 0) { puts("Warning: L1D unreadable, assuming 32 KB");   l1d_kb = 32;   }
    if (l2_kb  <= 0) { puts("Warning: L2  unreadable, assuming 256 KB");  l2_kb  = 256;  }
    if (l3_kb  <= 0) { puts("Warning: L3  unreadable, assuming 8192 KB"); l3_kb  = 8192; }

    printf("Cache detected:\n");
    printf("  L1D = %4d KB\n", l1d_kb);
    printf("  L2  = %4d KB\n", l2_kb);
    printf("  L3  = %4d KB\n\n", l3_kb);

    /* ── show BLIS ideal for reference ── */
    int kc_ideal = (l1d_kb * 1024) / NR;
    int mc_ideal = (l2_kb  * 1024) / kc_ideal;
    int nc_ideal = (l3_kb  * 1024) / kc_ideal;   /* no /NTHREADS — fixed */
    printf("BLIS ideal (100%% fill, int8, fixed NC formula):\n");
    printf("  KC_ideal = L1 / NR        = %d\n", kc_ideal);
    printf("  MC_ideal = L2 / KC_ideal  = %d\n", mc_ideal);
    printf("  NC_ideal = L3 / KC_ideal  = %d\n\n", nc_ideal);

    /* ── generate search candidates ── */
    int kc_cands[32], mc_cands[32], nc_cands[32];
    int n_kc, n_mc, n_nc;

    gen_kc(l1d_kb, l2_kb, kc_cands, &n_kc);
    printf("KC candidates (%d): ", n_kc);
    for (int i = 0; i < n_kc; i++) printf("%d ", kc_cands[i]);
    printf("\n");

    /* count total combinations */
    int total = 0;
    for (int ki = 0; ki < n_kc; ki++) {
        gen_mc(l2_kb, l3_kb, kc_cands[ki], mc_cands, &n_mc);
        gen_nc(l3_kb, kc_cands[ki], nc_cands, &n_nc);
        total += n_mc * n_nc;
    }
    printf("Total combinations: %d\n", total);
    printf("Reps per size: %d warm-up + %d timed\n\n", WARMUP, REPEATS);

    /* write driver once */
    write_driver(kernel_h_path);

    /* ── smoke test with kernel.h's own defaults ── */
    printf("Smoke test (MC=80 NC=1800 KC=4072)... ");
    fflush(stdout);
    double smoke = benchmark(80, 1800, 4072);
    if (smoke < 0) {
        printf("FAILED\n\nCompiler error (/tmp/at_err.txt):\n");
        printf("----------------------------------------------\n");
        system("cat /tmp/at_err.txt");
        printf("----------------------------------------------\n");
        return 1;
    }
    printf("%.2f GOPS (harmonic mean) — OK\n\n", smoke);

    printf("%-7s %-7s %-7s  %s\n", "MC", "NC", "KC", "SCORE(GOPS hmean)");
    printf("----------------------------------------------\n");

    int    best_mc = 80, best_nc = 1800, best_kc = 4072;
    double best_score = smoke;
    int    done = 0;

    for (int ki = 0; ki < n_kc; ki++) {
        int kc = kc_cands[ki];
        gen_mc(l2_kb, l3_kb, kc, mc_cands, &n_mc);
        gen_nc(l3_kb, kc, nc_cands, &n_nc);

        for (int mi = 0; mi < n_mc; mi++) {
            for (int ni = 0; ni < n_nc; ni++) {
                int mc = mc_cands[mi];
                int nc = nc_cands[ni];
                double g = benchmark(mc, nc, kc);
                done++;
                if (g < 0)
                    printf("%-7d %-7d %-7d  FAIL  [%d/%d]\n",
                           mc, nc, kc, done, total);
                else {
                    printf("%-7d %-7d %-7d  %.2f  [%d/%d]%s\n",
                           mc, nc, kc, g, done, total,
                           g > best_score ? "  ← best" : "");
                    fflush(stdout);
                }
                if (g > best_score) {
                    best_score = g;
                    best_mc = mc; best_nc = nc; best_kc = kc;
                }
            }
        }
    }

    printf("\n==============================================\n");
    printf("  BEST RESULT  (harmonic mean across sizes %d…%d)\n",
           TUNE_SIZES[0], TUNE_SIZES[N_TUNE_SIZES-1]);
    printf("==============================================\n");
    printf("  #define MC  %d\n", best_mc);
    printf("  #define NC  %d\n", best_nc);
    printf("  #define KC  %d\n", best_kc);
    printf("  Score = %.2f GOPS\n", best_score);
    printf("\n  Paste these three lines into kernel.h.\n");
    printf("==============================================\n");

    /* ── machine-readable output for the shell script / CI ── */
    FILE *out = fopen("/tmp/autotune_result.txt", "w");
    if (out) {
        fprintf(out, "%d %d %d\n", best_mc, best_nc, best_kc);
        fclose(out);
        printf("\n  Result also written to /tmp/autotune_result.txt\n");
    }

    return 0;
}