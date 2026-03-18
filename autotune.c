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

/* ── tuning knobs ── */
#define TUNE_SIZE  100  /* NxN benchmark matrix size   */
#define REPEATS    5      /* timed repetitions per trial */

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
 * BLIS-correct candidate generators
 *
 *  KC  — kc×nR panel of B fits in L1
 *         ideal: KC = L1_bytes / (NR * sizeof(int8))
 *         we search around 40%–100% of that ideal
 *         must be multiple of MR (so micro-kernel K-loop unrolls cleanly)
 *
 *  MC  — mc×kc panel of A fits in L2
 *         ideal: MC = L2_bytes / (KC * sizeof(int8))
 *         must be multiple of MR
 *
 *  NC  — kc×nc panel of B fits in L3
 *         ideal: NC = L3_bytes / (KC * sizeof(int8))
 *         must be multiple of NR
 * ──────────────────────────────────────────────────────────────── */
static void gen_kc(int l1d_kb, int *out, int *n)
{
    /* kc × nR × 1 byte <= L1D
     * => kc <= L1D_bytes / NR
     * use 40–100% of that ceiling */
    int ideal = (l1d_kb * 1024) / NR;
    *n = 0;
    int pct[] = { 100, 88, 75, 63, 50, 40 };
    for (int i = 0; i < 6; i++) {
        int kc = round_down((int)((long long)ideal * pct[i] / 100), MR);
        if (kc < MR) continue;
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j] == kc) { dup = 1; break; }
        if (!dup && *n < 16) out[(*n)++] = kc;
    }
}

static void gen_mc(int l2_kb, int kc, int *out, int *n)
{
    /* mc × kc × 1 byte <= L2
     * => mc <= L2_bytes / kc
     * use 40–100% of that ceiling */
    int ideal = (l2_kb * 1024) / kc;
    *n = 0;
    int pct[] = { 100, 85, 70, 55, 40 };
    for (int i = 0; i < 5; i++) {
        int mc = round_down((int)((long long)ideal * pct[i] / 100), MR);
        if (mc < MR) continue;
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j] == mc) { dup = 1; break; }
        if (!dup && *n < 16) out[(*n)++] = mc;
    }
}

static void gen_nc(int l3_kb, int kc, int *out, int *n)
{
    /* kc × nc × 1 byte <= L3
     * => nc <= L3_bytes / kc
     * use 40–100% of that ceiling */
    int ideal = (l3_kb * 1024) / (kc * NTHREADS);
    *n = 0;
    int pct[] = { 100, 85, 70, 55, 40 };
    for (int i = 0; i < 5; i++) {
        int nc = round_down((int)((long long)ideal * pct[i] / 100), NR);
        if (nc < NR) continue;
        int dup = 0;
        for (int j = 0; j < *n; j++) if (out[j] == nc) { dup = 1; break; }
        if (!dup && *n < 16) out[(*n)++] = nc;
    }
}

/* ─────────────────────────────────────────────────────────────────
 * Write benchmark driver — includes kernel.h directly.
 * MC/NC/KC are injected via -D flags at compile time.
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
"/* -DMC/NC/KC on the gcc cmdline override kernel.h's own defines.\n"
"   Undef here as belt-and-suspenders. */\n"
"#undef MC\n"
"#undef NC\n"
"#undef KC\n"
"#undef min\n"
"#include \"%s\"\n"
"\n"
"int main(void) {\n"
"    int N = %d;\n"
"    int8_t  *A = (int8_t* )malloc((size_t)N * N);\n"
"    int8_t  *B = (int8_t* )malloc((size_t)N * N);\n"
"    int32_t *C = (int32_t*)calloc((size_t)N * N, sizeof(int32_t));\n"
"    if (!A || !B || !C) { fputs(\"OOM\\n\", stderr); return 1; }\n"
"    for (int i = 0; i < N*N; i++) {\n"
"        A[i] = (int8_t)((i * 7 + 3) & 63);\n"
"        B[i] = (int8_t)((i * 5 + 1) & 63);\n"
"    }\n"
"    /* warm-up */\n"
"    kernel(N, N, N, A, N, B, N, C, N);\n"
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
        TUNE_SIZE,
        REPEATS,
        REPEATS);

    fclose(f);
}

/* ─────────────────────────────────────────────────────────────────
 * Compile + run one combination. Returns GOPS or -1.0 on failure.
 * ──────────────────────────────────────────────────────────────── */
static double benchmark(int mc, int nc, int kc)
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

    if (system(cmd) != 0) return -1.0;

    FILE *p = popen(TMP_BIN, "r");
    if (!p) return -1.0;
    double gops = -1.0;
    if (fscanf(p, "%lf", &gops) != 1) gops = -1.0;
    pclose(p);
    return gops;
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

    /* ── show ideal BLIS values for reference ── */
    int kc_ideal = (l1d_kb * 1024) / NR;
    int mc_ideal = (l2_kb  * 1024) / kc_ideal;
    int nc_ideal = (l3_kb * 1024) / (kc_ideal * NTHREADS);
    printf("BLIS ideal (100%% fill, int8):\n");
    printf("  KC_ideal = L1 / NR        = %d\n", kc_ideal);
    printf("  MC_ideal = L2 / KC_ideal  = %d\n", mc_ideal);
    printf("  NC_ideal = L3 / KC_ideal  = %d\n\n", nc_ideal);

    /* ── generate search candidates ── */
    int kc_cands[16], mc_cands[16], nc_cands[16];
    int n_kc, n_mc, n_nc;

    gen_kc(l1d_kb, kc_cands, &n_kc);
    printf("KC candidates: ");
    for (int i = 0; i < n_kc; i++) printf("%d ", kc_cands[i]);
    printf("\n");

    /* count total combinations */
    int total = 0;
    for (int ki = 0; ki < n_kc; ki++) {
        gen_mc(l2_kb, kc_cands[ki], mc_cands, &n_mc);
        gen_nc(l3_kb, kc_cands[ki], nc_cands, &n_nc);
        total += n_mc * n_nc;
    }
    printf("Total combinations: %d\n", total);
    printf("Benchmark: N=%d, %d timed repeats\n\n", TUNE_SIZE, REPEATS);

    /* write driver once */
    write_driver(kernel_h_path);

    /* ── smoke test with kernel.h's own defaults ── */
    printf("Smoke test (MC=192 NC=2048 KC=384)... ");
    fflush(stdout);
    double smoke = benchmark(192, 2048, 384);
    if (smoke < 0) {
        printf("FAILED\n\nCompiler error (/tmp/at_err.txt):\n");
        printf("----------------------------------------------\n");
        system("cat /tmp/at_err.txt");
        printf("----------------------------------------------\n");
        return 1;
    }
    printf("%.2f GOPS — OK\n\n", smoke);

    printf("%-7s %-7s %-7s  %s\n", "MC", "NC", "KC", "GOPS");
    printf("--------------------------------------\n");

    int    best_mc = 192, best_nc = 2048, best_kc = 384;
    double best_gops = smoke;
    int    done = 0;

    for (int ki = 0; ki < n_kc; ki++) {
        int kc = kc_cands[ki];
        gen_mc(l2_kb, kc, mc_cands, &n_mc);
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
                else
                    printf("%-7d %-7d %-7d  %.2f  [%d/%d]\n",
                           mc, nc, kc, g, done, total);
                fflush(stdout);
                if (g > best_gops) {
                    best_gops = g;
                    best_mc = mc; best_nc = nc; best_kc = kc;
                }
            }
        }
    }

    printf("\n==============================================\n");
    printf("  BEST RESULT\n");
    printf("==============================================\n");
    printf("  #define MC  %d\n", best_mc);
    printf("  #define NC  %d\n", best_nc);
    printf("  #define KC  %d\n", best_kc);
    printf("  GOPS = %.2f\n", best_gops);
    printf("\n  Paste these three lines into kernel.h.\n");
    printf("==============================================\n");

    return 0;
}