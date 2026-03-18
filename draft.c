// --- تعديل دالة pack_A لضمان الـ Zero-Out بـ 128 ---
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
                // الحشو بـ 128 لضمان ان النتيجة تكون صفر بعد الطرح
                *Buffer_A++ = 128; *Buffer_A++ = 128; *Buffer_A++ = 128; *Buffer_A++ = 128;
            }
        }
    }
}

// --- تعديل دالة pack_B لتصليح الـ Indexing ---
void pack_B(int8_t* B, int8_t* Buffer_B, int nc, int kc, int col_start, int row_start, int LDB, int32_t* B_col_correction) {
    // تصفير الجزء الخاص بالبلوك الحالي فقط في الـ correction array
    for (int x = 0; x < nc; x++) {
        B_col_correction[col_start + x] = 0;
    }
    
    for (int j_local = 0; j_local < nc; j_local += 16) {
        int nr = min(16, nc - j_local);
        for (int p_local = 0; p_local < kc; p_local += 4) { 
            for (int i = 0; i < nr; i++) {
                int8_t v0 = (p_local + 0 < kc) ? B(row_start + p_local + 0, col_start + j_local + i) : 0;
                int8_t v1 = (p_local + 1 < kc) ? B(row_start + p_local + 1, col_start + j_local + i) : 0;
                int8_t v2 = (p_local + 2 < kc) ? B(row_start + p_local + 2, col_start + j_local + i) : 0;
                int8_t v3 = (p_local + 3 < kc) ? B(row_start + p_local + 3, col_start + j_local + i) : 0;
                
                *Buffer_B++ = v0; *Buffer_B++ = v1; *Buffer_B++ = v2; *Buffer_B++ = v3;
                
                // التعديل: استخدام col_start + j_local + i لضمان الوصول للعمود الصح في المصفوفة الكبيرة N
                B_col_correction[col_start + j_local + i] += (int32_t)v0 * 128 + (int32_t)v1 * 128 + (int32_t)v2 * 128 + (int32_t)v3 * 128;
            }
            for (int i = nr; i < 16; i++) {
                *Buffer_B++ = 0; *Buffer_B++ = 0; *Buffer_B++ = 0; *Buffer_B++ = 0;
            }
        }
    }
}

// --- دالة الـ Kernel النهائية ---
void kernel(int32_t M, int32_t N, int32_t K, int8_t* A, int LDA, int8_t* B, int LDB, int32_t* C, int LDC) {

    int32_t* B_col_correction = (int32_t*)malloc(N * sizeof(int32_t));
    
    for (int j = 0; j < N; j += NC) {
        int nc = min(N-j, NC);
        for(int p = 0; p < K; p += KC) {
            int kc = min(K-p, KC);
            
            // نمرر j كـ col_start
            pack_B(B, Buffer_B, nc, kc, j, p, LDB, B_col_correction);
            int kc_padded = (kc + 3) & ~3; 

            for(int i = 0; i < M; i += MC) {
                int mc = min(M-i, MC);
                pack_A(A, Buffer_A, mc, kc, i, p, LDA);
                
                for(int jr = 0; jr < nc; jr += 16) {
                    int nr = min(nc-jr, 16);
                    for(int ir = 0; ir < mc; ir += 6) {
                        int mr = min(mc-ir, 6);

                        macro_kernel(mr, nr, kc, 
                                     &Buffer_A[ir * kc_padded], 
                                     &Buffer_B[jr * kc_padded], 
                                     &C(i+ir, j+jr), LDC);

                        // طرح الـ Bias لكل صف وعمود في التايل الحالي
                        for (int r = 0; r < mr; r++) {
                            for (int c = 0; c < nr; c++) {
                                // j+jr+c هو الـ Index الحقيقي للعمود في المصفوفة N
                                C(i + ir + r, j + jr + c) -= B_col_correction[j + jr + c];
                            }
                        }
                    }
                }
            }
        }
    }
    free(B_col_correction);
}