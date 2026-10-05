// tools/cudnn_sdpa_probe.c
// Numeric parity + throughput probe for the cudnn-frontend SDPA wrapper.
// 1) dlopens pip cudnn 9.10 libs (RTLD_GLOBAL) then libcudnn_sdpa_wrap.so
// 2) small fold-layout case (B=4,H=8,S=128,D=64) vs naive CPU softmax
// 3) real shapes timing with cuda events
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* CUDNN_DIR =
    "/data/dsh/lbrr-venv/lib/python3.12/site-packages/nvidia/cudnn/lib";

#define CK(x) do { int _e = (x); if (_e) { \
    fprintf(stderr, "FAIL %s -> %d (line %d)\n", #x, _e, __LINE__); exit(1); } } while (0)

static void* cmalloc(size_t sz) { void* p = NULL; cudaMalloc(&p, sz); return p; }
typedef int (*wrap_init_fn)(const char*);
static wrap_init_fn wrap_init_;
typedef int (*wrap_create_fn)(void**);
typedef int (*wrap_set_stream_fn)(void*, void*);
typedef int (*wrap_version_fn)(void);
typedef const char* (*wrap_err_fn)(void);
typedef int (*wrap_build_fn)(void*, long, long, long, long,
                             const long*, const long*, const long*, const long*,
                             float, int, int, void**, long*);
typedef int (*wrap_exec_fn)(void*, void*, void*, void*, void*, void*);

static wrap_create_fn wrap_create;
static wrap_set_stream_fn wrap_set_stream;
static wrap_version_fn wrap_version;
static wrap_err_fn wrap_err;
static wrap_build_fn wrap_build;
static wrap_exec_fn wrap_exec;

// --- fp16 <-> fp32 host conversion (IEEE 754 half) ---
static unsigned short f32_to_f16(float f) {
    unsigned u; memcpy(&u, &f, 4);
    unsigned sign = (u >> 16) & 0x8000;
    int exp = (int)((u >> 23) & 0xff) - 127 + 15;
    unsigned man = u & 0x7fffff;
    if (((u >> 23) & 0xff) == 0xff) return sign | 0x7c00;  // inf/nan
    if (exp <= 0) return sign;                             // flush to zero
    if (exp >= 31) return sign | 0x7c00;
    return (unsigned short)(sign | (exp << 10) | (man >> 13));
}
static float f16_to_f32(unsigned short h) {
    unsigned sign = (h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1f;
    unsigned man = h & 0x3ff;
    unsigned u;
    if (exp == 0) { if (man == 0) { u = sign; } else { // subnormal
        int e = -1; float m = (float)man / 1024.0f;
        while (m < 0.5f) { m *= 2.0f; e--; }
        u = sign | (((unsigned)(127 + e)) << 23) | ((unsigned)((m * 8388608.0f)) & 0x7fffff);
    } }
    else if (exp == 31) u = sign | 0x7f800000 | (man << 13);
    else u = sign | (((unsigned)(exp - 15 + 127)) << 23) | (man << 13);
    float f; memcpy(&f, &u, 4); return f;
}

int main(void) {
    // ---- dlopen the wrapper; it self-bootstraps the cudnn library family ----
    void* wrap = dlopen("/data/dsh/libcudnn_sdpa_wrap.so", RTLD_NOW | RTLD_LOCAL);
    if (!wrap) { fprintf(stderr, "dlopen wrap: %s\n", dlerror()); return 1; }
    wrap_init_ = (wrap_init_fn)dlsym(wrap, "wrap_init");
    if (!wrap_init_) { fprintf(stderr, "no wrap_init\n"); return 1; }
    CK(wrap_init_(CUDNN_DIR));
    wrap_create = (wrap_create_fn)dlsym(wrap, "wrap_create");
    wrap_set_stream = (wrap_set_stream_fn)dlsym(wrap, "wrap_set_stream");
    wrap_version = (wrap_version_fn)dlsym(wrap, "wrap_version");
    wrap_err = (wrap_err_fn)dlsym(wrap, "wrap_last_error");
    wrap_build = (wrap_build_fn)dlsym(wrap, "wrap_sdpa_build");
    wrap_exec = (wrap_exec_fn)dlsym(wrap, "wrap_sdpa_exec");
    if (!wrap_create || !wrap_build || !wrap_exec) { fprintf(stderr, "dlsym missing\n"); return 1; }

    void* handle = NULL;
    CK(wrap_create(&handle));
    printf("cudnn version: %d\n", wrap_version());

    // =========== numeric parity: B=4, H=8, S=128, D=64, fold layout ===========
    {
        const long B = 4, H = 8, S = 128, D = 64, TOK = B * S;
        const float SCALE = 0.125f;
        // host buffers in folded layout: qkv [TOK][1536], kr [TOK][512], out [TOK][512]
        unsigned short* h_qkv = malloc(TOK * 1536 * 2);
        unsigned short* h_kr  = malloc(TOK * 512 * 2);
        unsigned short* h_out = malloc(TOK * 512 * 2);
        for (long tok = 0; tok < TOK; tok++) {
            for (long i = 0; i < 1536; i++)
                h_qkv[tok * 1536 + i] = f32_to_f16(sinf((float)(i * 7 + tok * 13) * 0.017f));
            for (long i = 0; i < 512; i++)
                h_kr[tok * 512 + i] = f32_to_f16(cosf((float)(i * 5 + tok * 3) * 0.019f));
        }
        unsigned short *d_qkv, *d_kr, *d_out; void* d_ws = NULL;
        d_qkv = (unsigned short*)cmalloc(TOK * 1536 * 2); d_kr = (unsigned short*)cmalloc(TOK * 512 * 2);
        d_out = (unsigned short*)cmalloc(TOK * 512 * 2);
        cudaMemcpy(d_qkv, h_qkv, TOK * 1536 * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kr, h_kr, TOK * 512 * 2, cudaMemcpyHostToDevice);
        // element strides (b, h, s, d) in fp16 units
        long q_str[4] = {1536, 64, B * 1536, 1};      // Q view of qkv @0
        long k_str[4] = {512, 64, B * 512, 1};        // K from kr buffer
        long v_str[4] = {1536, 64, B * 1536, 1};      // V view of qkv @1024
        long o_str[4] = {512, 64, B * 512, 1};        // O folded token-major
        void* g = NULL; long ws = 0;
        // matrix: dtype x heur mode x fold/packed
        for (int dt = 0; dt < 2 && !g; dt++)
        for (int mode = 0; mode < 3 && !g; mode++) {
            int rc = wrap_build(handle, B, H, S, D, q_str, k_str, v_str, o_str, SCALE, mode, dt, &g, &ws);
            printf("dt%d mode%d fold: rc=%d %s\n", dt, mode, rc, rc ? wrap_err() : "OK");
        }
        if (!g) {
            long pq[4] = {H * S * D, S * D, D, 1};
            for (int dt = 0; dt < 2 && !g; dt++)
            for (int mode = 0; mode < 3 && !g; mode++) {
                int rc = wrap_build(handle, B, H, S, D, pq, pq, pq, pq, SCALE, mode, dt, &g, &ws);
                printf("dt%d mode%d packed: rc=%d %s\n", dt, mode, rc, rc ? wrap_err() : "OK");
            }
        }
        if (!g) { fprintf(stderr, "all builds failed\n"); return 1; }
        printf("parity plan ws=%ld bytes\n", ws);
        if (ws > 0) d_ws = cmalloc((size_t)ws);
        CK(wrap_exec(g, d_qkv, d_kr, (char*)d_qkv + 1024 * 2, d_out, d_ws));
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, TOK * 512 * 2, cudaMemcpyDeviceToHost);

        // naive reference, fp32 accumulate
        double sum_abs = 0, sum_ref = 0, max_abs = 0;
        float* srow = malloc(S * sizeof(float));
        for (long b = 0; b < B; b++) for (long h = 0; h < H; h++) {
            for (long t = 0; t < S; t++) {
                float m = -1e30f;
                for (long k = 0; k < S; k++) {
                    float acc = 0;
                    for (long d = 0; d < D; d++)
                        acc += f16_to_f32(h_qkv[(k * B + b) * 1536 + h * 64 + d]) * SCALE
                             * f16_to_f32(h_kr[(k * B + b) * 512 + h * 64 + d]);
                    srow[k] = acc; if (acc > m) m = acc;
                }
                float z = 0;
                for (long k = 0; k < S; k++) { srow[k] = expf(srow[k] - m); z += srow[k]; }
                for (long d = 0; d < D; d++) {
                    float ref = 0;
                    for (long k = 0; k < S; k++)
                        ref += srow[k] * f16_to_f32(h_qkv[(k * B + b) * 1536 + 1024 + h * 64 + d]);
                    ref /= z;
                    float got = f16_to_f32(h_out[(t * B + b) * 512 + h * 64 + d]);
                    double ad = fabs((double)got - ref);
                    sum_abs += ad; sum_ref += fabs((double)ref);
                    if (ad > max_abs) max_abs = ad;
                }
            }
        }
        printf("parity: max_abs=%.3e rel=%.3e (S=%ld)\n", max_abs, sum_abs / sum_ref, S);
        { FILE* f = fopen("/data/dsh/sdpa_parity.bin", "wb");
          long dims[4] = {B, H, S, D}; fwrite(dims, 8, 4, f);
          fwrite(h_qkv, 2, TOK * 1536, f); fwrite(h_kr, 2, TOK * 512, f);
          fwrite(h_out, 2, TOK * 512, f); fclose(f); printf("dumped parity buffers\n"); }
    }


    // =========== numeric parity 2: FREQ-axis fold layout ===========
    // (B groups over t, H=8, S=62 bands, D=64); token = t*62 + band.
    {
        const long B = 80, H = 8, S = 62, D = 64, BANDS = 62, TOK = B * BANDS;
        const float SCALE = 0.125f;
        unsigned short* h_qkv = malloc(TOK * 1536 * 2);
        unsigned short* h_kr  = malloc(TOK * 512 * 2);
        unsigned short* h_out = malloc(TOK * 512 * 2);
        for (long tok = 0; tok < TOK; tok++) {
            for (long i = 0; i < 1536; i++)
                h_qkv[tok * 1536 + i] = f32_to_f16(sinf((float)(i * 7 + tok * 13) * 0.017f));
            for (long i = 0; i < 512; i++)
                h_kr[tok * 512 + i] = f32_to_f16(cosf((float)(i * 5 + tok * 3) * 0.019f));
        }
        unsigned short *d_qkv, *d_kr, *d_out; void* d_ws = NULL;
        d_qkv = (unsigned short*)cmalloc(TOK * 1536 * 2); d_kr = (unsigned short*)cmalloc(TOK * 512 * 2);
        d_out = (unsigned short*)cmalloc(TOK * 512 * 2);
        cudaMemcpy(d_qkv, h_qkv, TOK * 1536 * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kr, h_kr, TOK * 512 * 2, cudaMemcpyHostToDevice);
        long q_str[4] = {BANDS * 1536, 64, 1536, 1};   // (t, head, band, d)
        long k_str[4] = {BANDS * 512, 64, 512, 1};
        long v_str[4] = {BANDS * 1536, 64, 1536, 1};
        long o_str[4] = {BANDS * 512, 64, 512, 1};
        void* g = NULL; long ws = 0;
        int rc = wrap_build(handle, B, H, S, D, q_str, k_str, v_str, o_str, SCALE, 0, 0, &g, &ws);
        if (rc) { fprintf(stderr, "freq build rc=%d: %s\n", rc, wrap_err()); return 1; }
        printf("freq parity plan ws=%ld\n", ws);
        if (ws > 0) d_ws = cmalloc((size_t)ws);
        CK(wrap_exec(g, d_qkv, d_kr, (char*)d_qkv + 1024 * 2, d_out, d_ws));
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, TOK * 512 * 2, cudaMemcpyDeviceToHost);
        double sum_abs = 0, sum_ref = 0, max_abs = 0;
        float* srow = malloc(S * sizeof(float));
        for (long t = 0; t < B; t += 37) for (long h = 0; h < H; h++) {
            for (long bq = 0; bq < S; bq++) {
                float m = -1e30f;
                for (long bk = 0; bk < S; bk++) {
                    float acc = 0;
                    for (long d = 0; d < D; d++)
                        acc += f16_to_f32(h_qkv[(t * BANDS + bq) * 1536 + h * 64 + d]) * SCALE
                             * f16_to_f32(h_kr[(t * BANDS + bk) * 512 + h * 64 + d]);
                    srow[bk] = acc; if (acc > m) m = acc;
                }
                float z = 0;
                for (long bk = 0; bk < S; bk++) { srow[bk] = expf(srow[bk] - m); z += srow[bk]; }
                for (long d = 0; d < D; d++) {
                    float ref = 0;
                    for (long bk = 0; bk < S; bk++)
                        ref += srow[bk] * f16_to_f32(h_qkv[(t * BANDS + bk) * 1536 + 1024 + h * 64 + d]);
                    ref /= z;
                    float got = f16_to_f32(h_out[(t * BANDS + bq) * 512 + h * 64 + d]);
                    double ad = fabs((double)got - ref);
                    sum_abs += ad; sum_ref += fabs((double)ref);
                    if (ad > max_abs) max_abs = ad;
                }
            }
        }
        printf("freq parity: max_abs=%.3e rel=%.3e\n", max_abs, sum_abs / sum_ref);
        { FILE* f = fopen("/data/dsh/sdpa_freq.bin", "wb");
          long dims[4] = {B, H, S, D}; fwrite(dims, 8, 4, f);
          fwrite(h_qkv, 2, TOK * 1536, f); fwrite(h_kr, 2, TOK * 512, f);
          fwrite(h_out, 2, TOK * 512, f); fclose(f); }
    }


    // =========== replay: Rust-dumped real L0-freq inputs through the freq graph ===========
    // First mimic the Rust process: build + warm the TIME plan (62,8,259,64)
    // with time-layout strides, then run the freq replay.
    {
        const long B = 259, S = 62, D = 64, TOK = B * 62;
        FILE* f = fopen("/data/dsh/dump_cudnn_qkv16.bin", "rb");
        if (f) {
            unsigned short* h_qkv = malloc(TOK * 1536 * 2);
            unsigned short* h_kr  = malloc(TOK * 512 * 2);
            unsigned short* h_out = malloc(TOK * 512 * 2);
            fread(h_qkv, 2, TOK * 1536, f); fclose(f);
            f = fopen("/data/dsh/dump_cudnn_k16r.bin", "rb");
            fread(h_kr, 2, TOK * 512, f); fclose(f);
            unsigned short *d_qkv, *d_kr, *d_out;
            d_qkv = (unsigned short*)cmalloc(TOK * 1536 * 2); d_kr = (unsigned short*)cmalloc(TOK * 512 * 2);
            d_out = (unsigned short*)cmalloc(TOK * 512 * 2);
            cudaMemcpy(d_qkv, h_qkv, TOK * 1536 * 2, cudaMemcpyHostToDevice);
            cudaMemcpy(d_kr, h_kr, TOK * 512 * 2, cudaMemcpyHostToDevice);
            long q_str[4] = {1536, 64, 62 * 1536, 1};   // time layout
            long k_str[4] = {512, 64, 62 * 512, 1};
            long o_str[4] = {512, 64, 62 * 512, 1};
            void* g = NULL; long ws = 0;
            int rc = wrap_build(handle, 62, 8, 259, D, q_str, k_str, q_str, o_str, 0.125f, 0, 0, &g, &ws);
            if (!rc) {
                for (int i = 0; i < 3; i++)
                    CK(wrap_exec(g, d_qkv, d_kr, (char*)d_qkv + 1024 * 2, d_out, ws ? cmalloc((size_t)ws) : NULL));
                cudaDeviceSynchronize();
                printf("time-plan warm done (ws=%ld)\n", ws);
            } else { printf("time-plan warm build rc=%d\n", rc); }
        }
    }
    {
        const long B = 259, H = 8, S = 62, D = 64, BANDS = 62, TOK = B * BANDS;
        FILE* f = fopen("/data/dsh/dump_cudnn_qkv16.bin", "rb");
        if (!f) { fprintf(stderr, "no qkv dump\n"); } else {
        unsigned short* h_qkv = malloc(TOK * 1536 * 2);
        unsigned short* h_kr  = malloc(TOK * 512 * 2);
        unsigned short* h_out = malloc(TOK * 512 * 2);
        fread(h_qkv, 2, TOK * 1536, f); fclose(f);
        f = fopen("/data/dsh/dump_cudnn_k16r.bin", "rb");
        if (fread(h_kr, 2, TOK * 512, f) != (size_t)(TOK * 512)) { fprintf(stderr, "k16r short\n"); }
        fclose(f);
        unsigned short *d_qkv, *d_kr, *d_out; void* d_ws = NULL;
        d_qkv = (unsigned short*)cmalloc(TOK * 1536 * 2); d_kr = (unsigned short*)cmalloc(TOK * 512 * 2);
        d_out = (unsigned short*)cmalloc(TOK * 512 * 2);
        cudaMemcpy(d_qkv, h_qkv, TOK * 1536 * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kr, h_kr, TOK * 512 * 2, cudaMemcpyHostToDevice);
        long q_str[4] = {BANDS * 1536, 64, 1536, 1};
        long k_str[4] = {BANDS * 512, 64, 512, 1};
        long v_str[4] = {BANDS * 1536, 64, 1536, 1};
        long o_str[4] = {BANDS * 512, 64, 512, 1};
        void* g = NULL; long ws = 0;
        int rc = wrap_build(handle, B, H, S, D, q_str, k_str, v_str, o_str, 0.125f, 0, 0, &g, &ws);
        if (rc) { fprintf(stderr, "replay build rc=%d: %s\n", rc, wrap_err()); return 1; }
        if (ws > 0) d_ws = cmalloc((size_t)ws);
        CK(wrap_exec(g, d_qkv, d_kr, (char*)d_qkv + 1024 * 2, d_out, d_ws));
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, TOK * 512 * 2, cudaMemcpyDeviceToHost);
        f = fopen("/data/dsh/freq_replay2.bin", "wb");
        fwrite(h_out, 2, TOK * 512, f); fclose(f);
        printf("replay written (%ld tokens)\n", TOK);
        }
    }

    // =========== throughput on real shapes ===========
    struct { const char* tag; long b, s; double t; } cases[] = {
        {"time-song ", 62, 1151}, {"time-bench", 62, 259},
        {"freq-song ", 1151, 62}, {"freq-bench", 259, 62},
    };
    for (unsigned c = 0; c < 4; c++) {
        long B = cases[c].b, H = 8, S = cases[c].s, D = 64, TOK = B * S;
        unsigned short *d_qkv, *d_kr, *d_out; void* d_ws = NULL;
        d_qkv = (unsigned short*)cmalloc(TOK * 1536 * 2); d_kr = (unsigned short*)cmalloc(TOK * 512 * 2);
        d_out = (unsigned short*)cmalloc(TOK * 512 * 2); cudaMemset(d_qkv, 0x3c, TOK * 1536 * 2);
        cudaMemset(d_kr, 0x3c, TOK * 512 * 2);
        long q_str[4] = {1536, 64, B * 1536, 1};
        long k_str[4] = {512, 64, B * 512, 1};
        long v_str[4] = {1536, 64, B * 1536, 1};
        long o_str[4] = {512, 64, B * 512, 1};
        void* g = NULL; long ws = 0;
        int rc = wrap_build(handle, B, H, S, D, q_str, k_str, v_str, o_str, 0.125f, 0, 0, &g, &ws);
        if (rc) rc = wrap_build(handle, B, H, S, D, q_str, k_str, v_str, o_str, 0.125f, 1, 0, &g, &ws);
        if (rc) rc = wrap_build(handle, B, H, S, D, q_str, k_str, v_str, o_str, 0.125f, 2, 0, &g, &ws);
        if (rc) { fprintf(stderr, "%s build rc=%d: %s\n", cases[c].tag, rc, wrap_err()); continue; }
        if (ws > 0) d_ws = cmalloc((size_t)ws);
        for (int i = 0; i < 20; i++)
            CK(wrap_exec(g, d_qkv, d_kr, (char*)d_qkv + 1024 * 2, d_out, d_ws));
        cudaDeviceSynchronize();
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, 0);
        const int N = 200;
        for (int i = 0; i < N; i++)
            CK(wrap_exec(g, d_qkv, d_kr, (char*)d_qkv + 1024 * 2, d_out, d_ws));
        cudaEventRecord(e1, 0); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= N;
        double fl = 2.0 * 2.0 * B * H * S * S * D;
        printf("%s B=%ld S=%ld: %.3f ms  %.1f TFLOPs  ws=%ld\n",
               cases[c].tag, B, S, ms, fl / (ms * 1e-3) * 1e-12, ws);
        cudaFree(d_qkv); cudaFree(d_kr); cudaFree(d_out); if (d_ws) cudaFree(d_ws);
    }
    printf("PROBE-OK\n");
    return 0;
}
