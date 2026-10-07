// tools/cudnn_sdpa_wrap.cpp
// Thin extern "C" wrapper over cudnn-frontend's fused SDPA (flash attention)
// graph. Compiled to a standalone .so with undefined cudnn symbols; the Rust
// side dlopens libcudnn*.so first (RTLD_GLOBAL) so this library resolves.
//
// Tensor geometry is fixed to our BS-RoFormer case: head_dim = 64, H = 8
// heads, BHSD dims parameterized, arbitrary element strides so the folded
// qkv16 layout (token-major [T*62][1536] fp16) can be viewed in place:
//   time axis: (B=62, H=8, S=T, D=64) -> strides (1536, 64, 62*1536, 1)
//              K comes from the separate roped buffer: (512, 64, 62*512, 1)
//   freq axis: (B=T, H=8, S=62, D=64) -> strides (62*1536, 64, 1536, 1)
//              (band stride 1536 = one token row)
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unordered_map>

#ifdef _WIN32
// Windows port: LoadLibrary/GetProcAddress replace dlopen/dlsym. The
// frontend resolves every cudnn symbol through cudnn_dlhandle, so there is
// no RTLD_GLOBAL equivalent to provide — loading each DLL once is enough.
#include <windows.h>
static void* my_dlopen(const char* path) { return (void*)LoadLibraryA(path); }
static void* my_dlsym(void* h, const char* name) { return (void*)GetProcAddress((HMODULE)h, name); }
static const char* my_dlerror() {
    static thread_local char buf[64];
    snprintf(buf, sizeof buf, "GetLastError=%lu", (unsigned long)GetLastError());
    return buf;
}
static void my_setenv_default(const char* n, const char* v) {
    if (!getenv(n)) _putenv_s(n, v);
}
#else
#include <dlfcn.h>
static void* my_dlopen(const char* path) { return dlopen(path, RTLD_GLOBAL | RTLD_NOW); }
static void* my_dlsym(void* h, const char* name) { return dlsym(h, name); }
static const char* my_dlerror() { return dlerror(); }
static void my_setenv_default(const char* n, const char* v) { setenv(n, v, 0); }
#endif

namespace fe = cudnn_frontend;

// The frontend resolves every cudnn symbol through this handle when built
// with NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING; we own the definition.
namespace cudnn_frontend {
#ifdef _WIN32
HMODULE cudnn_dlhandle = nullptr;
#else
void* cudnn_dlhandle = nullptr;
#endif
}

static cudnnStatus_t (*p_cudnnCreate)(cudnnHandle_t*);
static cudnnStatus_t (*p_cudnnSetStream)(cudnnHandle_t, cudaStream_t);
static cudnnStatus_t (*p_cudnnDestroy)(cudnnHandle_t);
static size_t (*p_cudnnGetVersion)();

struct WrapGraph {
    std::shared_ptr<fe::graph::Graph> g;
    cudnnHandle_t h;
    size_t ws;
};

static thread_local char g_err[512] = {0};

// .so exports every global symbol; a Windows DLL exports nothing unless
// asked, so mark the extern "C" surface explicitly.
#ifdef _WIN32
#define WRAP_API __declspec(dllexport)
#else
#define WRAP_API __attribute__((visibility("default")))
#endif

extern "C" {

// Load the cudnn library family from a directory into the frontend's global
// dlhandle, then resolve the few symbols this wrapper needs directly.
WRAP_API int wrap_init(const char* cudnn_lib_dir) {
#ifdef _WIN32
    // cuDNN 9 Windows wheel: one DLL per Linux sub-library, *_64_9.dll
    // naming, plus two extras (ext, engines_tensor_ir) the Linux build
    // ships inside libcudnn.so.9 itself.
    static const char* children[] = {"cudnn_ops64_9.dll", "cudnn_cnn64_9.dll",
        "cudnn_adv64_9.dll", "cudnn_graph64_9.dll", "cudnn_heuristic64_9.dll",
        "cudnn_engines_precompiled64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_ext64_9.dll", "cudnn_engines_tensor_ir64_9.dll"};
    static const char* main_lib = "cudnn64_9.dll";
    static const char* cudart_candidates[] = {"cudart64_13.dll", "cudart64_12.dll"};
#else
    static const char* children[] = {"libcudnn_ops.so.9", "libcudnn_cnn.so.9",
        "libcudnn_adv.so.9", "libcudnn_graph.so.9", "libcudnn_heuristic.so.9",
        "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9"};
    static const char* main_lib = "libcudnn.so.9";
    static const char* cudart_candidates[] = {"libcudart.so.13", "libcudart.so.12"};
#endif
    char path[512];
    for (auto* c : children) {
        snprintf(path, sizeof path, "%s/%s", cudnn_lib_dir, c);
        if (!my_dlopen(path)) {
            snprintf(g_err, sizeof g_err, "dlopen %s: %s", path, my_dlerror());
            return -10;
        }
    }
    snprintf(path, sizeof path, "%s/%s", cudnn_lib_dir, main_lib);
    void* h = my_dlopen(path);
    if (!h) { snprintf(g_err, sizeof g_err, "dlopen %s: %s", path, my_dlerror()); return -11; }
#ifdef _WIN32
    cudnn_frontend::cudnn_dlhandle = (HMODULE)h;
#else
    cudnn_frontend::cudnn_dlhandle = h;
#endif
    p_cudnnCreate = (cudnnStatus_t(*)(cudnnHandle_t*))my_dlsym(h, "cudnnCreate");
    p_cudnnSetStream = (cudnnStatus_t(*)(cudnnHandle_t, cudaStream_t))my_dlsym(h, "cudnnSetStream");
    p_cudnnDestroy = (cudnnStatus_t(*)(cudnnHandle_t))my_dlsym(h, "cudnnDestroy");
    p_cudnnGetVersion = (size_t(*)())my_dlsym(h, "cudnnGetVersion");
    if (!p_cudnnCreate || !p_cudnnSetStream || !p_cudnnDestroy || !p_cudnnGetVersion) {
        snprintf(g_err, sizeof g_err, "dlsym cudnn core symbols failed");
        return -12;
    }
    // Help the frontend's own shim find a cudart it is happy with.
    // CUDA 13.x names the runtime cudart64_130_0.dll while 12.x uses
    // cudart64_12.dll. Probe beside the cudnn libraries first (portable
    // bundle layout; an absolute path also survives bare-name loader
    // searches that skip the exe dir on Linux), then bare names through
    // the loader. Note the env roundtrip must stay on this side of the
    // FFI: MSVC getenv reads the CRT snapshot, which does not see Win32
    // SetEnvironmentVariable writes done by the Rust host.
    for (auto* cand : cudart_candidates) {
        snprintf(path, sizeof path, "%s/%s", cudnn_lib_dir, cand);
        if (my_dlopen(path)) {
            my_setenv_default("CUDNN_FRONTEND_CUDART_LIB_NAME", path);
            return 0;
        }
    }
    for (auto* cand : cudart_candidates) {
        if (my_dlopen(cand)) {
            my_setenv_default("CUDNN_FRONTEND_CUDART_LIB_NAME", cand);
            return 0;
        }
    }
    my_setenv_default("CUDNN_FRONTEND_CUDART_LIB_NAME", cudart_candidates[0]);
    return 0;
}

WRAP_API int wrap_create(void** out) {
    cudnnHandle_t h = nullptr;
    cudnnStatus_t st = p_cudnnCreate(&h);
    if (st != CUDNN_STATUS_SUCCESS) {
        snprintf(g_err, sizeof g_err, "cudnnCreate -> %d", (int)st);
        return (int)st;
    }
    *out = (void*)h;
    return 0;
}

WRAP_API int wrap_set_stream(void* h, void* stream) {
    return (int)p_cudnnSetStream((cudnnHandle_t)h, (cudaStream_t)stream);
}

WRAP_API int wrap_destroy(void* h) { return (int)p_cudnnDestroy((cudnnHandle_t)h); }

WRAP_API int wrap_version(void) { return (int)p_cudnnGetVersion(); }

WRAP_API const char* wrap_last_error(void) { return g_err; }

// All strides are in fp16 elements, arrays of 4 = (b, h, s, d).
WRAP_API int wrap_sdpa_build(void* h,
                    int64_t b, int64_t heads, int64_t s, int64_t d,
                    const int64_t* q_str, const int64_t* k_str,
                    const int64_t* v_str, const int64_t* o_str,
                    float attn_scale, int heur_mode, int io_dtype,
                    void** graph_out, int64_t* ws_out) {
    try {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(io_dtype == 1 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);
        auto Q = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Q").set_uid(1)
                                   .set_dim({b, heads, s, d})
                                   .set_stride({q_str[0], q_str[1], q_str[2], q_str[3]}));
        auto K = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("K").set_uid(2)
                                   .set_dim({b, heads, s, d})
                                   .set_stride({k_str[0], k_str[1], k_str[2], k_str[3]}));
        auto V = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("V").set_uid(3)
                                   .set_dim({b, heads, s, d})
                                   .set_stride({v_str[0], v_str[1], v_str[2], v_str[3]}));
        auto attrs = fe::graph::SDPA_attributes()
                         .set_name("sdpa")
                         .set_is_inference(true)
                         .set_attn_scale(attn_scale);
        auto out = graph->sdpa(Q, K, V, attrs);
        auto O = out[0];
        O->set_output(true)
            .set_dim({b, heads, s, d})
            .set_stride({o_str[0], o_str[1], o_str[2], o_str[3]})
            .set_uid(4);
        fe::HeurMode_t mode = fe::HeurMode_t::A;
        if (heur_mode == 1) mode = fe::HeurMode_t::B;
        if (heur_mode == 2) mode = fe::HeurMode_t::FALLBACK;
        auto st = graph->build((cudnnHandle_t)h, {mode});
        if (!st.is_good()) {
            snprintf(g_err, sizeof g_err, "build: code=%d msg=%s",
                     (int)st.get_code(), st.get_message().c_str());
            return -1;
        }
        auto* wg = new WrapGraph{graph, (cudnnHandle_t)h, (size_t)graph->get_workspace_size()};
        *graph_out = wg;
        *ws_out = (int64_t)wg->ws;
        return 0;
    } catch (fe::cudnnException const& e) {
        snprintf(g_err, sizeof g_err, "cudnnException: %s", e.what());
        return -2;
    } catch (std::exception const& e) {
        snprintf(g_err, sizeof g_err, "std::exception: %s", e.what());
        return -3;
    }
}

WRAP_API int wrap_sdpa_exec(void* graph, void* q, void* k, void* v, void* o, void* ws) {
    try {
        auto* wg = (WrapGraph*)graph;
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
            {1, q}, {2, k}, {3, v}, {4, o}};
        auto st = wg->g->execute(wg->h, pack, ws);
        if (!st.is_good()) {
            snprintf(g_err, sizeof g_err, "execute: code=%d msg=%s",
                     (int)st.get_code(), st.get_message().c_str());
            return -4;
        }
        return 0;
    } catch (std::exception const& e) {
        snprintf(g_err, sizeof g_err, "exec exception: %s", e.what());
        return -5;
    }
}

WRAP_API void wrap_sdpa_free(void* graph) { delete (WrapGraph*)graph; }

} // extern "C"
