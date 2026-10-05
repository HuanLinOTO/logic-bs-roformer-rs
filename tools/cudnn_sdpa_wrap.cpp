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
#include <cstdlib>
#include <dlfcn.h>
#include <memory>
#include <unordered_map>

namespace fe = cudnn_frontend;

// The frontend resolves every cudnn symbol through this handle when built
// with NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING; we own the definition.
namespace cudnn_frontend {
void* cudnn_dlhandle = nullptr;
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

extern "C" {

// Load the cudnn library family from a directory into the frontend's global
// dlhandle, then resolve the few symbols this wrapper needs directly.
int wrap_init(const char* cudnn_lib_dir) {
    static const char* children[] = {"libcudnn_ops.so.9", "libcudnn_cnn.so.9",
        "libcudnn_adv.so.9", "libcudnn_graph.so.9", "libcudnn_heuristic.so.9",
        "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9"};
    char path[512];
    for (auto* c : children) {
        snprintf(path, sizeof path, "%s/%s", cudnn_lib_dir, c);
        if (!dlopen(path, RTLD_GLOBAL | RTLD_NOW)) {
            snprintf(g_err, sizeof g_err, "dlopen %s: %s", path, dlerror());
            return -10;
        }
    }
    snprintf(path, sizeof path, "%s/libcudnn.so.9", cudnn_lib_dir);
    void* h = dlopen(path, RTLD_GLOBAL | RTLD_NOW);
    if (!h) { snprintf(g_err, sizeof g_err, "dlopen %s: %s", path, dlerror()); return -11; }
    cudnn_frontend::cudnn_dlhandle = h;
    p_cudnnCreate = (cudnnStatus_t(*)(cudnnHandle_t*))dlsym(h, "cudnnCreate");
    p_cudnnSetStream = (cudnnStatus_t(*)(cudnnHandle_t, cudaStream_t))dlsym(h, "cudnnSetStream");
    p_cudnnDestroy = (cudnnStatus_t(*)(cudnnHandle_t))dlsym(h, "cudnnDestroy");
    p_cudnnGetVersion = (size_t(*)())dlsym(h, "cudnnGetVersion");
    if (!p_cudnnCreate || !p_cudnnSetStream || !p_cudnnDestroy || !p_cudnnGetVersion) {
        snprintf(g_err, sizeof g_err, "dlsym cudnn core symbols failed");
        return -12;
    }
    // Help the frontend's own shim find a cudart it is happy with.
    if (!getenv("CUDNN_FRONTEND_CUDART_LIB_NAME"))
        setenv("CUDNN_FRONTEND_CUDART_LIB_NAME", "libcudart.so.13", 0);
    return 0;
}

int wrap_create(void** out) {
    cudnnHandle_t h = nullptr;
    cudnnStatus_t st = p_cudnnCreate(&h);
    if (st != CUDNN_STATUS_SUCCESS) {
        snprintf(g_err, sizeof g_err, "cudnnCreate -> %d", (int)st);
        return (int)st;
    }
    *out = (void*)h;
    return 0;
}

int wrap_set_stream(void* h, void* stream) {
    return (int)p_cudnnSetStream((cudnnHandle_t)h, (cudaStream_t)stream);
}

int wrap_destroy(void* h) { return (int)p_cudnnDestroy((cudnnHandle_t)h); }

int wrap_version(void) { return (int)p_cudnnGetVersion(); }

const char* wrap_last_error(void) { return g_err; }

// All strides are in fp16 elements, arrays of 4 = (b, h, s, d).
int wrap_sdpa_build(void* h,
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

int wrap_sdpa_exec(void* graph, void* q, void* k, void* v, void* o, void* ws) {
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

void wrap_sdpa_free(void* graph) { delete (WrapGraph*)graph; }

} // extern "C"
