#pragma once

#include "imgen.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define IMGEN_LOG(...) do { fprintf(stderr, __VA_ARGS__); } while (0)

enum imgen_arch {
    IMGEN_ARCH_QWEN_IMAGE,
    IMGEN_ARCH_LUMINA2, // Z-Image
};

enum imgen_vae_type {
    IMGEN_VAE_FLUX, // AutoencoderKL, 16 latent channels (Flux, Z-Image)
    IMGEN_VAE_WAN,  // Wan2.1 causal 3D VAE used as image VAE (Qwen-Image)
};

// tensors of one model file, allocated on one backend buffer
struct imgen_weights {
    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    std::map<std::string, ggml_tensor *> tensors;

    ggml_tensor * get(const std::string & name, bool required = true) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            if (required) {
                throw std::runtime_error("imgen: missing tensor '" + name + "'");
            }
            return nullptr;
        }
        return it->second;
    }

    bool has(const std::string & name) const { return tensors.count(name) > 0; }
};

// load all tensors from a GGUF or safetensors file, converting BF16 to F16
// keep_type: if false, floating point tensors with ndim <= 1 become F32 and others F16
void imgen_load_weights(const std::string & path, ggml_backend_buffer_type_t buft, imgen_weights & out, bool keep_type);

struct imgen_dit_hparams {
    int   n_layers    = 0;
    int   n_refiner   = 0; // lumina2 only
    int   dim         = 0;
    int   n_heads     = 0;
    int   n_kv_heads  = 0;
    int   head_dim    = 0;
    int   txt_dim     = 0; // text encoder hidden size
    int   in_ch       = 16;
    int   patch       = 2;
    int   axes[3]     = {0, 0, 0}; // rope dims per axis (t, h, w)
    float theta       = 0.0f;
    float eps         = 1e-6f;
};

// one graph build + compute on the backend scheduler
struct imgen_runner {
    ggml_backend_t backend_gpu = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_t>             backends;
    std::vector<ggml_backend_buffer_type_t> bufts;
    ggml_backend_sched_ptr sched;
    ggml_backend_buffer_type_t buft_weights = nullptr;

    std::vector<uint8_t> buf_meta;
    ggml_context_ptr     ctx0;
    ggml_cgraph *        gf = nullptr;
    int                  max_nodes = 0;
    int                  n_threads = 0;
    bool                 flash_attn = true;

    imgen_runner(const imgen_params & params, int max_nodes);
    ~imgen_runner();

    ggml_context * graph_begin();
    void graph_compute();
    bool supports(const ggml_tensor * op) const;
};

struct imgen_text_encoder {
    const llama_model * model = nullptr;
    llama_context *     ctx   = nullptr;
    int                 tap_layer = -1; // < 0 = final norm output
    int                 n_tokens_cur = 0;
    std::vector<float>  captured; // hidden states from tap_layer

    ~imgen_text_encoder();
};

struct imgen_context {
    imgen_arch     arch;
    imgen_vae_type vae_type;
    imgen_dit_hparams hp;
    imgen_params   params;

    std::unique_ptr<imgen_runner> runner;
    imgen_weights  dit;
    imgen_weights  vae;
    imgen_text_encoder te;
};

// text conditioning: [txt_dim x n_tokens] row major (token major)
struct imgen_cond {
    int n_tokens = 0;
    std::vector<float> embd;
};

// DiT forward: latent [in_ch*patch*patch x n_img] packed, returns velocity of the same shape
void imgen_dit_forward(imgen_context & ctx, const imgen_cond & cond, int hp, int wp, float t, const std::vector<float> & x, std::vector<float> & out);

// VAE decode: latent [w x h x in_ch] (ggml layout, w fastest) to RGB float in [0,1], [w*8 x h*8 x 3]
void imgen_vae_decode(imgen_context & ctx, int wl, int hl, const std::vector<float> & z, std::vector<float> & rgb);

bool imgen_encode_prompt(imgen_context & ctx, const std::string & prompt, imgen_cond & out);
