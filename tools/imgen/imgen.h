#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// Text-to-image generation with diffusion transformers (Qwen-Image, Z-Image).
//
// The DiT weights come from a GGUF file (general.architecture = qwen_image | lumina2),
// the VAE decoder from a safetensors or GGUF file, and the text encoder is a regular
// llama_model (Qwen2.5-VL for qwen_image, Qwen3 for lumina2).

#ifdef __cplusplus
extern "C" {
#endif

struct imgen_context;

struct imgen_params {
    bool use_gpu;
    ggml_backend_dev_t device; // optional, nullptr = auto
    int  n_threads;
    bool flash_attn;
    bool verbose;
};

struct imgen_request {
    const char * prompt;
    const char * negative_prompt; // optional
    int     width;                // multiple of 16, 0 = 1024
    int     height;               // multiple of 16, 0 = 1024
    int     steps;                // 0 = model default
    float   cfg_scale;            // < 0 = model default
    int64_t seed;                 // < 0 = random
};

struct imgen_image {
    int       width;
    int       height;
    uint8_t * data; // RGB, width*height*3 bytes, free with imgen_image_free
};

// called after each denoising step, return false to cancel
typedef bool (*imgen_progress_callback)(int step, int n_steps, void * user_data);

GGML_API struct imgen_params imgen_params_default(void);

// true when the GGUF at path is a supported diffusion transformer
GGML_API bool imgen_is_dit_gguf(const char * path);

// text_encoder must outlive the context
GGML_API struct imgen_context * imgen_init(
        const char *         dit_path,
        const char *         vae_path,
        const llama_model *  text_encoder,
        struct imgen_params  params);

GGML_API void imgen_free(struct imgen_context * ctx);

GGML_API const char * imgen_arch_name(const struct imgen_context * ctx);

GGML_API int   imgen_default_steps(const struct imgen_context * ctx);
GGML_API float imgen_default_cfg_scale(const struct imgen_context * ctx);

// returns false on error (see log), out->data must be freed with imgen_image_free
GGML_API bool imgen_generate(
        struct imgen_context *     ctx,
        const struct imgen_request * req,
        struct imgen_image *       out,
        imgen_progress_callback    progress,
        void *                     progress_user_data);

GGML_API void imgen_image_free(struct imgen_image * img);

// encode RGB pixels as PNG, returns malloc'd buffer (free with free()), nullptr on error
GGML_API uint8_t * imgen_image_to_png(const struct imgen_image * img, size_t * n_bytes);

#ifdef __cplusplus
}
#endif
