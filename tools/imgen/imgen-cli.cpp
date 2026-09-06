#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "imgen.h"

#include <cstdio>
#include <string>

static bool on_progress(int step, int n_steps, void *) {
    LOG("\rstep %d/%d", step, n_steps);
    if (step == n_steps) {
        LOG("\n");
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMGEN)) {
        return 1;
    }
    common_init();

    if (params.prompt.empty()) {
        LOG_ERR("error: a prompt is required (-p)\n");
        return 1;
    }
    if (params.imgen.text_encoder.path.empty() || params.imgen.vae.path.empty()) {
        LOG_ERR("error: --text-encoder and --vae are required (or use -hf with a supported repo)\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * text_encoder = llama_model_load_from_file(params.imgen.text_encoder.path.c_str(), mparams);
    if (!text_encoder) {
        LOG_ERR("error: failed to load text encoder '%s'\n", params.imgen.text_encoder.path.c_str());
        return 1;
    }

    imgen_params iparams = imgen_params_default();
    iparams.n_threads  = params.cpuparams.n_threads;
    iparams.use_gpu    = params.n_gpu_layers != 0;
    iparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    iparams.verbose    = params.verbosity > 0;

    imgen_context * ctx = imgen_init(params.model.path.c_str(), params.imgen.vae.path.c_str(), text_encoder, iparams);
    if (!ctx) {
        LOG_ERR("error: failed to initialize image generation\n");
        return 1;
    }

    imgen_request req;
    req.prompt          = params.prompt.c_str();
    req.negative_prompt = params.imgen.negative_prompt.c_str();
    req.width           = params.imgen.width;
    req.height          = params.imgen.height;
    req.steps           = params.imgen.steps;
    req.cfg_scale       = params.imgen.cfg_scale;
    req.seed            = params.sampling.seed == LLAMA_DEFAULT_SEED ? -1 : (int64_t) params.sampling.seed;

    const int64_t t0 = ggml_time_ms();
    imgen_image img;
    if (!imgen_generate(ctx, &req, &img, on_progress, nullptr)) {
        LOG_ERR("error: generation failed\n");
        return 1;
    }

    size_t n_png = 0;
    uint8_t * png = imgen_image_to_png(&img, &n_png);
    FILE * f = png ? fopen(params.out_file.c_str(), "wb") : nullptr;
    if (!f || fwrite(png, 1, n_png, f) != n_png) {
        LOG_ERR("error: failed to write '%s'\n", params.out_file.c_str());
        return 1;
    }
    fclose(f);
    free(png);
    LOG_INF("wrote %dx%d image to '%s' (%.1f s)\n", img.width, img.height, params.out_file.c_str(), (ggml_time_ms() - t0) / 1000.0);

    imgen_image_free(&img);
    imgen_free(ctx);
    llama_model_free(text_encoder);
    llama_backend_free();
    return 0;
}
