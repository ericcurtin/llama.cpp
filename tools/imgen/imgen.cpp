#include "imgen-impl.h"

#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

//
// runner
//

imgen_runner::imgen_runner(const imgen_params & params, int max_nodes) : max_nodes(max_nodes) {
    n_threads  = params.n_threads;
    flash_attn = params.flash_attn;

    backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend_cpu) {
        throw std::runtime_error("imgen: failed to initialize CPU backend");
    }
    if (params.use_gpu) {
        if (params.device) {
            backend_gpu = ggml_backend_dev_init(params.device, nullptr);
        } else {
            backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
            if (!backend_gpu) {
                backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
            }
        }
    }
    if (backend_gpu) {
        backends.push_back(backend_gpu);
        bufts.push_back(ggml_backend_get_default_buffer_type(backend_gpu));
    }
    backends.push_back(backend_cpu);
    bufts.push_back(ggml_backend_get_default_buffer_type(backend_cpu));
    buft_weights = bufts[0];

    IMGEN_LOG("%s: using %s backend\n", __func__, ggml_backend_name(backends[0]));

    sched.reset(ggml_backend_sched_new(backends.data(), bufts.data(), backends.size(), max_nodes, false, true));

    if (getenv("IMGEN_DEBUG_NAN")) {
        // report the first node that produces a non-finite value
        ggml_backend_sched_set_eval_callback(sched.get(), [](ggml_tensor * t, bool ask, void *) {
            if (ask) {
                return true;
            }
            if ((t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) || !ggml_is_contiguous(t) || t->view_src || !t->buffer) {
                return true;
            }
            std::vector<uint8_t> buf(ggml_nbytes(t));
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
            std::vector<float> vals(ggml_nelements(t));
            if (t->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) buf.data(), vals.data(), vals.size());
            } else {
                memcpy(vals.data(), buf.data(), buf.size());
            }
            float vmax = 0;
            for (float v : vals) {
                if (!std::isfinite(v)) {
                    IMGEN_LOG("NaN in %s (%s) [%lld %lld %lld %lld] src0=%s src1=%s\n", t->name, ggml_op_desc(t),
                              (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                              t->src[0] ? t->src[0]->name : "-", t->src[1] ? t->src[1]->name : "-");
                    return false;
                }
                vmax = std::max(vmax, std::fabs(v));
            }
            if (vmax > 60000.0f) {
                IMGEN_LOG("large value %.1f in %s (%s)\n", vmax, t->name, ggml_op_desc(t));
            }
            return true;
        }, nullptr);
    }

    if (n_threads > 0) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend_cpu);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (fn) {
                fn(backend_cpu, n_threads);
            }
        }
    }

    buf_meta.resize(ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false));
}

imgen_runner::~imgen_runner() {
    sched.reset();
    if (backend_gpu) {
        ggml_backend_free(backend_gpu);
    }
    ggml_backend_free(backend_cpu);
}

ggml_context * imgen_runner::graph_begin() {
    ggml_init_params ip = { buf_meta.size(), buf_meta.data(), true };
    ctx0.reset(ggml_init(ip));
    gf = ggml_new_graph_custom(ctx0.get(), max_nodes, false);
    ggml_backend_sched_reset(sched.get());
    return ctx0.get();
}

// inputs must be set by the caller between alloc and compute, so this is split in two:
// graph_begin() -> build -> ggml_backend_sched_alloc_graph -> set inputs -> graph_compute()
void imgen_runner::graph_compute() {
    ggml_status st = ggml_backend_sched_graph_compute(sched.get(), gf);
    if (st != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("imgen: graph compute failed");
    }
}

bool imgen_runner::supports(const ggml_tensor * op) const {
    return ggml_backend_supports_op(backends[0], op);
}

//
// public API
//

imgen_params imgen_params_default() {
    imgen_params p;
    p.use_gpu    = true;
    p.device     = nullptr;
    p.n_threads  = 0;
    p.flash_attn = true;
    p.verbose    = false;
    return p;
}

static std::string gguf_arch(const char * path) {
    gguf_init_params gparams = { true, nullptr };
    gguf_context * gctx = gguf_init_from_file(path, gparams);
    if (!gctx) {
        return "";
    }
    std::string arch;
    const int64_t key = gguf_find_key(gctx, "general.architecture");
    if (key >= 0 && gguf_get_kv_type(gctx, key) == GGUF_TYPE_STRING) {
        arch = gguf_get_val_str(gctx, key);
    }
    gguf_free(gctx);
    return arch;
}

static bool arch_from_name(const std::string & name, imgen_arch & arch) {
    if (name == "qwen_image") {
        arch = IMGEN_ARCH_QWEN_IMAGE;
        return true;
    }
    if (name == "lumina2") {
        arch = IMGEN_ARCH_LUMINA2;
        return true;
    }
    return false;
}

bool imgen_is_dit_gguf(const char * path) {
    imgen_arch arch;
    return arch_from_name(gguf_arch(path), arch);
}

const char * imgen_arch_name(const imgen_context * ctx) {
    return ctx->arch == IMGEN_ARCH_QWEN_IMAGE ? "qwen_image" : "lumina2";
}

int imgen_default_steps(const imgen_context * ctx) {
    return ctx->arch == IMGEN_ARCH_QWEN_IMAGE ? 50 : 8;
}

float imgen_default_cfg_scale(const imgen_context * ctx) {
    return ctx->arch == IMGEN_ARCH_QWEN_IMAGE ? 4.0f : 1.0f;
}

static int count_layers(const imgen_weights & w, const std::string & prefix) {
    int n = 0;
    while (w.has(prefix + std::to_string(n) + ".attention_norm1.weight") ||
           w.has(prefix + std::to_string(n) + ".attn.to_q.weight")) {
        n++;
    }
    return n;
}

static void detect_hparams(imgen_context & ctx) {
    auto & hp = ctx.hp;
    auto & w  = ctx.dit;
    if (ctx.arch == IMGEN_ARCH_QWEN_IMAGE) {
        hp.n_layers = count_layers(w, "transformer_blocks.");
        hp.dim      = w.get("img_in.weight")->ne[1];
        hp.txt_dim  = w.get("txt_in.weight")->ne[0];
        hp.head_dim = w.get("transformer_blocks.0.attn.norm_q.weight")->ne[0];
        hp.n_heads  = hp.dim / hp.head_dim;
        hp.n_kv_heads = hp.n_heads;
        hp.axes[0] = 16; hp.axes[1] = 56; hp.axes[2] = 56;
        hp.theta = 10000.0f;
        hp.eps   = 1e-6f;
    } else {
        hp.n_layers  = count_layers(w, "layers.");
        hp.n_refiner = count_layers(w, "noise_refiner.");
        hp.dim       = w.get("x_embedder.weight")->ne[1];
        hp.txt_dim   = w.get("cap_embedder.1.weight")->ne[0];
        hp.head_dim  = w.get("layers.0.attention.q_norm.weight")->ne[0];
        hp.n_heads   = hp.dim / hp.head_dim;
        hp.n_kv_heads = (w.get("layers.0.attention.qkv.weight")->ne[1] / hp.head_dim - hp.n_heads) / 2;
        hp.axes[0] = 32; hp.axes[1] = 48; hp.axes[2] = 48;
        hp.theta = 256.0f;
        hp.eps   = 1e-5f;
    }
    if (hp.axes[0] + hp.axes[1] + hp.axes[2] != hp.head_dim) {
        throw std::runtime_error("imgen: unexpected head_dim " + std::to_string(hp.head_dim));
    }
    if (hp.n_layers == 0) {
        throw std::runtime_error("imgen: no transformer layers found in DiT");
    }
}

imgen_context * imgen_init(const char * dit_path, const char * vae_path, const llama_model * text_encoder, imgen_params params) {
    try {
        auto ctx = std::make_unique<imgen_context>();
        ctx->params = params;

        if (!arch_from_name(gguf_arch(dit_path), ctx->arch)) {
            IMGEN_LOG("%s: '%s' is not a supported diffusion transformer\n", __func__, dit_path);
            return nullptr;
        }

        ctx->runner = std::make_unique<imgen_runner>(params, 32768);

        const int64_t t0 = ggml_time_ms();
        imgen_load_weights(dit_path, ctx->runner->buft_weights, ctx->dit, true);
        detect_hparams(*ctx);
        IMGEN_LOG("%s: loaded %s DiT: %d layers, dim %d, %d heads, %.2f GiB (%lld ms)\n", __func__,
                  imgen_arch_name(ctx.get()), ctx->hp.n_layers, ctx->hp.dim, ctx->hp.n_heads,
                  ggml_backend_buffer_get_size(ctx->dit.buf.get()) / 1024.0 / 1024.0 / 1024.0,
                  (long long) (ggml_time_ms() - t0));

        imgen_load_weights(vae_path, ctx->runner->buft_weights, ctx->vae, false);
        if (ctx->vae.has("decoder.mid.block_1.conv1.weight")) {
            ctx->vae_type = IMGEN_VAE_FLUX;
        } else if (ctx->vae.has("decoder.middle.0.residual.2.weight")) {
            ctx->vae_type = IMGEN_VAE_WAN;
        } else {
            IMGEN_LOG("%s: unrecognized VAE '%s' (expected Flux ae or Wan/Qwen-Image vae)\n", __func__, vae_path);
            return nullptr;
        }
        const imgen_vae_type want = ctx->arch == IMGEN_ARCH_QWEN_IMAGE ? IMGEN_VAE_WAN : IMGEN_VAE_FLUX;
        if (ctx->vae_type != want) {
            IMGEN_LOG("%s: VAE '%s' does not match the DiT architecture\n", __func__, vae_path);
            return nullptr;
        }
        IMGEN_LOG("%s: loaded %s VAE\n", __func__, ctx->vae_type == IMGEN_VAE_FLUX ? "flux" : "wan");

        if (!text_encoder) {
            IMGEN_LOG("%s: text encoder model is required\n", __func__);
            return nullptr;
        }
        ctx->te.model = text_encoder;
        if (llama_model_n_embd(text_encoder) != ctx->hp.txt_dim) {
            IMGEN_LOG("%s: text encoder hidden size %d does not match DiT text dim %d (expected %s)\n", __func__,
                      llama_model_n_embd(text_encoder), ctx->hp.txt_dim,
                      ctx->arch == IMGEN_ARCH_QWEN_IMAGE ? "Qwen2.5-VL-7B" : "Qwen3-4B");
            return nullptr;
        }

        return ctx.release();
    } catch (const std::exception & e) {
        IMGEN_LOG("%s: %s\n", __func__, e.what());
        return nullptr;
    }
}

void imgen_free(imgen_context * ctx) {
    delete ctx;
}

void imgen_image_free(imgen_image * img) {
    if (img) {
        free(img->data);
        img->data = nullptr;
    }
}

//
// sampling
//

static void debug_stats(const imgen_context & ctx, const char * name, const std::vector<float> & v) {
    if (!ctx.params.verbose) {
        return;
    }
    double sum = 0, sum2 = 0;
    float vmin = INFINITY, vmax = -INFINITY;
    size_t n_nan = 0;
    for (float x : v) {
        if (std::isnan(x) || std::isinf(x)) {
            n_nan++;
            continue;
        }
        sum += x; sum2 += (double) x * x;
        vmin = std::min(vmin, x); vmax = std::max(vmax, x);
    }
    const double mean = sum / std::max<size_t>(1, v.size() - n_nan);
    IMGEN_LOG("%s: %s n=%zu nan=%zu min=%.4f max=%.4f mean=%.4f rms=%.4f\n", __func__, name, v.size(), n_nan, vmin, vmax, mean,
              std::sqrt(sum2 / std::max<size_t>(1, v.size() - n_nan)));
}

// FlowMatchEulerDiscreteScheduler: linspace(1, 1/n, n) sigmas with shift, terminal 0
static std::vector<float> make_sigmas(const imgen_context & ctx, int n_steps, int n_img_tokens) {
    std::vector<float> sigmas(n_steps + 1, 0.0f);
    for (int i = 0; i < n_steps; i++) {
        sigmas[i] = 1.0f - (float) i * (1.0f - 1.0f / n_steps) / std::max(1, n_steps - 1);
    }
    if (ctx.arch == IMGEN_ARCH_QWEN_IMAGE) {
        // dynamic exponential shift, mu from image sequence length
        const float base_len = 256, max_len = 8192, base_shift = 0.5f, max_shift = 0.9f;
        const float m  = (max_shift - base_shift) / (max_len - base_len);
        const float mu = n_img_tokens * m + (base_shift - m * base_len);
        for (int i = 0; i < n_steps; i++) {
            sigmas[i] = std::exp(mu) / (std::exp(mu) + (1.0f / sigmas[i] - 1.0f));
        }
        // stretch so the last sigma lands on shift_terminal
        const float shift_terminal = 0.02f;
        const float scale = (1.0f - sigmas[n_steps - 1]) / (1.0f - shift_terminal);
        for (int i = 0; i < n_steps; i++) {
            sigmas[i] = 1.0f - (1.0f - sigmas[i]) / scale;
        }
    } else {
        const float shift = 3.0f;
        for (int i = 0; i < n_steps; i++) {
            sigmas[i] = shift * sigmas[i] / (1.0f + (shift - 1.0f) * sigmas[i]);
        }
    }
    return sigmas;
}

// packed DiT latent [in_ch*4 x hp*wp] -> VAE latent [w x h x in_ch]
static void unpack_latent(const imgen_context & ctx, int hp, int wp, const std::vector<float> & x, std::vector<float> & z) {
    const int c_lat = ctx.hp.in_ch;
    const int w = wp * 2, h = hp * 2;
    z.assign((size_t) w * h * c_lat, 0.0f);
    for (int i = 0; i < hp; i++) {
        for (int j = 0; j < wp; j++) {
            const float * tok = x.data() + (size_t) (i * wp + j) * c_lat * 4;
            for (int c = 0; c < c_lat; c++) {
                for (int ph = 0; ph < 2; ph++) {
                    for (int pw = 0; pw < 2; pw++) {
                        // qwen_image packs (c, ph, pw), lumina2 packs (ph, pw, c)
                        const int f = ctx.arch == IMGEN_ARCH_QWEN_IMAGE ? c*4 + ph*2 + pw : (ph*2 + pw)*c_lat + c;
                        z[(size_t) c * w * h + (size_t) (2*i + ph) * w + (2*j + pw)] = tok[f];
                    }
                }
            }
        }
    }
}

bool imgen_generate(imgen_context * ctx, const imgen_request * req, imgen_image * out, imgen_progress_callback progress, void * progress_ud) {
    try {
        int width  = req->width  > 0 ? req->width  : 1024;
        int height = req->height > 0 ? req->height : 1024;
        width  = std::max(16, width  / 16 * 16);
        height = std::max(16, height / 16 * 16);
        const int wp = width / 16, hp = height / 16;
        const int n_img = wp * hp;
        const int steps = req->steps > 0 ? req->steps : imgen_default_steps(ctx);
        const float cfg = req->cfg_scale >= 0.0f ? req->cfg_scale : imgen_default_cfg_scale(ctx);
        const bool do_cfg = cfg > 1.0f;

        int64_t t0 = ggml_time_ms();
        imgen_cond cond, uncond;
        if (!imgen_encode_prompt(*ctx, req->prompt ? req->prompt : "", cond)) {
            return false;
        }
        if (do_cfg && !imgen_encode_prompt(*ctx, req->negative_prompt ? req->negative_prompt : "", uncond)) {
            return false;
        }
        IMGEN_LOG("%s: encoded prompt: %d tokens (%lld ms)\n", __func__, cond.n_tokens, (long long) (ggml_time_ms() - t0));
        debug_stats(*ctx, "cond", cond.embd);

        std::mt19937_64 rng(req->seed >= 0 ? (uint64_t) req->seed : std::random_device{}());
        std::normal_distribution<float> normal(0.0f, 1.0f);
        const size_t n_lat = (size_t) n_img * ctx->hp.in_ch * 4;
        std::vector<float> x(n_lat);
        for (auto & v : x) {
            v = normal(rng);
        }

        const std::vector<float> sigmas = make_sigmas(*ctx, steps, n_img);
        std::vector<float> v_pos, v_neg;

        for (int i = 0; i < steps; i++) {
            t0 = ggml_time_ms();
            const float sigma = sigmas[i];
            // lumina2 is trained on t = 1 - sigma and predicts -velocity
            const float t = ctx->arch == IMGEN_ARCH_QWEN_IMAGE ? sigma : 1.0f - sigma;
            imgen_dit_forward(*ctx, cond, hp, wp, t, x, v_pos);
            debug_stats(*ctx, "dit_out", v_pos);
            if (do_cfg) {
                imgen_dit_forward(*ctx, uncond, hp, wp, t, x, v_neg);
                if (ctx->arch == IMGEN_ARCH_QWEN_IMAGE) {
                    // true CFG with per-token norm rescale
                    for (int tok = 0; tok < n_img; tok++) {
                        float * p = v_pos.data() + (size_t) tok * ctx->hp.in_ch * 4;
                        float * n = v_neg.data() + (size_t) tok * ctx->hp.in_ch * 4;
                        double norm_p = 0, norm_c = 0;
                        for (int k = 0; k < ctx->hp.in_ch * 4; k++) {
                            const float comb = n[k] + cfg * (p[k] - n[k]);
                            norm_p += (double) p[k] * p[k];
                            norm_c += (double) comb * comb;
                            n[k] = comb;
                        }
                        const float r = norm_c > 0 ? (float) std::sqrt(norm_p / norm_c) : 1.0f;
                        for (int k = 0; k < ctx->hp.in_ch * 4; k++) {
                            p[k] = n[k] * r;
                        }
                    }
                } else {
                    for (size_t k = 0; k < n_lat; k++) {
                        v_pos[k] = v_pos[k] + cfg * (v_pos[k] - v_neg[k]);
                    }
                }
            }
            const float sign = ctx->arch == IMGEN_ARCH_QWEN_IMAGE ? 1.0f : -1.0f;
            const float dt = sigmas[i + 1] - sigma;
            for (size_t k = 0; k < n_lat; k++) {
                x[k] += dt * sign * v_pos[k];
            }
            IMGEN_LOG("%s: step %d/%d, sigma %.4f (%lld ms)\n", __func__, i + 1, steps, sigma, (long long) (ggml_time_ms() - t0));
            if (progress && !progress(i + 1, steps, progress_ud)) {
                IMGEN_LOG("%s: cancelled\n", __func__);
                return false;
            }
        }

        t0 = ggml_time_ms();
        std::vector<float> z, rgb;
        unpack_latent(*ctx, hp, wp, x, z);
        debug_stats(*ctx, "latent", z);
        imgen_vae_decode(*ctx, wp * 2, hp * 2, z, rgb);
        debug_stats(*ctx, "rgb", rgb);
        IMGEN_LOG("%s: VAE decode (%lld ms)\n", __func__, (long long) (ggml_time_ms() - t0));

        out->width  = width;
        out->height = height;
        out->data   = (uint8_t *) malloc((size_t) width * height * 3);
        for (int y = 0; y < height; y++) {
            for (int xx = 0; xx < width; xx++) {
                for (int c = 0; c < 3; c++) {
                    const float v = rgb[(size_t) c * width * height + (size_t) y * width + xx];
                    out->data[((size_t) y * width + xx) * 3 + c] = (uint8_t) std::lround(std::min(1.0f, std::max(0.0f, v)) * 255.0f);
                }
            }
        }
        return true;
    } catch (const std::exception & e) {
        IMGEN_LOG("%s: %s\n", __func__, e.what());
        return false;
    }
}

//
// PNG writer (stored deflate blocks, no compression)
//

static uint32_t crc32_update(uint32_t crc, const uint8_t * data, size_t n) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) {
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            }
            table[i] = c;
        }
        init = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < n; i++) {
        crc = table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    }
    return ~crc;
}

static void put_be32(std::vector<uint8_t> & v, uint32_t x) {
    v.push_back(x >> 24); v.push_back(x >> 16); v.push_back(x >> 8); v.push_back(x);
}

static void put_chunk(std::vector<uint8_t> & png, const char * type, const std::vector<uint8_t> & data) {
    put_be32(png, data.size());
    const size_t start = png.size();
    png.insert(png.end(), type, type + 4);
    png.insert(png.end(), data.begin(), data.end());
    put_be32(png, crc32_update(0, png.data() + start, png.size() - start));
}

uint8_t * imgen_image_to_png(const imgen_image * img, size_t * n_bytes) {
    const size_t row = (size_t) img->width * 3 + 1;
    std::vector<uint8_t> raw(row * img->height);
    for (int y = 0; y < img->height; y++) {
        raw[y * row] = 0; // filter: none
        memcpy(raw.data() + y * row + 1, img->data + (size_t) y * img->width * 3, row - 1);
    }

    std::vector<uint8_t> idat = { 0x78, 0x01 }; // zlib header, no compression
    uint32_t a = 1, b = 0;
    for (uint8_t v : raw) {
        a = (a + v) % 65521;
        b = (b + a) % 65521;
    }
    for (size_t off = 0; off < raw.size(); off += 65535) {
        const uint16_t len = (uint16_t) std::min<size_t>(65535, raw.size() - off);
        idat.push_back(off + len >= raw.size() ? 1 : 0);
        idat.push_back(len & 0xff); idat.push_back(len >> 8);
        idat.push_back(~len & 0xff); idat.push_back((~len >> 8) & 0xff);
        idat.insert(idat.end(), raw.begin() + off, raw.begin() + off + len);
    }
    put_be32(idat, (b << 16) | a);

    std::vector<uint8_t> ihdr;
    put_be32(ihdr, img->width);
    put_be32(ihdr, img->height);
    ihdr.insert(ihdr.end(), { 8, 2, 0, 0, 0 }); // 8-bit RGB

    std::vector<uint8_t> png = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };
    put_chunk(png, "IHDR", ihdr);
    put_chunk(png, "IDAT", idat);
    put_chunk(png, "IEND", {});

    uint8_t * buf = (uint8_t *) malloc(png.size());
    if (!buf) {
        return nullptr;
    }
    memcpy(buf, png.data(), png.size());
    *n_bytes = png.size();
    return buf;
}
