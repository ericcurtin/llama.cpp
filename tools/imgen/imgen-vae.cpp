#include "imgen-impl.h"

#include <cmath>

namespace {

// tensors are [W, H, C, 1]
struct vae_graph {
    imgen_context & ctx;
    ggml_context *  g;

    vae_graph(imgen_context & ctx, ggml_context * g) : ctx(ctx), g(g) {}

    ggml_tensor * w(const std::string & name) { return ctx.vae.get(name); }
    bool has(const std::string & name) { return ctx.vae.has(name); }

    ggml_tensor * per_channel(ggml_tensor * v) {
        return ggml_reshape_4d(g, v, 1, 1, v->ne[0], 1);
    }

    ggml_tensor * conv(ggml_tensor * x, const std::string & name) {
        ggml_tensor * k = w(name + ".weight");
        const int pad = k->ne[0] / 2;
        ggml_tensor * y = ctx.runner->conv_im2col ? ggml_conv_2d(g, k, x, 1, 1, pad, pad, 1, 1)
                                                  : ggml_conv_2d_direct(g, k, x, 1, 1, pad, pad, 1, 1);
        if (ggml_tensor * b = ctx.vae.get(name + ".bias", false)) {
            y = ggml_add(g, y, per_channel(b));
        }
        return y;
    }

    ggml_tensor * upsample(ggml_tensor * x) {
        return ggml_upscale(g, x, 2, GGML_SCALE_MODE_NEAREST);
    }

    // single head attention over all spatial positions, q/k/v [W, H, C, 1]
    ggml_tensor * spatial_attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) {
        const int64_t W = q->ne[0], H = q->ne[1], C = q->ne[2];
        const int64_t n = W * H;
        const float scale = 1.0f / std::sqrt((float) C);

        ggml_tensor * qt = ggml_cont(g, ggml_transpose(g, ggml_reshape_2d(g, ggml_cont(g, q), n, C))); // [C, n]
        ggml_tensor * kt = ggml_cont(g, ggml_transpose(g, ggml_reshape_2d(g, ggml_cont(g, k), n, C)));
        ggml_tensor * vv = ggml_reshape_2d(g, ggml_cont(g, v), n, C);                                  // [n, C]

        // limit the [n, chunk] score matrix to ~256 MiB
        const int64_t chunk = std::max<int64_t>(1, std::min<int64_t>(n, (64ll << 20) / n));
        ggml_tensor * out = nullptr;
        for (int64_t off = 0; off < n; off += chunk) {
            const int64_t len = std::min(chunk, n - off);
            ggml_tensor * qc = ggml_view_2d(g, qt, C, len, qt->nb[1], off * qt->nb[1]);
            ggml_tensor * kq = ggml_mul_mat(g, kt, qc); // [n, len], raw scores reach 1e8
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(g, kq, nullptr, scale, 0.0f);
            ggml_tensor * oc = ggml_mul_mat(g, vv, kq); // [C, len]
            ggml_mul_mat_set_prec(oc, GGML_PREC_F32);
            out = out ? ggml_concat(g, out, oc, 1) : oc;
        }
        out = ggml_cont(g, ggml_transpose(g, out)); // [n, C]
        return ggml_reshape_4d(g, out, W, H, C, 1);
    }
};

struct flux_vae_graph : vae_graph {
    using vae_graph::vae_graph;

    ggml_tensor * norm(ggml_tensor * x, const std::string & name) {
        x = ggml_group_norm(g, x, 32, 1e-6f);
        x = ggml_mul(g, x, per_channel(w(name + ".weight")));
        return ggml_add(g, x, per_channel(w(name + ".bias")));
    }

    ggml_tensor * resnet(ggml_tensor * x, const std::string & pfx) {
        ggml_tensor * h = x;
        if (has(pfx + "nin_shortcut.weight")) {
            h = conv(x, pfx + "nin_shortcut");
        }
        x = conv(ggml_silu(g, norm(x, pfx + "norm1")), pfx + "conv1");
        x = conv(ggml_silu(g, norm(x, pfx + "norm2")), pfx + "conv2");
        return ggml_add(g, x, h);
    }

    ggml_tensor * attn(ggml_tensor * x, const std::string & pfx) {
        ggml_tensor * h = norm(x, pfx + "norm");
        h = spatial_attention(conv(h, pfx + "q"), conv(h, pfx + "k"), conv(h, pfx + "v"));
        return ggml_add(g, x, conv(h, pfx + "proj_out"));
    }

    ggml_tensor * build(ggml_tensor * z) {
        ggml_tensor * x = conv(z, "decoder.conv_in");
        x = resnet(x, "decoder.mid.block_1.");
        x = attn(x, "decoder.mid.attn_1.");
        x = resnet(x, "decoder.mid.block_2.");
        for (int i = 3; i >= 0; i--) {
            const std::string pfx = "decoder.up." + std::to_string(i) + ".";
            for (int b = 0; has(pfx + "block." + std::to_string(b) + ".conv1.weight"); b++) {
                x = resnet(x, pfx + "block." + std::to_string(b) + ".");
            }
            if (has(pfx + "upsample.conv.weight")) {
                x = conv(upsample(x), pfx + "upsample.conv");
            }
        }
        x = ggml_silu(g, norm(x, "decoder.norm_out"));
        return conv(x, "decoder.conv_out");
    }
};

// Wan2.1 VAE decoder restricted to a single frame: every causal 3D conv reduces to a 2D conv
struct wan_vae_graph : vae_graph {
    using vae_graph::vae_graph;

    // RMS norm over channels
    ggml_tensor * norm(ggml_tensor * x, const std::string & name) {
        ggml_tensor * xt = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3)); // [C, W, H]
        xt = ggml_mul(g, ggml_rms_norm(g, xt, 1e-12f), w(name + ".gamma"));
        return ggml_cont(g, ggml_permute(g, xt, 2, 0, 1, 3));
    }

    ggml_tensor * resnet(ggml_tensor * x, const std::string & pfx) {
        ggml_tensor * h = x;
        if (has(pfx + "shortcut.weight")) {
            h = conv(x, pfx + "shortcut");
        }
        x = conv(ggml_silu(g, norm(x, pfx + "residual.0")), pfx + "residual.2");
        x = conv(ggml_silu(g, norm(x, pfx + "residual.3")), pfx + "residual.6");
        return ggml_add(g, x, h);
    }

    ggml_tensor * attn(ggml_tensor * x, const std::string & pfx) {
        ggml_tensor * qkv = conv(norm(x, pfx + "norm"), pfx + "to_qkv");
        const int64_t C = x->ne[2];
        auto part = [&](int i) {
            return ggml_view_4d(g, qkv, qkv->ne[0], qkv->ne[1], C, 1, qkv->nb[1], qkv->nb[2], qkv->nb[3], i * C * qkv->nb[2]);
        };
        ggml_tensor * h = spatial_attention(part(0), part(1), part(2));
        return ggml_add(g, x, conv(h, pfx + "proj"));
    }

    ggml_tensor * build(ggml_tensor * z) {
        ggml_tensor * x = conv(z, "conv2"); // post_quant_conv
        x = conv(x, "decoder.conv1");
        x = resnet(x, "decoder.middle.0.");
        x = attn(x, "decoder.middle.1.");
        x = resnet(x, "decoder.middle.2.");
        for (int i = 0; ; i++) {
            const std::string pfx = "decoder.upsamples." + std::to_string(i) + ".";
            if (has(pfx + "resample.1.weight")) {
                x = conv(upsample(x), pfx + "resample.1");
            } else if (has(pfx + "residual.2.weight")) {
                x = resnet(x, pfx);
            } else {
                break;
            }
        }
        x = ggml_silu(g, norm(x, "decoder.head.0"));
        return conv(x, "decoder.head.2");
    }
};

const float WAN_LATENT_MEAN[16] = { -0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                                     0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f };
const float WAN_LATENT_STD[16]  = { 2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                                     3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f };

} // namespace

void imgen_vae_decode(imgen_context & ctx, int wl, int hl, const std::vector<float> & z_in, std::vector<float> & rgb) {
    const int c_lat = ctx.hp.in_ch;
    std::vector<float> z(z_in.size());
    for (int c = 0; c < c_lat; c++) {
        const float * src = z_in.data() + (size_t) c * wl * hl;
        float * dst = z.data() + (size_t) c * wl * hl;
        for (size_t i = 0; i < (size_t) wl * hl; i++) {
            dst[i] = ctx.vae_type == IMGEN_VAE_FLUX ? src[i] / 0.3611f + 0.1159f : src[i] * WAN_LATENT_STD[c] + WAN_LATENT_MEAN[c];
        }
    }

    imgen_runner & runner = *ctx.runner;
    ggml_context * g = runner.graph_begin();

    ggml_tensor * inp = ggml_new_tensor_4d(g, GGML_TYPE_F32, wl, hl, c_lat, 1);
    ggml_set_name(inp, "z");
    ggml_set_input(inp);

    ggml_tensor * out;
    if (ctx.vae_type == IMGEN_VAE_FLUX) {
        out = flux_vae_graph(ctx, g).build(inp);
    } else {
        out = wan_vae_graph(ctx, g).build(inp);
    }
    ggml_set_output(out);
    ggml_build_forward_expand(runner.gf, out);

    if (!ggml_backend_sched_alloc_graph(runner.sched.get(), runner.gf)) {
        throw std::runtime_error("imgen: failed to allocate VAE graph");
    }
    ggml_backend_tensor_set(inp, z.data(), 0, ggml_nbytes(inp));
    runner.graph_compute();

    rgb.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, rgb.data(), 0, ggml_nbytes(out));
    for (auto & v : rgb) {
        v = (v + 1.0f) * 0.5f;
    }
}
