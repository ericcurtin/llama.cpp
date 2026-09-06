#include "imgen-impl.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace {

// the FFN hidden activations of these models reach 1e5, which overflows the F16 that
// several backends use for matmul inputs; the down projection input is pre-scaled by this
constexpr float FFN_DOWN_SCALE = 64.0f;

struct input_data {
    ggml_tensor * t;
    const void *  data;
};

struct dit_graph {
    imgen_context &           ctx;
    const imgen_dit_hparams & hp;
    ggml_context *            g;
    std::vector<input_data>   inputs;

    dit_graph(imgen_context & ctx, ggml_context * g) : ctx(ctx), hp(ctx.hp), g(g) {}

    ggml_tensor * input(const char * name, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, const void * data) {
        ggml_tensor * t = ggml_new_tensor_4d(g, type, ne0, ne1, ne2, ne3);
        ggml_set_name(t, name);
        ggml_set_input(t);
        inputs.push_back({ t, data });
        return t;
    }

    ggml_tensor * w(const std::string & name) { return ctx.dit.get(name); }
    ggml_tensor * w_opt(const std::string & name) { return ctx.dit.get(name, false); }

    // scale > 1 divides the input and multiplies the output back, for activations beyond the F16 range
    ggml_tensor * linear(ggml_tensor * x, const std::string & name, float scale = 1.0f) {
        if (scale != 1.0f) {
            x = ggml_scale(g, x, 1.0f / scale);
        }
        ggml_tensor * y = ggml_mul_mat(g, w(name + ".weight"), x);
        ggml_set_name(y, name.c_str());
        // outputs reach 1e5, F16 accumulation would saturate
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        if (scale != 1.0f) {
            y = ggml_scale(g, y, scale);
        }
        if (ggml_tensor * b = w_opt(name + ".bias")) {
            y = ggml_add(g, y, b);
        }
        return y;
    }

    ggml_tensor * rms_norm(ggml_tensor * x, ggml_tensor * weight, float eps) {
        x = ggml_rms_norm(g, x, eps);
        return weight ? ggml_mul(g, x, weight) : x;
    }

    // x * (1 + scale) + shift, scale/shift are [dim, 1]
    ggml_tensor * modulate(ggml_tensor * x, ggml_tensor * scale, ggml_tensor * shift) {
        x = ggml_add(g, x, ggml_mul(g, x, scale));
        return shift ? ggml_add(g, x, shift) : x;
    }

    ggml_tensor * chunk(ggml_tensor * v, int i, int64_t n) {
        return ggml_view_2d(g, v, n, 1, v->nb[1], i * n * ggml_element_size(v));
    }

    // rotate adjacent pairs of x [hd, nh, L] with cos/sin [1, hd/2, 1, L]
    ggml_tensor * rope(ggml_tensor * x, ggml_tensor * cos, ggml_tensor * sin) {
        const int64_t hd = x->ne[0], nh = x->ne[1], L = x->ne[2];
        ggml_tensor * x4 = ggml_reshape_4d(g, x, 2, hd/2, nh, L);
        ggml_tensor * x0 = ggml_cont(g, ggml_view_4d(g, x4, 1, hd/2, nh, L, x4->nb[1], x4->nb[2], x4->nb[3], 0));
        ggml_tensor * x1 = ggml_cont(g, ggml_view_4d(g, x4, 1, hd/2, nh, L, x4->nb[1], x4->nb[2], x4->nb[3], ggml_element_size(x4)));
        ggml_tensor * o0 = ggml_sub(g, ggml_mul(g, x0, cos), ggml_mul(g, x1, sin));
        ggml_tensor * o1 = ggml_add(g, ggml_mul(g, x0, sin), ggml_mul(g, x1, cos));
        return ggml_reshape_3d(g, ggml_concat(g, o0, o1, 0), hd, nh, L);
    }

    // q [hd, nh, L], k/v [hd, nkv, L] -> [hd*nh, L]
    ggml_tensor * attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) {
        const int64_t hd = q->ne[0], nh = q->ne[1], L = q->ne[2];
        const float scale = 1.0f / std::sqrt((float) hd);
        ggml_tensor * out;
        if (ctx.runner->flash_attn) {
            k = ggml_cast(g, k, GGML_TYPE_F16);
            v = ggml_cast(g, v, GGML_TYPE_F16);
            q = ggml_permute(g, q, 0, 2, 1, 3);
            k = ggml_permute(g, k, 0, 2, 1, 3);
            v = ggml_permute(g, v, 0, 2, 1, 3);
            out = ggml_flash_attn_ext(g, q, k, v, nullptr, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        } else {
            q = ggml_permute(g, q, 0, 2, 1, 3); // [hd, L, nh]
            k = ggml_permute(g, k, 0, 2, 1, 3);
            ggml_tensor * kq = ggml_mul_mat(g, k, q); // [Lk, Lq, nh]
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(g, kq, nullptr, scale, 0.0f);
            ggml_tensor * vt = ggml_cont(g, ggml_permute(g, v, 1, 2, 0, 3)); // [L, hd, nkv]
            out = ggml_mul_mat(g, vt, kq); // [hd, Lq, nh]
            ggml_mul_mat_set_prec(out, GGML_PREC_F32);
            out = ggml_cont(g, ggml_permute(g, out, 0, 2, 1, 3)); // [hd, nh, L]
        }
        return ggml_reshape_2d(g, out, hd * nh, L);
    }

    ggml_tensor * timestep_embedding(ggml_tensor * t, int dim) {
        return ggml_timestep_embedding(g, t, dim, 10000);
    }
};

// cos/sin for 3-axis rope, positions [n][3], tables [hd/2][n] (pair major, token fastest) transposed to [n][hd/2]
void rope_tables(const imgen_dit_hparams & hp, const std::vector<std::array<int, 3>> & pos, std::vector<float> & cos_t, std::vector<float> & sin_t) {
    const int n_pairs = hp.head_dim / 2;
    cos_t.resize((size_t) pos.size() * n_pairs);
    sin_t.resize((size_t) pos.size() * n_pairs);
    for (size_t i = 0; i < pos.size(); i++) {
        int p = 0;
        for (int a = 0; a < 3; a++) {
            const int d = hp.axes[a];
            for (int j = 0; j < d / 2; j++, p++) {
                const double freq  = 1.0 / std::pow((double) hp.theta, (2.0 * j) / d);
                const double angle = pos[i][a] * freq;
                cos_t[i * n_pairs + p] = (float) std::cos(angle);
                sin_t[i * n_pairs + p] = (float) std::sin(angle);
            }
        }
    }
}

//
// Qwen-Image: dual stream MMDiT, attention over [txt, img]
//

struct qwen_image_graph : dit_graph {
    using dit_graph::dit_graph;

    ggml_tensor * build(const imgen_cond & cond, int hp_, int wp, const float & t_scaled, const std::vector<float> & x_in,
                        const std::vector<float> & cos_t, const std::vector<float> & sin_t) {
        const int n_img = hp_ * wp;
        const int n_txt = cond.n_tokens;
        const int n_all = n_txt + n_img;
        const int dim   = hp.dim;

        ggml_tensor * x   = input("x",   GGML_TYPE_F32, hp.in_ch * 4, n_img, 1, 1, x_in.data());
        ggml_tensor * txt = input("txt", GGML_TYPE_F32, hp.txt_dim, n_txt, 1, 1, cond.embd.data());
        ggml_tensor * t   = input("t",   GGML_TYPE_F32, 1, 1, 1, 1, &t_scaled);
        ggml_tensor * cos = input("cos", GGML_TYPE_F32, 1, hp.head_dim / 2, 1, n_all, cos_t.data());
        ggml_tensor * sin = input("sin", GGML_TYPE_F32, 1, hp.head_dim / 2, 1, n_all, sin_t.data());

        ggml_tensor * img = linear(x, "img_in");
        txt = rms_norm(txt, w("txt_norm.weight"), hp.eps);
        txt = linear(txt, "txt_in");

        ggml_tensor * temb = timestep_embedding(t, 256);
        temb = linear(temb, "time_text_embed.timestep_embedder.linear_1");
        temb = ggml_silu(g, temb);
        temb = linear(temb, "time_text_embed.timestep_embedder.linear_2");
        ggml_tensor * temb_act = ggml_silu(g, temb);

        for (int il = 0; il < hp.n_layers; il++) {
            const std::string pfx = "transformer_blocks." + std::to_string(il) + ".";

            ggml_tensor * img_mod = linear(temb_act, pfx + "img_mod.1");
            ggml_tensor * txt_mod = linear(temb_act, pfx + "txt_mod.1");

            ggml_tensor * img_m = modulate(ggml_norm(g, img, hp.eps), chunk(img_mod, 1, dim), chunk(img_mod, 0, dim));
            ggml_tensor * txt_m = modulate(ggml_norm(g, txt, hp.eps), chunk(txt_mod, 1, dim), chunk(txt_mod, 0, dim));

            auto heads = [&](ggml_tensor * v, int n) { return ggml_reshape_3d(g, v, hp.head_dim, hp.n_heads, n); };

            ggml_tensor * q  = rms_norm(heads(linear(img_m, pfx + "attn.to_q"), n_img), w(pfx + "attn.norm_q.weight"), hp.eps);
            ggml_tensor * k  = rms_norm(heads(linear(img_m, pfx + "attn.to_k"), n_img), w(pfx + "attn.norm_k.weight"), hp.eps);
            ggml_tensor * v  = heads(linear(img_m, pfx + "attn.to_v"), n_img);
            ggml_tensor * tq = rms_norm(heads(linear(txt_m, pfx + "attn.add_q_proj"), n_txt), w(pfx + "attn.norm_added_q.weight"), hp.eps);
            ggml_tensor * tk = rms_norm(heads(linear(txt_m, pfx + "attn.add_k_proj"), n_txt), w(pfx + "attn.norm_added_k.weight"), hp.eps);
            ggml_tensor * tv = heads(linear(txt_m, pfx + "attn.add_v_proj"), n_txt);

            q = rope(ggml_concat(g, tq, q, 2), cos, sin);
            k = rope(ggml_concat(g, tk, k, 2), cos, sin);
            v = ggml_concat(g, tv, v, 2);

            ggml_tensor * attn = attention(q, k, v);
            ggml_tensor * attn_txt = ggml_view_2d(g, attn, dim, n_txt, attn->nb[1], 0);
            ggml_tensor * attn_img = ggml_view_2d(g, attn, dim, n_img, attn->nb[1], (size_t) n_txt * attn->nb[1]);

            img = ggml_add(g, img, ggml_mul(g, linear(attn_img, pfx + "attn.to_out.0"),  chunk(img_mod, 2, dim)));
            txt = ggml_add(g, txt, ggml_mul(g, linear(attn_txt, pfx + "attn.to_add_out"), chunk(txt_mod, 2, dim)));

            ggml_tensor * img_m2 = modulate(ggml_norm(g, img, hp.eps), chunk(img_mod, 4, dim), chunk(img_mod, 3, dim));
            ggml_tensor * img_ff = linear(ggml_gelu(g, linear(img_m2, pfx + "img_mlp.net.0.proj")), pfx + "img_mlp.net.2", FFN_DOWN_SCALE);
            img = ggml_add(g, img, ggml_mul(g, img_ff, chunk(img_mod, 5, dim)));

            ggml_tensor * txt_m2 = modulate(ggml_norm(g, txt, hp.eps), chunk(txt_mod, 4, dim), chunk(txt_mod, 3, dim));
            ggml_tensor * txt_ff = linear(ggml_gelu(g, linear(txt_m2, pfx + "txt_mlp.net.0.proj")), pfx + "txt_mlp.net.2", FFN_DOWN_SCALE);
            txt = ggml_add(g, txt, ggml_mul(g, txt_ff, chunk(txt_mod, 5, dim)));
        }

        ggml_tensor * mod_out = linear(temb_act, "norm_out.linear");
        img = modulate(ggml_norm(g, img, hp.eps), chunk(mod_out, 0, dim), chunk(mod_out, 1, dim));
        return linear(img, "proj_out");
    }

    static void positions(const imgen_dit_hparams &, int hp_, int wp, int n_txt, std::vector<std::array<int, 3>> & pos) {
        const int max_vid = std::max(hp_ / 2, wp / 2);
        for (int i = 0; i < n_txt; i++) {
            pos.push_back({ max_vid + i, max_vid + i, max_vid + i });
        }
        // image positions are centered (scale_rope)
        for (int h = 0; h < hp_; h++) {
            for (int w = 0; w < wp; w++) {
                pos.push_back({ 0, h - (hp_ - hp_ / 2), w - (wp - wp / 2) });
            }
        }
    }
};

//
// Z-Image (Lumina2): single stream, attention over [img, cap], both padded to a multiple of 32
//

constexpr int SEQ_MULTI_OF = 32;

struct lumina2_graph : dit_graph {
    using dit_graph::dit_graph;

    ggml_tensor * block(ggml_tensor * x, const std::string & pfx, ggml_tensor * adaln, ggml_tensor * cos, ggml_tensor * sin) {
        const int64_t L   = x->ne[1];
        const int     dim = hp.dim;
        ggml_tensor * scale_msa = nullptr, * gate_msa = nullptr, * scale_mlp = nullptr, * gate_mlp = nullptr;
        if (adaln) {
            ggml_tensor * mod = linear(adaln, pfx + "adaLN_modulation.0");
            scale_msa = chunk(mod, 0, dim);
            gate_msa  = ggml_tanh(g, chunk(mod, 1, dim));
            scale_mlp = chunk(mod, 2, dim);
            gate_mlp  = ggml_tanh(g, chunk(mod, 3, dim));
        }

        ggml_tensor * h = rms_norm(x, w(pfx + "attention_norm1.weight"), hp.eps);
        if (scale_msa) {
            h = modulate(h, scale_msa, nullptr);
        }

        ggml_tensor * qkv = linear(h, pfx + "attention.qkv");
        const int64_t nq = (int64_t) hp.n_heads * hp.head_dim;
        const int64_t nk = (int64_t) hp.n_kv_heads * hp.head_dim;
        const size_t  es = ggml_element_size(qkv);
        ggml_tensor * q = ggml_cont(g, ggml_view_2d(g, qkv, nq, L, qkv->nb[1], 0));
        ggml_tensor * k = ggml_cont(g, ggml_view_2d(g, qkv, nk, L, qkv->nb[1], nq * es));
        ggml_tensor * v = ggml_cont(g, ggml_view_2d(g, qkv, nk, L, qkv->nb[1], (nq + nk) * es));
        q = ggml_reshape_3d(g, q, hp.head_dim, hp.n_heads, L);
        k = ggml_reshape_3d(g, k, hp.head_dim, hp.n_kv_heads, L);
        v = ggml_reshape_3d(g, v, hp.head_dim, hp.n_kv_heads, L);
        q = rope(rms_norm(q, w(pfx + "attention.q_norm.weight"), hp.eps), cos, sin);
        k = rope(rms_norm(k, w(pfx + "attention.k_norm.weight"), hp.eps), cos, sin);

        ggml_tensor * attn = linear(attention(q, k, v), pfx + "attention.out");
        attn = rms_norm(attn, w(pfx + "attention_norm2.weight"), hp.eps);
        x = ggml_add(g, x, gate_msa ? ggml_mul(g, attn, gate_msa) : attn);

        h = rms_norm(x, w(pfx + "ffn_norm1.weight"), hp.eps);
        if (scale_mlp) {
            h = modulate(h, scale_mlp, nullptr);
        }
        ggml_tensor * ff = ggml_swiglu_split(g, linear(h, pfx + "feed_forward.w1"), linear(h, pfx + "feed_forward.w3"));
        ff = linear(ff, pfx + "feed_forward.w2", FFN_DOWN_SCALE);
        ff = rms_norm(ff, w(pfx + "ffn_norm2.weight"), hp.eps);
        return ggml_add(g, x, gate_mlp ? ggml_mul(g, ff, gate_mlp) : ff);
    }

    ggml_tensor * pad_tokens(ggml_tensor * x, int n_pad, const char * pad_token) {
        if (n_pad == 0) {
            return x;
        }
        ggml_tensor * pad = ggml_repeat_4d(g, w(pad_token), hp.dim, n_pad, 1, 1);
        return ggml_concat(g, x, pad, 1);
    }

    ggml_tensor * slice_tokens(ggml_tensor * t, int off, int n) {
        return ggml_view_4d(g, t, t->ne[0], t->ne[1], t->ne[2], n, t->nb[1], t->nb[2], t->nb[3], (size_t) off * t->nb[3]);
    }

    ggml_tensor * build(const imgen_cond & cond, int hp_, int wp, const float & t_scaled, const std::vector<float> & x_in,
                        const std::vector<float> & cos_t, const std::vector<float> & sin_t) {
        const int n_img     = hp_ * wp;
        const int n_img_pad = (n_img + SEQ_MULTI_OF - 1) / SEQ_MULTI_OF * SEQ_MULTI_OF;
        const int n_cap     = cond.n_tokens;
        const int n_cap_pad = (n_cap + SEQ_MULTI_OF - 1) / SEQ_MULTI_OF * SEQ_MULTI_OF;
        const int n_all     = n_img_pad + n_cap_pad;

        ggml_tensor * x   = input("x",   GGML_TYPE_F32, hp.in_ch * 4, n_img, 1, 1, x_in.data());
        ggml_tensor * cap = input("cap", GGML_TYPE_F32, hp.txt_dim, n_cap, 1, 1, cond.embd.data());
        ggml_tensor * t   = input("t",   GGML_TYPE_F32, 1, 1, 1, 1, &t_scaled);
        ggml_tensor * cos = input("cos", GGML_TYPE_F32, 1, hp.head_dim / 2, 1, n_all, cos_t.data());
        ggml_tensor * sin = input("sin", GGML_TYPE_F32, 1, hp.head_dim / 2, 1, n_all, sin_t.data());

        ggml_tensor * cos_img = slice_tokens(cos, 0, n_img_pad);
        ggml_tensor * sin_img = slice_tokens(sin, 0, n_img_pad);
        ggml_tensor * cos_cap = slice_tokens(cos, n_img_pad, n_cap_pad);
        ggml_tensor * sin_cap = slice_tokens(sin, n_img_pad, n_cap_pad);

        ggml_tensor * adaln = timestep_embedding(t, 256);
        adaln = linear(adaln, "t_embedder.mlp.0");
        adaln = ggml_silu(g, adaln);
        adaln = linear(adaln, "t_embedder.mlp.2");

        x = pad_tokens(linear(x, "x_embedder"), n_img_pad - n_img, "x_pad_token");
        for (int il = 0; il < hp.n_refiner; il++) {
            x = block(x, "noise_refiner." + std::to_string(il) + ".", adaln, cos_img, sin_img);
        }

        cap = rms_norm(cap, w("cap_embedder.0.weight"), hp.eps);
        cap = pad_tokens(linear(cap, "cap_embedder.1"), n_cap_pad - n_cap, "cap_pad_token");
        for (int il = 0; il < hp.n_refiner; il++) {
            cap = block(cap, "context_refiner." + std::to_string(il) + ".", nullptr, cos_cap, sin_cap);
        }

        ggml_tensor * u = ggml_concat(g, x, cap, 1);
        for (int il = 0; il < hp.n_layers; il++) {
            u = block(u, "layers." + std::to_string(il) + ".", adaln, cos, sin);
        }

        ggml_tensor * scale = linear(ggml_silu(g, adaln), "final_layer.adaLN_modulation.1");
        u = modulate(ggml_norm(g, u, 1e-6f), scale, nullptr);
        u = linear(u, "final_layer.linear");
        return ggml_view_2d(g, u, u->ne[0], n_img, u->nb[1], 0);
    }

    static void positions(const imgen_dit_hparams &, int hp_, int wp, int n_cap, std::vector<std::array<int, 3>> & pos) {
        const int n_img     = hp_ * wp;
        const int n_img_pad = (n_img + SEQ_MULTI_OF - 1) / SEQ_MULTI_OF * SEQ_MULTI_OF;
        const int n_cap_pad = (n_cap + SEQ_MULTI_OF - 1) / SEQ_MULTI_OF * SEQ_MULTI_OF;
        for (int h = 0; h < hp_; h++) {
            for (int w = 0; w < wp; w++) {
                pos.push_back({ n_cap_pad + 1, h, w });
            }
        }
        for (int i = n_img; i < n_img_pad; i++) {
            pos.push_back({ 0, 0, 0 });
        }
        for (int i = 0; i < n_cap; i++) {
            pos.push_back({ 1 + i, 0, 0 });
        }
        for (int i = n_cap; i < n_cap_pad; i++) {
            pos.push_back({ 0, 0, 0 });
        }
    }
};

} // namespace

void imgen_dit_forward(imgen_context & ctx, const imgen_cond & cond, int hp, int wp, float t, const std::vector<float> & x, std::vector<float> & out) {
    imgen_runner & runner = *ctx.runner;
    ggml_context * g = runner.graph_begin();

    std::vector<std::array<int, 3>> pos;
    std::vector<float> cos_t, sin_t;
    const float t_scaled = t * 1000.0f;

    ggml_tensor * res;
    std::vector<input_data> inputs;
    if (ctx.arch == IMGEN_ARCH_QWEN_IMAGE) {
        qwen_image_graph gb(ctx, g);
        qwen_image_graph::positions(ctx.hp, hp, wp, cond.n_tokens, pos);
        rope_tables(ctx.hp, pos, cos_t, sin_t);
        res = gb.build(cond, hp, wp, t_scaled, x, cos_t, sin_t);
        inputs = std::move(gb.inputs);
    } else {
        lumina2_graph gb(ctx, g);
        lumina2_graph::positions(ctx.hp, hp, wp, cond.n_tokens, pos);
        rope_tables(ctx.hp, pos, cos_t, sin_t);
        res = gb.build(cond, hp, wp, t_scaled, x, cos_t, sin_t);
        inputs = std::move(gb.inputs);
    }
    ggml_set_output(res);
    ggml_build_forward_expand(runner.gf, res);

    if (!ggml_backend_sched_alloc_graph(runner.sched.get(), runner.gf)) {
        throw std::runtime_error("imgen: failed to allocate DiT graph");
    }
    for (auto & in : inputs) {
        ggml_backend_tensor_set(in.t, in.data, 0, ggml_nbytes(in.t));
    }
    runner.graph_compute();

    out.resize(ggml_nelements(res));
    if (ggml_is_contiguous(res)) {
        ggml_backend_tensor_get(res, out.data(), 0, ggml_nbytes(res));
    } else {
        // view of the first n_img columns of a larger contiguous tensor
        ggml_backend_tensor_get(res->view_src, out.data(), res->view_offs, out.size() * sizeof(float));
    }
}
