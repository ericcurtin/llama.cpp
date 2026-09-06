#include "imgen-impl.h"

#include <cstring>

static const char * TEMPLATE_QWEN_IMAGE =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
static const int  DROP_QWEN_IMAGE = 34; // tokens of the system prefix

static const char * TEMPLATE_LUMINA2 = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";

imgen_text_encoder::~imgen_text_encoder() {
    if (ctx) {
        llama_free(ctx);
    }
}

static bool te_cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto * te = (imgen_text_encoder *) ud;
    char want[32];
    snprintf(want, sizeof(want), "l_out-%d", te->tap_layer);
    const bool match = strcmp(t->name, want) == 0;
    if (ask) {
        return match;
    }
    if (match && t->ne[1] == te->n_tokens_cur && t->type == GGML_TYPE_F32) {
        te->captured.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, te->captured.data(), 0, ggml_nbytes(t));
    }
    return true;
}

static bool te_init(imgen_context & ctx) {
    auto & te = ctx.te;
    if (te.ctx) {
        return true;
    }
    // Z-Image uses hidden_states[-2], the input of the last layer
    te.tap_layer = ctx.arch == IMGEN_ARCH_LUMINA2 ? llama_model_n_layer(te.model) - 2 : -1;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = 2048;
    cp.n_batch      = 2048;
    cp.n_ubatch     = 2048;
    cp.n_seq_max    = 1;
    cp.embeddings   = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.no_perf      = true;
    if (ctx.params.n_threads > 0) {
        cp.n_threads       = ctx.params.n_threads;
        cp.n_threads_batch = ctx.params.n_threads;
    }
    if (te.tap_layer >= 0) {
        cp.cb_eval           = te_cb_eval;
        cp.cb_eval_user_data = &te;
    }
    te.ctx = llama_init_from_model(const_cast<llama_model *>(te.model), cp);
    if (!te.ctx) {
        IMGEN_LOG("%s: failed to create text encoder context\n", __func__);
        return false;
    }
    return true;
}

bool imgen_encode_prompt(imgen_context & ctx, const std::string & prompt, imgen_cond & out) {
    if (!te_init(ctx)) {
        return false;
    }
    auto & te = ctx.te;
    const llama_vocab * vocab = llama_model_get_vocab(te.model);

    const char * tmpl = ctx.arch == IMGEN_ARCH_QWEN_IMAGE ? TEMPLATE_QWEN_IMAGE : TEMPLATE_LUMINA2;
    const int drop = ctx.arch == IMGEN_ARCH_QWEN_IMAGE ? DROP_QWEN_IMAGE : 0;
    std::string text(snprintf(nullptr, 0, tmpl, prompt.c_str()) + 1, '\0');
    snprintf(text.data(), text.size(), tmpl, prompt.c_str());
    text.pop_back();

    const int n_ctx = llama_n_ctx(te.ctx);
    std::vector<llama_token> tokens(n_ctx);
    int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), false, true);
    if (n < 0) {
        IMGEN_LOG("%s: prompt too long, truncated to %d tokens\n", __func__, n_ctx);
        n = n_ctx;
    }
    tokens.resize(n);
    if (n <= drop) {
        IMGEN_LOG("%s: prompt produced no tokens\n", __func__);
        return false;
    }

    llama_memory_clear(llama_get_memory(te.ctx), true);
    te.n_tokens_cur = n;
    te.captured.clear();

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = true;
    }
    batch.n_tokens = n;
    const int rc = llama_decode(te.ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        IMGEN_LOG("%s: llama_decode failed (%d)\n", __func__, rc);
        return false;
    }

    const int n_embd = ctx.hp.txt_dim;
    out.n_tokens = n - drop;
    out.embd.resize((size_t) out.n_tokens * n_embd);

    if (te.tap_layer >= 0) {
        if ((int) te.captured.size() != n * n_embd) {
            IMGEN_LOG("%s: failed to capture layer %d output\n", __func__, te.tap_layer);
            return false;
        }
        memcpy(out.embd.data(), te.captured.data() + (size_t) drop * n_embd, out.embd.size() * sizeof(float));
    } else {
        for (int i = drop; i < n; i++) {
            const float * e = llama_get_embeddings_ith(te.ctx, i);
            if (!e) {
                IMGEN_LOG("%s: no embeddings for token %d\n", __func__, i);
                return false;
            }
            memcpy(out.embd.data() + (size_t) (i - drop) * n_embd, e, n_embd * sizeof(float));
        }
    }
    return true;
}
