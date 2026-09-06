#include "imgen-impl.h"

#include "gguf.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <fstream>

using json = nlohmann::ordered_json;

namespace {

struct tensor_src {
    std::string          name;
    ggml_type            type;       // type in file
    std::vector<int64_t> ne;         // file shape, ggml order (ne[0] fastest)
    size_t               offset;     // absolute byte offset in file
};

// BF16 is converted to F16, except for vectors (biases, norms) which are used as F32 operands
ggml_type target_type(const tensor_src & src, size_t n_dims, bool keep_type) {
    if (src.type == GGML_TYPE_BF16) {
        return n_dims <= 1 ? GGML_TYPE_F32 : GGML_TYPE_F16;
    }
    if (keep_type || (src.type != GGML_TYPE_F32 && src.type != GGML_TYPE_F16)) {
        return src.type;
    }
    return n_dims <= 1 ? GGML_TYPE_F32 : GGML_TYPE_F16;
}

// drop size-1 dims of vectors stored as [C,1,1,1], keep everything else
std::vector<int64_t> squeeze(const std::vector<int64_t> & ne) {
    int n_big = 0;
    for (auto v : ne) {
        n_big += v != 1;
    }
    if (n_big > 1 || ne.size() <= 1) {
        return ne;
    }
    for (auto v : ne) {
        if (v != 1) {
            return {v};
        }
    }
    return {1};
}

void convert_row(const void * src, ggml_type src_type, void * dst, ggml_type dst_type, int64_t n) {
    if (src_type == dst_type) {
        memcpy(dst, src, ggml_row_size(src_type, n));
        return;
    }
    std::vector<float> tmp(n);
    switch (src_type) {
        case GGML_TYPE_F32:  memcpy(tmp.data(), src, n*sizeof(float)); break;
        case GGML_TYPE_F16:  ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, tmp.data(), n); break;
        case GGML_TYPE_BF16: ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, tmp.data(), n); break;
        default: throw std::runtime_error("imgen: unsupported source type for conversion");
    }
    switch (dst_type) {
        case GGML_TYPE_F32: memcpy(dst, tmp.data(), n*sizeof(float)); break;
        case GGML_TYPE_F16: ggml_fp32_to_fp16_row(tmp.data(), (ggml_fp16_t *) dst, n); break;
        default: throw std::runtime_error("imgen: unsupported target type for conversion");
    }
}

std::vector<tensor_src> read_gguf_index(const std::string & path) {
    ggml_context * meta = nullptr;
    gguf_init_params gparams = { /*no_alloc*/ true, /*ctx*/ &meta };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gparams);
    if (!gctx) {
        throw std::runtime_error("imgen: failed to open GGUF " + path);
    }
    const size_t data_offset = gguf_get_data_offset(gctx);
    std::vector<tensor_src> out;
    const int64_t n = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n; i++) {
        tensor_src t;
        t.name   = gguf_get_tensor_name(gctx, i);
        t.type   = gguf_get_tensor_type(gctx, i);
        t.offset = data_offset + gguf_get_tensor_offset(gctx, i);
        ggml_tensor * cur = ggml_get_tensor(meta, t.name.c_str());
        GGML_ASSERT(cur);
        for (int d = 0; d < ggml_n_dims(cur); d++) {
            t.ne.push_back(cur->ne[d]);
        }
        out.push_back(std::move(t));
    }
    gguf_free(gctx);
    ggml_free(meta);
    return out;
}

std::vector<tensor_src> read_safetensors_index(std::ifstream & fin) {
    uint64_t n_header = 0;
    fin.read((char *) &n_header, sizeof(n_header));
    if (!fin || n_header > (1u << 26)) {
        throw std::runtime_error("imgen: bad safetensors header");
    }
    std::string header(n_header, '\0');
    fin.read(header.data(), n_header);
    const size_t data_start = 8 + n_header;

    json j = json::parse(header);
    std::vector<tensor_src> out;
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() == "__metadata__") {
            continue;
        }
        const auto & v = it.value();
        const std::string dtype = v.at("dtype");
        tensor_src t;
        t.name = it.key();
        if (dtype == "F32") {
            t.type = GGML_TYPE_F32;
        } else if (dtype == "F16") {
            t.type = GGML_TYPE_F16;
        } else if (dtype == "BF16") {
            t.type = GGML_TYPE_BF16;
        } else {
            throw std::runtime_error("imgen: unsupported safetensors dtype " + dtype + " for " + t.name);
        }
        const auto & shape = v.at("shape");
        for (auto rit = shape.rbegin(); rit != shape.rend(); ++rit) {
            t.ne.push_back(rit->get<int64_t>());
        }
        t.offset = data_start + v.at("data_offsets").at(0).get<size_t>();
        out.push_back(t);
    }
    return out;
}

} // namespace

void imgen_load_weights(const std::string & path, ggml_backend_buffer_type_t buft, imgen_weights & out, bool keep_type) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("imgen: cannot open " + path);
    }

    char magic[4] = {};
    fin.read(magic, 4);
    fin.seekg(0);

    std::vector<tensor_src> index;
    if (memcmp(magic, "GGUF", 4) == 0) {
        index = read_gguf_index(path);
    } else {
        index = read_safetensors_index(fin);
    }

    ggml_init_params ip = { ggml_tensor_overhead() * (index.size() + 1), nullptr, true };
    out.ctx.reset(ggml_init(ip));

    struct plan_item {
        const tensor_src * src;
        ggml_tensor *      dst;
        int64_t            kt;   // temporal kernel size of 5D conv weights, 1 otherwise
    };
    std::vector<plan_item> plan;

    for (const auto & src : index) {
        std::vector<int64_t> ne = squeeze(src.ne);
        int64_t kt = 1;
        // Wan VAE 3D conv [KW,KH,KT,IC,OC]: a single frame only sees the last temporal slice
        if (ne.size() == 5) {
            kt = ne[2];
            ne.erase(ne.begin() + 2);
        }
        if (ne.size() > GGML_MAX_DIMS) {
            throw std::runtime_error("imgen: tensor " + src.name + " has too many dims");
        }
        ggml_type type = target_type(src, ne.size(), keep_type);
        ggml_tensor * t = ggml_new_tensor(out.ctx.get(), type, ne.size(), ne.data());
        ggml_set_name(t, src.name.c_str());
        out.tensors[src.name] = t;
        plan.push_back({ &src, t, kt });
    }

    out.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(out.ctx.get(), buft));
    if (!out.buf) {
        throw std::runtime_error("imgen: failed to allocate weights for " + path);
    }
    ggml_backend_buffer_set_usage(out.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<uint8_t> raw;
    std::vector<uint8_t> conv;
    for (auto & item : plan) {
        const tensor_src & src = *item.src;
        ggml_tensor * dst = item.dst;

        const int64_t n_rows  = ggml_nelements(dst) / dst->ne[0];
        const size_t  src_row = ggml_row_size(src.type, dst->ne[0]);
        const size_t  dst_row = ggml_row_size(dst->type, dst->ne[0]);

        conv.resize(dst_row * n_rows);

        if (item.kt == 1) {
            raw.resize(src_row * n_rows);
            fin.seekg(src.offset);
            fin.read((char *) raw.data(), raw.size());
            for (int64_t r = 0; r < n_rows; r++) {
                convert_row(raw.data() + r*src_row, src.type, conv.data() + r*dst_row, dst->type, dst->ne[0]);
            }
        } else {
            // src rows are grouped as [kt][kh] per (ic, oc); keep rows of the last kt
            const int64_t kh = dst->ne[1];
            const int64_t n_groups = n_rows / kh;
            raw.resize(src_row * kh * item.kt);
            for (int64_t g = 0; g < n_groups; g++) {
                fin.seekg(src.offset + g * raw.size());
                fin.read((char *) raw.data(), raw.size());
                const uint8_t * last = raw.data() + (item.kt - 1) * kh * src_row;
                for (int64_t r = 0; r < kh; r++) {
                    convert_row(last + r*src_row, src.type, conv.data() + (g*kh + r)*dst_row, dst->type, dst->ne[0]);
                }
            }
        }
        if (!fin) {
            throw std::runtime_error("imgen: failed to read tensor " + src.name);
        }
        ggml_backend_tensor_set(dst, conv.data(), 0, conv.size());
    }
}
