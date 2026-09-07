#include "server-mediagen.h"
#include "server-common.h"

#include "mediagen.h"
#include "base64.hpp"
#include "build-info.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <thread>

#define MG_SRV_INF(fmt, ...) LOG_INF("mediagen: " fmt, __VA_ARGS__)
#define MG_SRV_ERR(fmt, ...) LOG_ERR("mediagen: " fmt, __VA_ARGS__)

static std::string b64(const uint8_t * data, size_t n) {
    return base64::encode((const char *) data, n);
}

// SSE events from the generation thread to the HTTP writer
struct mg_event_queue {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::string> events;
    bool done = false;

    void push(const json & ev) {
        std::lock_guard<std::mutex> lock(mutex);
        events.push_back("data: " + safe_json_to_str(ev) + "\n\n");
        cv.notify_all();
    }
    void finish() {
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
        cv.notify_all();
    }
    // false when done
    bool pop(std::string & out) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return !events.empty() || done; });
        if (events.empty()) {
            return false;
        }
        out = std::move(events.front());
        events.pop_front();
        return true;
    }
};

// parse "WxH" size strings
static bool parse_size(const std::string & s, int & w, int & h) {
    const size_t x = s.find_first_of("xX*");
    if (x == std::string::npos) {
        return false;
    }
    try {
        w = std::stoi(s.substr(0, x));
        h = std::stoi(s.substr(x + 1));
    } catch (...) {
        return false;
    }
    return w > 0 && h > 0;
}

// request limits
static constexpr int MG_MAX_DIM    = 4096;
static constexpr int MG_MAX_FRAMES = 1025;
static constexpr int MG_MAX_STEPS  = 200;

static int in_range(const char * name, int v, int lo, int hi) {
    if (v < lo || v > hi) {
        throw std::invalid_argument(std::string("\"") + name + "\" must be between " + std::to_string(lo) + " and " + std::to_string(hi));
    }
    return v;
}

// nearest multiple of m within the size limits
static int round_dim(const char * name, int v, int m) {
    v = in_range(name, v, 1, MG_MAX_DIM);
    return std::max(m, (v + m / 2) / m * m);
}

server_mediagen::server_mediagen() {
    not_supported = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 400;
        res->data = safe_json_to_str({{"error", format_error_response(
            "the loaded model is a media generation model, use /v1/images/generations, /v1/videos or /v1/audio/speech",
            ERROR_TYPE_NOT_SUPPORTED)}});
        return res;
    };

    get_models = [this](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        json caps = json::array();
        if (ctx && mediagen_supports_image(ctx)) caps.push_back("image");
        if (ctx && mediagen_supports_video(ctx)) caps.push_back("video");
        if (ctx && mediagen_supports_audio(ctx)) caps.push_back("audio");
        res->data = safe_json_to_str({
            {"models", json::array({
                {
                    {"name",  model_name},
                    {"model", model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""},
                    {"type", "model"},
                    {"description", ""},
                    {"tags", json::array({""})},
                    {"capabilities", caps},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", "ltx"},
                        {"families", json::array({"ltx"})},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            })},
            {"object", "list"},
            {"data", json::array({
                {
                    {"id",       model_name},
                    {"object",   "model"},
                    {"created",  std::time(0)},
                    {"owned_by", "llamacpp"},
                    {"capabilities", caps},
                }
            })}
        });
        return res;
    };

    get_props = [this](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->data = safe_json_to_str({
            { "model_alias", model_name },
            { "model_path",  params.model.path },
            { "modalities",  json {
                {"vision", false},
                {"video",  false},
                {"audio",  false},
                {"image_generation", ctx && mediagen_supports_image(ctx)},
                {"video_generation", ctx && mediagen_supports_video(ctx)},
                {"audio_generation", ctx && mediagen_supports_audio(ctx)},
            } },
            { "default_generation_settings", json {
                {"width",     params.mediagen.width},
                {"height",    params.mediagen.height},
                {"fps",       params.mediagen.fps},
                {"steps",     params.mediagen.steps},
                {"cfg_scale", params.mediagen.cfg_scale},
            } },
            { "build_info", std::string(llama_build_info()) },
        });
        return res;
    };

    // POST /v1/images/generations: OpenAI fields plus negative_prompt, seed, steps, cfg_scale, width, height
    post_images_generations = [this](const server_http_req & req) -> server_http_res_ptr {
        if (!ctx) {
            throw std::runtime_error("media generation model not loaded");
        }
        const json body = json::parse(req.body);
        const std::string prompt = json_value(body, "prompt", std::string());
        if (prompt.empty()) {
            throw std::invalid_argument("\"prompt\" is required");
        }
        const int n = json_value(body, "n", 1);
        if (n != 1) {
            throw std::invalid_argument("only n=1 is supported");
        }
        const std::string response_format = json_value(body, "response_format", std::string("b64_json"));
        if (response_format != "b64_json") {
            throw std::invalid_argument("only response_format=b64_json is supported");
        }

        mediagen_gen_params gp = mediagen_gen_params_default();
        int width = params.mediagen.width, height = params.mediagen.height;
        const std::string size = json_value(body, "size", std::string());
        if (!size.empty() && size != "auto" && !parse_size(size, width, height)) {
            throw std::invalid_argument("invalid \"size\", expected WIDTHxHEIGHT");
        }
        width  = json_value(body, "width",  width);
        height = json_value(body, "height", height);
        gp.width     = round_dim("width", width, 32);
        gp.height    = round_dim("height", height, 32);
        gp.n_frames  = 1;
        gp.n_steps   = in_range("steps", json_value(body, "steps", params.mediagen.steps), 0, MG_MAX_STEPS);
        gp.cfg_scale = json_value(body, "cfg_scale", params.mediagen.cfg_scale);
        gp.seed      = (uint32_t) json_value(body, "seed", (int64_t) -1);
        if (json_value(body, "seed", (int64_t) -1) < 0) {
            gp.seed = UINT32_MAX;
        }
        gp.enhance_prompt = json_value(body, "enhance_prompt", params.mediagen.enhance_prompt);
        const std::string negative = json_value(body, "negative_prompt", params.mediagen.negative_prompt);
        const bool stream = json_value(body, "stream", false);

        auto generate = [this, gp, prompt, negative](mediagen_progress_cb cb, void * ud, std::string & png_b64, std::string & revised) -> bool {
            mediagen_gen_params p = gp;
            p.prompt          = prompt.c_str();
            p.negative_prompt = negative.c_str();
            p.progress        = cb;
            p.progress_user_data = ud;
            std::lock_guard<std::mutex> lock(mutex);
            mediagen_result * res = mediagen_generate(ctx, &p);
            if (!res) {
                return false;
            }
            size_t n_png = 0;
            uint8_t * png = mediagen_encode_png(res->rgb, res->width, res->height, &n_png);
            revised = res->revised_prompt ? res->revised_prompt : prompt;
            mediagen_result_free(res);
            if (!png) {
                return false;
            }
            png_b64 = b64(png, n_png);
            free(png);
            return true;
        };

        if (!stream) {
            std::string png_b64, revised;
            if (!generate(nullptr, nullptr, png_b64, revised)) {
                throw std::runtime_error("image generation failed");
            }
            auto res = std::make_unique<server_http_res>();
            res->data = safe_json_to_str({
                {"created", std::time(0)},
                {"data", json::array({ { {"b64_json", png_b64}, {"revised_prompt", revised} } })},
                {"usage", { {"input_tokens", 0}, {"output_tokens", 0}, {"total_tokens", 0} }},
            });
            return res;
        }

        // streaming: progress events, then the image
        struct stream_state {
            mg_event_queue queue;
            std::thread worker;
            std::atomic<bool> cancelled{false};
        };
        auto state = std::make_shared<stream_state>();
        state->worker = std::thread([state, generate]() {
            auto cb = [](int step, int n_steps, void * ud) -> bool {
                stream_state * st = (stream_state *) ud;
                st->queue.push({{"type", "image_generation.progress"}, {"step", step}, {"total", n_steps}});
                return !st->cancelled.load();
            };
            std::string png_b64, revised;
            if (generate(cb, state.get(), png_b64, revised)) {
                state->queue.push({{"type", "image_generation.completed"}, {"b64_json", png_b64}, {"revised_prompt", revised}, {"created_at", std::time(0)}});
            } else {
                state->queue.push({{"type", "error"}, {"error", {{"message", "image generation failed"}}}});
            }
            state->queue.finish();
        });

        struct stream_res : server_http_res {
            std::shared_ptr<stream_state> state;
            ~stream_res() override {
                if (state) {
                    state->cancelled.store(true);
                    if (state->worker.joinable()) {
                        state->worker.join();
                    }
                }
            }
        };
        auto res = std::make_unique<stream_res>();
        res->state = state;
        res->content_type = "text/event-stream";
        res->headers["Cache-Control"] = "no-cache";
        res->next = [state](std::string & out) -> bool {
            return state->queue.pop(out);
        };
        return res;
    };

    // GET /v1/videos/:id
    get_video = [this](const server_http_req & req) {
        const std::string id = req.get_param("id");
        std::lock_guard<std::mutex> lock(videos_mutex);
        for (auto & v : videos) {
            if (v.id == id) {
                auto res = std::make_unique<server_http_res>();
                res->data = v.meta;
                return res;
            }
        }
        auto res = std::make_unique<server_http_res>();
        res->status = 404;
        res->data = safe_json_to_str({{"error", format_error_response("video not found", ERROR_TYPE_NOT_FOUND)}});
        return res;
    };

    // GET /v1/videos/:id/content -> video/mp4
    get_video_content = [this](const server_http_req & req) {
        const std::string id = req.get_param("id");
        std::lock_guard<std::mutex> lock(videos_mutex);
        for (auto & v : videos) {
            if (v.id == id) {
                auto res = std::make_unique<server_http_res>();
                if (v.mp4.empty()) {
                    res->status = 501;
                    res->data = safe_json_to_str({{"error", format_error_response(
                        "mp4 output needs the ffmpeg binary in PATH; the frames are in the job object", ERROR_TYPE_NOT_SUPPORTED)}});
                    return res;
                }
                res->content_type = "video/mp4";
                res->headers["Content-Disposition"] = "attachment; filename=\"" + id + ".mp4\"";
                res->data = v.mp4;
                return res;
            }
        }
        auto res = std::make_unique<server_http_res>();
        res->status = 404;
        res->data = safe_json_to_str({{"error", format_error_response("video not found", ERROR_TYPE_NOT_FOUND)}});
        return res;
    };

    // POST /v1/videos: synchronous, returns the completed job; the mp4 is served by /v1/videos/:id/content
    // (frames and audio are inlined with response_format "frames" or without ffmpeg)
    post_videos = [this](const server_http_req & req) {
        if (!ctx) {
            throw std::runtime_error("media generation model not loaded");
        }
        const json body = json::parse(req.body);
        const std::string prompt = json_value(body, "prompt", std::string());
        if (prompt.empty()) {
            throw std::invalid_argument("\"prompt\" is required");
        }
        mediagen_gen_params gp = mediagen_gen_params_default();
        int width = params.mediagen.width, height = params.mediagen.height;
        const std::string size = json_value(body, "size", std::string());
        if (!size.empty() && size != "auto" && !parse_size(size, width, height)) {
            throw std::invalid_argument("invalid \"size\", expected WIDTHxHEIGHT");
        }
        gp.width  = round_dim("width", json_value(body, "width",  width), 32);
        gp.height = round_dim("height", json_value(body, "height", height), 32);
        gp.fps    = json_value(body, "fps", params.mediagen.fps);
        if (!(gp.fps >= 1.0f && gp.fps <= 120.0f)) {
            throw std::invalid_argument("\"fps\" must be between 1 and 120");
        }
        double seconds = 0.0;
        if (body.contains("seconds")) {
            seconds = body["seconds"].is_string() ? std::stod(body["seconds"].get<std::string>()) : body["seconds"].get<double>();
        }
        int n_frames = json_value(body, "frames", 0);
        if (n_frames <= 0) {
            n_frames = seconds > 0 ? (int) std::lround(std::min(seconds, 60.0) * gp.fps) : 33;
        }
        n_frames     = in_range("frames", n_frames, 9, MG_MAX_FRAMES);
        gp.n_frames  = (n_frames - 1) / 8 * 8 + 1;
        gp.n_steps   = in_range("steps", json_value(body, "steps", params.mediagen.steps), 0, MG_MAX_STEPS);
        gp.cfg_scale = json_value(body, "cfg_scale", params.mediagen.cfg_scale);
        gp.seed      = json_value(body, "seed", (int64_t) -1) < 0 ? UINT32_MAX : (uint32_t) json_value(body, "seed", (int64_t) 0);
        gp.gen_audio = json_value(body, "audio", mediagen_supports_audio(ctx));
        gp.enhance_prompt = json_value(body, "enhance_prompt", params.mediagen.enhance_prompt);
        const std::string negative = json_value(body, "negative_prompt", params.mediagen.negative_prompt);
        gp.prompt          = prompt.c_str();
        gp.negative_prompt = negative.c_str();

        mediagen_result * r;
        {
            std::lock_guard<std::mutex> lock(mutex);
            r = mediagen_generate(ctx, &gp);
        }
        if (!r) {
            throw std::runtime_error("video generation failed");
        }
        const std::string response_format = json_value(body, "response_format", std::string("mp4"));
        std::string mp4;
        {
            size_t n = 0;
            uint8_t * buf = mediagen_encode_mp4(r, &n);
            if (buf) {
                mp4.assign((const char *) buf, n);
                free(buf);
            }
        }
        static std::atomic<int> counter{0};
        const std::string id = "video_" + std::to_string(std::time(0)) + "_" + std::to_string(counter++);
        json out = {
            {"id", id},
            {"object", "video"},
            {"model", model_name},
            {"status", "completed"},
            {"progress", 100},
            {"created_at", std::time(0)},
            {"completed_at", std::time(0)},
            {"size", std::to_string(r->width) + "x" + std::to_string(r->height)},
            {"seconds", std::to_string((double) r->n_frames / r->fps)},
            {"fps", r->fps},
            {"n_frames", r->n_frames},
            {"has_audio", r->pcm != nullptr},
            {"revised_prompt", r->revised_prompt ? std::string(r->revised_prompt) : prompt},
        };
        if (!mp4.empty()) {
            out["content_url"] = "/v1/videos/" + id + "/content";
        }
        if (response_format == "frames" || mp4.empty()) {
            json frames = json::array();
            const size_t frame_bytes = (size_t) r->width * r->height * 3;
            for (int f = 0; f < r->n_frames; f++) {
                size_t n_png = 0;
                uint8_t * png = mediagen_encode_png(r->rgb + f * frame_bytes, r->width, r->height, &n_png);
                if (!png) {
                    mediagen_result_free(r);
                    throw std::runtime_error("failed to encode frame");
                }
                frames.push_back(b64(png, n_png));
                free(png);
            }
            out["frames"] = frames;
            if (r->pcm) {
                size_t n_wav = 0;
                uint8_t * wav = mediagen_encode_wav(r->pcm, r->n_samples, r->n_channels, r->sample_rate, &n_wav);
                if (wav) {
                    out["audio"] = { {"format", "wav"}, {"sample_rate", r->sample_rate}, {"channels", r->n_channels}, {"b64_wav", b64(wav, n_wav)} };
                    free(wav);
                }
            }
        }
        mediagen_result_free(r);
        // GET serves the job without the inline frames
        json stored = out;
        stored.erase("frames");
        stored.erase("audio");
        {
            std::lock_guard<std::mutex> lock(videos_mutex);
            videos.push_back({ id, safe_json_to_str(stored), std::move(mp4) });
            while (videos.size() > 8) {
                videos.erase(videos.begin());
            }
        }
        auto res = std::make_unique<server_http_res>();
        res->data = safe_json_to_str(out);
        return res;
    };

    // POST /v1/audio/speech: text to audio, returns a wav file
    post_audio_speech = [this](const server_http_req & req) {
        if (!ctx) {
            throw std::runtime_error("media generation model not loaded");
        }
        if (!mediagen_supports_audio(ctx)) {
            throw std::invalid_argument("the loaded model does not support audio generation");
        }
        const json body = json::parse(req.body);
        const std::string prompt = json_value(body, "input", std::string());
        if (prompt.empty()) {
            throw std::invalid_argument("\"input\" is required");
        }
        const std::string fmt = json_value(body, "response_format", std::string("wav"));
        if (fmt != "wav") {
            throw std::invalid_argument("only response_format=wav is supported");
        }
        mediagen_gen_params gp = mediagen_gen_params_default();
        const double seconds = json_value(body, "seconds", 4.0);
        if (!(seconds > 0.0 && seconds <= 60.0)) {
            throw std::invalid_argument("\"seconds\" must be between 0 and 60");
        }
        gp.fps      = params.mediagen.fps;
        gp.width    = 256; // audio is generated jointly with a small video
        gp.height   = 256;
        gp.n_frames = std::max(9, (int) std::lround(seconds * gp.fps) / 8 * 8 + 1);
        gp.n_steps  = in_range("steps", json_value(body, "steps", params.mediagen.steps), 0, MG_MAX_STEPS);
        gp.cfg_scale = json_value(body, "cfg_scale", params.mediagen.cfg_scale);
        gp.seed     = json_value(body, "seed", (int64_t) -1) < 0 ? UINT32_MAX : (uint32_t) json_value(body, "seed", (int64_t) 0);
        gp.gen_audio = true;
        gp.enhance_prompt = json_value(body, "enhance_prompt", params.mediagen.enhance_prompt);
        const std::string negative = json_value(body, "negative_prompt", params.mediagen.negative_prompt);
        gp.prompt          = prompt.c_str();
        gp.negative_prompt = negative.c_str();

        mediagen_result * r;
        {
            std::lock_guard<std::mutex> lock(mutex);
            r = mediagen_generate(ctx, &gp);
        }
        if (!r || !r->pcm) {
            mediagen_result_free(r);
            throw std::runtime_error("audio generation failed");
        }
        size_t n_wav = 0;
        uint8_t * wav = mediagen_encode_wav(r->pcm, r->n_samples, r->n_channels, r->sample_rate, &n_wav);
        mediagen_result_free(r);
        if (!wav) {
            throw std::runtime_error("failed to encode audio");
        }
        auto res = std::make_unique<server_http_res>();
        res->content_type = "audio/wav";
        res->data.assign((const char *) wav, n_wav);
        free(wav);
        return res;
    };
}

server_mediagen::~server_mediagen() {
    unload();
}

bool server_mediagen::is_diffusion_model(const common_params & params) {
    return !params.model.path.empty() && mediagen_is_diffusion_model(params.model.path.c_str());
}

bool server_mediagen::load(const common_params & params_) {
    params = params_;
    model_name = params.model_alias.empty() ? params.model.get_name() : *params.model_alias.begin();
    if (params.mediagen.text_encoder.path.empty()) {
        MG_SRV_ERR("%s", "a text encoder is required for the diffusion model (use --text-encoder or --text-encoder-hf)\n");
        return false;
    }
    if (params.mediagen.vae.path.empty()) {
        MG_SRV_ERR("%s", "a VAE is required for the diffusion model (use --vae)\n");
        return false;
    }
    llama_model_params mparams = common_model_params_to_llama(params);
    text_model = llama_model_load_from_file(params.mediagen.text_encoder.path.c_str(), mparams);
    if (!text_model) {
        MG_SRV_ERR("failed to load text encoder %s\n", params.mediagen.text_encoder.path.c_str());
        return false;
    }
    mediagen_context_params cparams = mediagen_context_params_default();
    cparams.model_path     = params.model.path.c_str();
    cparams.vae_path       = params.mediagen.vae.path.c_str();
    cparams.audio_vae_path = params.mediagen.audio_vae.path.empty() ? nullptr : params.mediagen.audio_vae.path.c_str();
    cparams.text_proj_path = params.mediagen.text_proj.path.empty() ? nullptr : params.mediagen.text_proj.path.c_str();
    cparams.text_model     = text_model;
    cparams.use_gpu        = true; // -ngl applies to the text encoder only
    cparams.n_threads      = params.cpuparams.n_threads;
    cparams.flash_attn     = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.verbosity      = GGML_LOG_LEVEL_INFO;
    ctx = mediagen_init(cparams);
    if (!ctx) {
        MG_SRV_ERR("%s", "failed to initialize the media generation model\n");
        llama_model_free(text_model);
        text_model = nullptr;
        return false;
    }
    MG_SRV_INF("media generation model loaded: image=%d video=%d audio=%d\n",
               mediagen_supports_image(ctx), mediagen_supports_video(ctx), mediagen_supports_audio(ctx));
    return true;
}

void server_mediagen::unload() {
    if (ctx) {
        mediagen_free(ctx);
        ctx = nullptr;
    }
    if (text_model) {
        llama_model_free(text_model);
        text_model = nullptr;
    }
}
