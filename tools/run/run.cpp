#include "log.h"
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <cstring>
#include <signal.h>

#include "linenoise.cpp/linenoise.h"

// Global variables for process management
static std::atomic<bool> should_exit{false};

// Direct llama.cpp integration class
class LlamaRunner {
private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_sampler* smpl = nullptr;
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted;
    int prev_len = 0;

public:
    LlamaRunner() = default;
    
    ~LlamaRunner() {
        // free resources
        for (auto& msg : messages) {
            free(const_cast<char*>(msg.content));
        }
        
        if (smpl) llama_sampler_free(smpl);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    bool initialize(const common_params& params) {
        // initialize the model
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = params.n_gpu_layers;

        model = llama_model_load_from_file(params.model.path.c_str(), model_params);
        if (model == nullptr) {
            LOG_ERR("unable to load model from '%s'\n", params.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model);

        // initialize the context
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = params.n_ctx;
        ctx_params.n_batch = params.n_batch;
        ctx_params.no_perf = false;

        ctx = llama_init_from_model(model, ctx_params);
        if (ctx == nullptr) {
            LOG_ERR("failed to create the llama_context\n");
            return false;
        }

        // initialize the sampler
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        smpl = llama_sampler_chain_init(sparams);

        // Configure sampling based on params
        if (params.sampling.temp > 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.sampling.temp));
        }
        if (params.sampling.top_k > 0) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.sampling.top_k));
        }
        if (params.sampling.top_p < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.sampling.top_p, 1));
        }
        
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        // Prepare formatted buffer
        formatted.resize(llama_n_ctx(ctx));

        return true;
    }

    std::string generate_response(const std::string& user_input) {
        if (!model || !ctx || !smpl) {
            return "Error: Model not initialized";
        }

        const char* tmpl = llama_model_chat_template(model, nullptr);
        if (!tmpl) {
            return "Error: No chat template available";
        }

        // Add the user input to the message list and format it
        messages.push_back({"user", strdup(user_input.c_str())});
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        }
        
        if (new_len < 0) {
            return "Error: Failed to apply chat template";
        }

        // Remove previous messages to obtain the prompt to generate the response
        std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);

        // Generate response
        std::string response = generate(prompt);

        // Add the response to the messages
        messages.push_back({"assistant", strdup(response.c_str())});
        prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
        
        return response;
    }

private:
    std::string generate(const std::string& prompt) {
        if (prompt.empty()) {
            return "";
        }

        std::string response;
        const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;

        // Tokenize the prompt
        const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
            return "Error: Failed to tokenize prompt";
        }

        // Prepare a batch for the prompt
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token_id;
        
        while (!should_exit.load()) {
            // Check if we have enough space in the context
            int n_ctx_max = llama_n_ctx(ctx);
            int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            
            if (n_ctx_used + batch.n_tokens > n_ctx_max) {
                return response + "\n[Context limit reached]";
            }

            int ret = llama_decode(ctx, batch);
            if (ret != 0) {
                return response + "\n[Decode error]";
            }

            // Sample the next token
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // Is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            // Convert token to string
            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                return response + "\n[Token conversion error]";
            }
            
            std::string piece(buf, n);
            response += piece;

            // Prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);
        }

        return response;
    }
};

// Signal handlers
static void sigint_handler(int sig) {
    (void)sig;
    should_exit.store(true);
}

static void sigterm_handler(int sig) {
    (void)sig;
    should_exit.store(true);
}

// Main interactive loop
static int interactive_loop(LlamaRunner& runner) {
    std::cout << "\nChat with the model (Ctrl-D to quit, Ctrl-C to exit):\n";
    
    const char* input;
    while ((input = linenoise("> ")) != nullptr && !should_exit.load()) {
        std::string user_input(input);
        linenoiseHistoryAdd(input);
        linenoiseFree(const_cast<char*>(input));
        
        if (user_input.empty()) {
            continue;
        }
        
        std::cout << std::flush;
        std::string response = runner.generate_response(user_input);
        
        if (should_exit.load()) {
            std::cout << "\n[Interrupted]\n";
            break;
        } else {
            std::cout << response << "\n\n";
        }
    }
    
    return 0;
}

static void print_usage(const char * program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "\nThis tool provides an interactive chat interface using llama.cpp directly.\n";
    std::cout << "\nCommon options:\n";
    std::cout << "  -h, --help                  Show this help\n";
    std::cout << "  -m,    --model FNAME        model path (required)\n";
    std::cout << "  -c, --ctx-size N            Context size (default: 2048)\n";
    std::cout << "  -n, --predict N             Number of tokens to predict (default: -1, unlimited)\n";
    std::cout << "  -t, --threads N             Number of threads\n";
    std::cout << "  -ngl, --n-gpu-layers N      Number of layers to offload to GPU (default: 99)\n";
    std::cout << "  --temp N                    Temperature for sampling (default: 0.8)\n";
    std::cout << "  --top-k N                   Top-k sampling (default: 40)\n";
    std::cout << "  --top-p N                   Top-p sampling (default: 0.9)\n";
}

int main(int argc, char** argv) {
    // Parse arguments using common_params
    common_params params;
    
    // Set defaults
    params.n_ctx = 2048;
    params.n_predict = -1; // unlimited
    params.n_gpu_layers = 99;
    params.sampling.temp = 0.8f;
    params.sampling.top_k = 40;
    params.sampling.top_p = 0.9f;
    
    // Simple argument parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                params.model.path = argv[++i];
            } else {
                std::cerr << "Error: --model requires a value\n";
                return 1;
            }
        } else if (arg == "-c" || arg == "--ctx-size") {
            if (i + 1 < argc) {
                params.n_ctx = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --ctx-size requires a value\n";
                return 1;
            }
        } else if (arg == "-n" || arg == "--predict") {
            if (i + 1 < argc) {
                params.n_predict = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --predict requires a value\n";
                return 1;
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                params.cpuparams.n_threads = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --threads requires a value\n";
                return 1;
            }
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (i + 1 < argc) {
                params.n_gpu_layers = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --n-gpu-layers requires a value\n";
                return 1;
            }
        } else if (arg == "--temp") {
            if (i + 1 < argc) {
                params.sampling.temp = std::stof(argv[++i]);
            } else {
                std::cerr << "Error: --temp requires a value\n";
                return 1;
            }
        } else if (arg == "--top-k") {
            if (i + 1 < argc) {
                params.sampling.top_k = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --top-k requires a value\n";
                return 1;
            }
        } else if (arg == "--top-p") {
            if (i + 1 < argc) {
                params.sampling.top_p = std::stof(argv[++i]);
            } else {
                std::cerr << "Error: --top-p requires a value\n";
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (params.model.path.empty()) {
        std::cerr << "Error: Model path is required (-m/--model)\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // Load dynamic backends
    ggml_backend_load_all();
    
    // Setup signal handlers
    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigterm_handler);
    
    // Initialize llama runner
    LlamaRunner runner;
    
    std::cout << "Loading model '" << params.model.path << "'...\n";
    
    if (!runner.initialize(params)) {
        std::cerr << "Failed to initialize model\n";
        return 1;
    }
    
    std::cout << "Model loaded successfully!\n";
    
    // Start interactive loop
    int result = interactive_loop(runner);
    
    return result;
}
