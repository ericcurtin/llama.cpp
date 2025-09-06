// llama-run: Interactive chat interface using server infrastructure with shim layer
// This implementation shares code with llama-server by using a shim layer instead of HTTP

#include "server-shim.hpp"

// Include what we need from server utils, but avoid full utils.hpp for now
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "arg.h"  // For common_params_parse

// For JSON handling
#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include "linenoise.cpp/linenoise.h"

#include <iostream>
#include <string>
#include <atomic>
#include <signal.h>
#include <memory>
#include <thread>

using json = nlohmann::ordered_json;

// Global state management
static std::atomic<bool> should_exit{false};
static std::atomic<bool> interrupt_response{false};
static std::unique_ptr<llama_run::Server> svr;

// Simple server context to hold what we need
struct simple_server_context {
    common_params params;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* sampler = nullptr;
    common_init_result llama_init;
    
    ~simple_server_context() {
        if (sampler) {
            llama_sampler_free(sampler);
        }
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
    }
};

static std::unique_ptr<simple_server_context> ctx_server;

// Signal handlers
static void sigint_handler(int sig) {
    (void)sig;
    interrupt_response.store(true);
}

static void sigterm_handler(int sig) {
    (void)sig;
    should_exit.store(true);
}

// Initialize the simple server context
static bool init_server_context(common_params& params) {
    ctx_server = std::make_unique<simple_server_context>();
    ctx_server->params = params;
    
    // Initialize the model using common_init_result
    ctx_server->llama_init = common_init_from_params(params);
    
    ctx_server->model = ctx_server->llama_init.model.get();
    ctx_server->ctx = ctx_server->llama_init.context.get();
    
    if (!ctx_server->model || !ctx_server->ctx) {
        LOG_ERR("Failed to initialize model or context\n");
        return false;
    }
    
    // Initialize sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    ctx_server->sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(ctx_server->sampler, llama_sampler_init_greedy());
    
    return true;
}

// Extract content from chat completion response 
static std::string extract_chat_content(const json& response) {
    try {
        if (response.contains("choices") && response["choices"].is_array() && !response["choices"].empty()) {
            const auto& choice = response["choices"][0];
            if (choice.contains("message") && choice["message"].contains("content")) {
                return choice["message"]["content"].get<std::string>();
            }
        }
        return "Error: No content found in response";
    } catch (const std::exception& e) {
        return "Error parsing response: " + std::string(e.what());
    }
}

// Simple chat completion using direct llama API
static std::string simple_chat_completion(const std::string& message) {
    if (!ctx_server || !ctx_server->ctx || !ctx_server->sampler) {
        return "Error: Server not initialized";
    }
    
    try {
        // Get vocab for tokenization
        const llama_vocab* vocab = llama_model_get_vocab(ctx_server->model);
        
        // Tokenize the input message
        const int n_prompt = llama_tokenize(vocab, message.c_str(), message.size(), nullptr, 0, true, true);
        if (n_prompt < 0) {
            return "Error: Failed to tokenize input";
        }
        
        std::vector<llama_token> prompt_tokens(n_prompt);
        if (llama_tokenize(vocab, message.c_str(), message.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            return "Error: Failed to tokenize input";
        }
        
        // Clear KV cache by resetting sampler state
        llama_sampler_reset(ctx_server->sampler);
        
        // Prepare batch for the prompt
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        
        // Evaluate the prompt
        if (llama_decode(ctx_server->ctx, batch) != 0) {
            return "Error: Failed to decode prompt";
        }
        
        // Generate response
        std::string response;
        const int max_tokens = ctx_server->params.n_predict < 0 ? 128 : ctx_server->params.n_predict;
        
        for (int i = 0; i < max_tokens && !should_exit.load() && !interrupt_response.load(); ++i) {
            // Sample next token
            llama_token token = llama_sampler_sample(ctx_server->sampler, ctx_server->ctx, -1);
            
            // Check if it's end of generation
            if (llama_vocab_is_eog(vocab, token)) {
                break;
            }
            
            // Convert token to piece
            char buf[128];
            int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
            if (n < 0) {
                break;
            }
            
            response += std::string(buf, n);
            
            // Prepare next batch with the sampled token
            batch = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx_server->ctx, batch) != 0) {
                break;
            }
        }
        
        return response;
        
    } catch (const std::exception& e) {
        return "Error: " + std::string(e.what());
    }
}

// Interactive loop
static int interactive_loop() {
    std::cout << "\nChat with the model (Ctrl-D to quit, Ctrl-C to interrupt response):\n";
    
    const char* input;
    while ((input = linenoise("> ")) != nullptr && !should_exit.load()) {
        std::string user_input(input);
        linenoiseHistoryAdd(input);
        linenoiseFree(const_cast<char*>(input));
        
        if (user_input.empty()) {
            continue;
        }
        
        // Reset interrupt flag
        interrupt_response.store(false);
        
        std::cout << std::flush;
        std::string response = simple_chat_completion(user_input);
        
        if (interrupt_response.load()) {
            std::cout << "\n[Response interrupted - press Ctrl-D to quit]\n";
            interrupt_response.store(false);
        } else {
            std::cout << response << "\n\n";
        }
    }
    
    return 0;
}

static void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [server-options]\n";
    std::cout << "\nThis tool provides an interactive chat interface using shared llama.cpp server infrastructure.\n";
    std::cout << "All options are passed through to the llama server configuration.\n";
    std::cout << "\nCommon options:\n";
    std::cout << "  -h, --help                  Show this help\n";
    std::cout << "  -m,    --model FNAME        model path\n";
    std::cout << "  -hf,   -hfr, --hf-repo      <user>/<model>[:quant] Hugging Face model repository\n";
    std::cout << "  -c, --ctx-size N            Context size\n";
    std::cout << "  -n, --predict N             Number of tokens to predict\n";
    std::cout << "  -t, --threads N             Number of threads\n";
    std::cout << "\nFor all server options, run: llama-server --help\n";
}

int main(int argc, char** argv) {
    // Parse arguments using shared argument parsing system
    common_params params;
    
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    
    // Check for help
    if (params.usage) {
        print_usage(argv[0]);
        return 0;
    }
    
    common_init();
    
    // Setup signal handlers
    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigterm_handler);
    
    // Initialize llama backend
    llama_backend_init();
    llama_numa_init(params.numa);
    
    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", 
            params.cpuparams.n_threads, params.cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");
    
    // Initialize server context
    std::cout << "Loading model..." << std::flush;
    if (!init_server_context(params)) {
        std::cerr << " failed!\n";
        std::cerr << "Failed to initialize server context\n";
        return 1;
    }
    std::cout << " ready!\n";
    
    // Start interactive loop
    int result = interactive_loop();
    
    // Cleanup
    ctx_server.reset();
    llama_backend_free();
    
    return result;
}
