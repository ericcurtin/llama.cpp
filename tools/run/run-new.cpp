// llama-run: Interactive chat interface using server infrastructure without sockets
// This implementation shares code with llama-server by using a shim layer

#include "server-shim.hpp"

// Import shared server utilities
#include "../server/utils.hpp"

#include "linenoise.cpp/linenoise.h"

#include <iostream>
#include <string>
#include <atomic>
#include <signal.h>
#include <memory>
#include <thread>

// Re-declare types and functions we need from server.cpp without including the whole file
struct server_context;

// Global variables for session management  
static std::atomic<bool> should_exit{false};
static std::atomic<bool> interrupt_response{false};

// Use llama_run namespace to avoid conflicts
using Server = llama_run::Server;
using Request = llama_run::Request;
using Response = llama_run::Response;

// Server context and state - forward declare what we need
static std::unique_ptr<server_context> ctx_server;
static std::unique_ptr<Server> svr;
static std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

// Signal handlers
static void sigint_handler(int sig) {
    (void)sig;
    interrupt_response.store(true);
}

static void sigterm_handler(int sig) {
    (void)sig;
    should_exit.store(true);
}

// Helper function to extract content from chat completion response
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

// Create a chat completion request and get response directly
static std::string chat_completion(const std::string& message) {
    if (!svr) {
        return "Error: Server not initialized";
    }
    
    try {
        // Create request JSON
        json request = {
            {"model", "unknown"},
            {"messages", json::array({
                {{"role", "user"}, {"content", message}}
            })},
            {"stream", false}
        };
        
        // Call the chat completion handler directly through our shim
        auto response = svr->call_handler("POST", "/v1/chat/completions", request.dump());
        
        if (response.status != 200) {
            return "Error: Server returned status " + std::to_string(response.status) + 
                   (!response.body.empty() ? ": " + response.body : "");
        }
        
        // Parse response and extract content
        json response_json = json::parse(response.body);
        return extract_chat_content(response_json);
        
    } catch (const std::exception& e) {
        return "Error: " + std::string(e.what());
    }
}

// Check if server is ready
static bool is_server_ready() {
    if (!svr) return false;
    
    try {
        auto response = svr->call_handler("GET", "/v1/models");
        return response.status == 200;
    } catch (...) {
        return false;
    }
}

// Main interactive loop
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
        
        // Reset interrupt flag before starting request
        interrupt_response.store(false);
        
        std::cout << std::flush;
        std::string response = chat_completion(user_input);
        
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
    std::cout << "\nThis tool provides an interactive chat interface using llama.cpp server infrastructure.\n";
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

// TODO: This is where we need to implement the server initialization logic
// For now, this is a placeholder that shows the structure
static bool initialize_server(common_params& params) {
    // This would contain the server initialization logic from server.cpp
    // adapted to work with our shim layer instead of httplib::Server
    
    std::cout << "Initializing server with shim layer...\n";
    
    // Create our shim server
    svr = std::make_unique<Server>();
    
    // TODO: Set up all the handlers that are normally set up in server.cpp
    // This includes:
    // - Middleware setup
    // - Route handlers (chat completions, models, etc.)
    // - Error handlers
    // - Exception handlers
    
    // For now, just set up a basic handler
    svr->Get("/v1/models", [](const Request&, Response& res) {
        json models = {
            {"object", "list"},
            {"data", json::array({
                {{"id", "unknown"}, {"object", "model"}}
            })}
        };
        res.set_content(models.dump(), "application/json");
    });
    
    // Basic chat completion handler (placeholder)
    svr->Post("/v1/chat/completions", [](const Request& req, Response& res) {
        json response = {
            {"choices", json::array({
                {{"message", {{"role", "assistant"}, {"content", "This is a placeholder response. Server initialization not yet complete."}}}}
            })}
        };
        res.set_content(response.dump(), "application/json");
    });
    
    state.store(SERVER_STATE_READY);
    return true;
}

int main(int argc, char** argv) {
    // Parse arguments using the same system as llama-server
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
    
    // Initialize server with shim layer
    if (!initialize_server(params)) {
        std::cerr << "Failed to initialize server\n";
        return 1;
    }
    
    // Wait for server to be ready
    std::cout << "Starting server..." << std::flush;
    while (!is_server_ready() && !should_exit.load()) {
        std::cout << "." << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    if (should_exit.load()) {
        return 1;
    }
    
    std::cout << " ready!\n";
    
    // Start interactive loop
    int result = interactive_loop();
    
    // Cleanup
    llama_backend_free();
    
    return result;
}