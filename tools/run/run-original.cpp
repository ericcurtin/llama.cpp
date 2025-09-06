// Reuse server infrastructure with shim layer
#include "server-shim.hpp"

// Import server utilities
#include "../server/utils.hpp"

#include "linenoise.cpp/linenoise.h"

#include <iostream>
#include <string>
#include <atomic>

// Global variables for session management
static std::atomic<bool> should_exit{false};
static std::atomic<bool> interrupt_response{false};

// Import server infrastructure - we need to include all the server implementation
#include "../server/server.cpp"

// Signal handlers
static void sigint_handler(int sig) {
    (void)sig;
    // Set flag to interrupt response, but don't print here
    interrupt_response.store(true);
}

static int cleanup_and_exit(int exit_code) {
    if (server_pid > 0) {
        if (kill(server_pid, SIGTERM) == -1) {
            LOG_ERR("kill failed");
        }

        if (waitpid(server_pid, nullptr, 0) == -1) {
            LOG_ERR("waitpid failed");
        }
    }

    return exit_code;
}

static void sigterm_handler(int sig) {
    (void)sig;
    should_exit.store(true);
}

// Start llama-server process
static bool start_server(const std::vector<std::string> & args, int port) {
    server_pid = fork();
    
    if (server_pid == -1) {
        perror("fork failed");
        return false;
    }
    
    if (server_pid == 0) {
        // Child process - execute llama-server
        std::vector<std::string> server_args_vec;
        server_args_vec.push_back("llama-server");
        
        // Add custom port
        server_args_vec.push_back("--port");
        server_args_vec.push_back(std::to_string(port));
        
        // Add all original arguments except the program name
        for (size_t i = 1; i < args.size(); ++i) {
            // Skip any existing --port arguments to avoid conflicts
            if (args[i] == "--port") {
                i++; // Skip the port value too
                continue;
            }
            server_args_vec.push_back(args[i]);
        }
        
        // Convert to char* array for execvp
        std::vector<char*> server_args;
        for (const auto& arg : server_args_vec) {
            server_args.push_back(const_cast<char*>(arg.c_str()));
        }
        server_args.push_back(nullptr);
        
        // Try different paths for llama-server
        std::vector<std::string> server_paths = {
            "./build/bin/llama-server",
            "./llama-server",
            "llama-server"
        };
        
        for (const auto& path : server_paths) {
            execvp(path.c_str(), server_args.data());
        }
        
        perror("Failed to execute llama-server");
        exit(1);
    }
    
    return true;
}

// Wait for server to be ready, timeout is not excessive as we could be
// downloading a model
static bool wait_for_server(HttpClient & client, int max_wait_seconds = 3000) {
    std::cout << "Starting llama-server..." << std::flush;
    
    for (int i = 0; i < max_wait_seconds; ++i) {
        if (should_exit.load()) {
            return false;
        }
        
        if (client.is_server_ready()) {
            std::cout << " ready!\n";
            server_ready.store(true);
            return true;
        }
        
        std::cout << "." << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    std::cout << " timeout!\n";
    return false;
}

// Main interactive loop
static int interactive_loop(HttpClient & client) {
    std::cout << "\nChat with the model (Ctrl-D to quit, Ctrl-C to interrupt response):\n";
    
    const char* input;
    while ((input = linenoise("> ")) != nullptr && !should_exit.load()) {
        std::string user_input(input);
        linenoiseHistoryAdd(input);
        linenoiseFree(const_cast<char*>(input)); // linenoiseFree expects char*
        
        if (user_input.empty()) {
            continue;
        }
        
        // Reset interrupt flag before starting request
        interrupt_response.store(false);
        
        std::cout << std::flush;
        std::string response = client.chat_completion(user_input);
        
        if (interrupt_response.load()) {
            std::cout << "\n[Response interrupted - press Ctrl-D to quit]\n";
            interrupt_response.store(false);
        } else {
            std::cout << response << "\n\n";
        }
    }
    
    return 0;
}

static void print_usage(const char * program_name) {
    std::cout << "Usage: " << program_name << " [server-options]\n";
    std::cout << "\nThis tool starts a llama-server process and provides an interactive chat interface.\n";
    std::cout << "All options except --port are passed through to llama-server.\n";
    std::cout << "\nCommon options:\n";
    std::cout << "  -h, --help                  Show this help\n";
    std::cout << "  -m,    --model FNAME        model path (default: `models/$filename` with filename from `--hf-file`\n";
    std::cout << "                              or `--model-url` if set, otherwise models/7B/ggml-model-f16.gguf)\n";
    std::cout << "  -hf,   -hfr, --hf-repo      <user>/<model>[:quant]\n";
    std::cout << "                              Hugging Face model repository; quant is optional, case-insensitive,\n";
    std::cout << "                              default to Q4_K_M, or falls back to the first file in the repo if\n";
    std::cout << "                              Q4_K_M doesn't exist.\n";
    std::cout << "                              mmproj is also downloaded automatically if available. to disable, add\n";
    std::cout << "                              --no-mmproj\n";
    std::cout << "                              example: unsloth/phi-4-GGUF:q4_k_m\n";
    std::cout << "                              (default: unused)\n";
    std::cout << "  -c, --ctx-size N            Context size\n";
    std::cout << "  -n, --predict N             Number of tokens to predict\n";
    std::cout << "  -t, --threads N             Number of threads\n";
    std::cout << "\nFor all server options, run: llama-server --help\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Check for help
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    // Setup signal handlers
    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigterm_handler);
    
    // Convert args to vector
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
    }
    
    // Find a free port (start from 8080 and increment)
    int port = 8080;
    for (int i = 0; i < 100; ++i) {
        // Simple check if port is available by trying to bind
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock >= 0) {
            struct sockaddr_in addr;
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            addr.sin_addr.s_addr = INADDR_ANY;
            
            if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                close(sock);
                break; // Port is available
            }
            close(sock);
        }
        port++;
    }
    
    // Start server
    if (!start_server(args, port)) {
        std::cerr << "Failed to start llama-server\n";
        return 1;
    }
    
    // Create HTTP client
    HttpClient client("127.0.0.1", port);
    
    // Wait for server to be ready
    if (!wait_for_server(client)) {
        std::cerr << "Server failed to start in time\n";
        return cleanup_and_exit(1);
    }
    
    // Start interactive loop
    int result = interactive_loop(client);
    
    // Cleanup
    return cleanup_and_exit(result);
}
#else
int main(int argc, char** argv) {
    std::cerr << "Error: llama-run requires CURL support enabled.\n";
    return 1;
}
#endif
