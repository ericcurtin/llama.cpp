#include "log.h"

#include <iostream>
#include <string>
#include <vector>
#include <cerrno>
#include <cstdlib>
#include <sstream>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstring>
#include <cassert>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <process.h>
#include <io.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

#if defined(LLAMA_USE_CURL)
#include <curl/curl.h>

#include "linenoise.cpp/linenoise.h"

// Platform-specific definitions
#if defined(_WIN32)
typedef HANDLE process_handle_t;
typedef DWORD process_id_t;
#define INVALID_PROCESS_HANDLE INVALID_HANDLE_VALUE
#else
typedef pid_t process_handle_t;
typedef pid_t process_id_t;
#define INVALID_PROCESS_HANDLE -1
#endif

// Global variables for process management
static process_handle_t server_process = INVALID_PROCESS_HANDLE;
static std::atomic<bool> server_ready{false};
static std::atomic<bool> should_exit{false};
static std::atomic<bool> interrupt_response{false};

// HTTP client for communicating with llama-server
class HttpClient {
private:
    CURL* curl;
    std::string response_data;
    std::string base_url;

    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
        if (interrupt_response.load()) {
            return 0; // Stop the transfer
        }
        size_t realsize = size * nmemb;
        userp->append((char*)contents, realsize);
        return realsize;
    }

public:
    HttpClient(const std::string& host, int port) {
        curl = curl_easy_init();
        base_url = "http://" + host + ":" + std::to_string(port);
    }

    ~HttpClient() {
        if (curl) {
            curl_easy_cleanup(curl);
        }
    }

    bool is_server_ready() {
        if (!curl) return false;
        
        response_data.clear();
        // Try the models endpoint instead of health
        std::string url = base_url + "/v1/models";
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 2L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 1L);
        
        CURLcode res = curl_easy_perform(curl);
        return res == CURLE_OK && !response_data.empty();
    }

    std::string chat_completion(const std::string& message) {
        if (!curl) return "Error: HTTP client not initialized";
        
        response_data.clear();
        std::string url = base_url + "/v1/chat/completions";
        
        // Create simple JSON request string
        std::string escaped_message = escape_json_string(message);
        std::string request_str = R"({"model": "unknown", "messages": [{"role": "user", "content": ")" 
                                + escaped_message + R"("}], "stream": false})";
        
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_str.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
        
        interrupt_response.store(false);
        CURLcode res = curl_easy_perform(curl);
        
        curl_slist_free_all(headers);
        
        if (res != CURLE_OK) {
            if (res == CURLE_OPERATION_TIMEDOUT) {
                return "Error: Request timed out";
            } else if (res == CURLE_COULDNT_CONNECT) {
                return "Error: Could not connect to server";
            } else {
                return "Error: " + std::string(curl_easy_strerror(res));
            }
        }
        
        if (interrupt_response.load()) {
            return ""; // Empty response for interrupted requests
        }
        
        // Simple JSON parsing to extract content
        return extract_content_from_response(response_data);
    }

private:
    std::string escape_json_string(const std::string& input) {
        std::string result;
        for (char c : input) {
            switch (c) {
                case '"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default: result += c; break;
            }
        }
        return result;
    }
    
    std::string extract_content_from_response(const std::string& response) {
        // Simple extraction of content from JSON response
        // Look for "content":"..." pattern
        size_t content_pos = response.find("\"content\":\"");
        if (content_pos == std::string::npos) {
            return "Error: No content found in response";
        }
        
        content_pos += 11; // Skip "content":"
        size_t end_pos = content_pos;
        
        // Find the end of the content string, handling escaped quotes
        while (end_pos < response.length()) {
            if (response[end_pos] == '"' && (end_pos == content_pos || response[end_pos - 1] != '\\')) {
                break;
            }
            end_pos++;
        }
        
        if (end_pos >= response.length()) {
            return "Error: Malformed response";
        }
        
        std::string content = response.substr(content_pos, end_pos - content_pos);
        return unescape_json_string(content);
    }
    
    std::string unescape_json_string(const std::string& input) {
        std::string result;
        for (size_t i = 0; i < input.length(); ++i) {
            if (input[i] == '\\' && i + 1 < input.length()) {
                switch (input[i + 1]) {
                    case '"': result += '"'; i++; break;
                    case '\\': result += '\\'; i++; break;
                    case 'n': result += '\n'; i++; break;
                    case 'r': result += '\r'; i++; break;
                    case 't': result += '\t'; i++; break;
                    default: result += input[i]; break;
                }
            } else {
                result += input[i];
            }
        }
        return result;
    }
};

// Signal handlers
#if !defined(_WIN32)
static void sigint_handler(int sig) {
    (void)sig;
    // Set flag to interrupt response, but don't print here
    interrupt_response.store(true);
}

static void sigterm_handler(int sig) {
    (void)sig;
    should_exit.store(true);
}
#endif

static int cleanup_and_exit(int exit_code) {
#if defined(_WIN32)
    if (server_process != INVALID_PROCESS_HANDLE) {
        TerminateProcess(server_process, 0);
        WaitForSingleObject(server_process, INFINITE);
        CloseHandle(server_process);
    }
#else
    if (server_process != INVALID_PROCESS_HANDLE) {
        if (kill(server_process, SIGTERM) == -1) {
            LOG_ERR("kill failed");
        }

        if (waitpid(server_process, nullptr, 0) == -1) {
            LOG_ERR("waitpid failed");
        }
    }
#endif

    return exit_code;
}

// Start llama-server process
static bool start_server(const std::vector<std::string> & args, int port) {
#if defined(_WIN32)
    // Windows implementation using CreateProcess
    std::string command_line = "llama-server --port " + std::to_string(port);
    
    // Add all original arguments except the program name
    for (size_t i = 1; i < args.size(); ++i) {
        // Skip any existing --port arguments to avoid conflicts
        if (args[i] == "--port") {
            i++; // Skip the port value too
            continue;
        }
        command_line += " " + args[i];
    }
    
    STARTUPINFO si = {0};
    PROCESS_INFORMATION pi = {0};
    si.cb = sizeof(si);
    
    // Try to start the server process
    if (!CreateProcess(NULL, const_cast<char*>(command_line.c_str()), NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        return false;
    }
    
    server_process = pi.hProcess;
    CloseHandle(pi.hThread); // We don't need the thread handle
    return true;
#else
    // Unix implementation using fork/exec
    server_process = fork();
    
    if (server_process == -1) {
        perror("fork failed");
        return false;
    }
    
    if (server_process == 0) {
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
#endif
}

// Wait for server to be ready, timeout is not excessive as we could be
// downloading a model
static bool wait_for_server(HttpClient & client, int max_wait_seconds = 3000) {
    LOG_INF("Starting llama-server...");
    
    for (int i = 0; i < max_wait_seconds; ++i) {
        if (should_exit.load()) {
            return false;
        }
        
        if (client.is_server_ready()) {
            LOG_INF(" ready!\n");
            server_ready.store(true);
            return true;
        }
        
        LOG_INF(".");
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    LOG_INF(" timeout!\n");
    return false;
}

// Main interactive loop
static int interactive_loop(HttpClient & client) {
    LOG_INF("\nChat with the model (Ctrl-D to quit, Ctrl-C to interrupt response):\n");
    
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
        
        std::string response = client.chat_completion(user_input);
        
        if (interrupt_response.load()) {
            LOG_INF("\n[Response interrupted - press Ctrl-D to quit]\n");
            interrupt_response.store(false);
        } else {
            LOG_INF("%s\n\n", response.c_str());
        }
    }
    
    return 0;
}

static void print_usage(const char * program_name) {
    LOG_INF("Usage: %s [server-options]\n", program_name);
    LOG_INF("\nThis tool starts a llama-server process and provides an interactive chat interface.\n");
    LOG_INF("All options except --port are passed through to llama-server.\n");
    LOG_INF("\nCommon options:\n");
    LOG_INF("  -h, --help                  Show this help\n");
    LOG_INF("  -m,    --model FNAME        model path (default: `models/$filename` with filename from `--hf-file`\n");
    LOG_INF("                              or `--model-url` if set, otherwise models/7B/ggml-model-f16.gguf)\n");
    LOG_INF("  -hf,   -hfr, --hf-repo      <user>/<model>[:quant]\n");
    LOG_INF("                              Hugging Face model repository; quant is optional, case-insensitive,\n");
    LOG_INF("                              default to Q4_K_M, or falls back to the first file in the repo if\n");
    LOG_INF("                              Q4_K_M doesn't exist.\n");
    LOG_INF("                              mmproj is also downloaded automatically if available. to disable, add\n");
    LOG_INF("                              --no-mmproj\n");
    LOG_INF("                              example: unsloth/phi-4-GGUF:q4_k_m\n");
    LOG_INF("                              (default: unused)\n");
    LOG_INF("  -c, --ctx-size N            Context size\n");
    LOG_INF("  -n, --predict N             Number of tokens to predict\n");
    LOG_INF("  -t, --threads N             Number of threads\n");
    LOG_INF("\nFor all server options, run: llama-server --help\n");
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
#if !defined(_WIN32)
    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigterm_handler);
#endif
    
    // Convert args to vector
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
    }
    
    // Find a free port (start from 8080 and increment)
    int port = 8080;
    for (int i = 0; i < 100; ++i) {
        // Simple check if port is available by trying to bind
#if defined(_WIN32)
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            port++;
            continue;
        }
        
        SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock != INVALID_SOCKET) {
            struct sockaddr_in addr;
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            addr.sin_addr.s_addr = INADDR_ANY;
            
            if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                closesocket(sock);
                WSACleanup();
                break; // Port is available
            }
            closesocket(sock);
        }
        WSACleanup();
#else
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
#endif
        port++;
    }
    
    // Start server
    if (!start_server(args, port)) {
        LOG_ERR("Failed to start llama-server\n");
        return 1;
    }
    
    // Create HTTP client
    HttpClient client("127.0.0.1", port);
    
    // Wait for server to be ready
    if (!wait_for_server(client)) {
        LOG_ERR("Server failed to start in time\n");
        return cleanup_and_exit(1);
    }
    
    // Start interactive loop
    int result = interactive_loop(client);
    
    // Cleanup
    return cleanup_and_exit(result);
}
#else
int main(int argc, char** argv) {
    LOG_ERR("Error: llama-run requires CURL support enabled.\n");
    return 1;
}
#endif
