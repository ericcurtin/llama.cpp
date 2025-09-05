#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <cerrno>
#include <cstdlib>
#include <sstream>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstring>
#include <cassert>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#if defined(LLAMA_USE_CURL)
#include <curl/curl.h>
#endif

#include "linenoise.cpp/linenoise.h"

// Global variables for process management
static pid_t server_pid = -1;
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
void sigint_handler(int sig) {
    (void)sig;
    // Set flag to interrupt response, but don't print here
    interrupt_response.store(true);
}

void cleanup_and_exit(int exit_code) {
    if (server_pid > 0) {
        kill(server_pid, SIGTERM);
        waitpid(server_pid, nullptr, 0);
    }
    exit(exit_code);
}

void sigterm_handler(int sig) {
    (void)sig;
    should_exit.store(true);
}

// Start llama-server process
bool start_server(const std::vector<std::string>& args, int port) {
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

// Wait for server to be ready
bool wait_for_server(HttpClient& client, int max_wait_seconds = 30) {
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
int interactive_loop(HttpClient& client) {
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

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [server-options] model\n";
    std::cout << "\nThis tool starts a llama-server process and provides an interactive chat interface.\n";
    std::cout << "All options except --port are passed through to llama-server.\n";
    std::cout << "\nCommon options:\n";
    std::cout << "  -h, --help                   Show this help\n";
    std::cout << "  -m MODEL                     Model path (required)\n";
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
    
#if defined(LLAMA_USE_CURL)
    curl_global_init(CURL_GLOBAL_DEFAULT);
#else
    std::cerr << "Error: This build doesn't support HTTP client functionality.\n";
    std::cerr << "Please rebuild with CURL support enabled.\n";
    return 1;
#endif
    
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
        cleanup_and_exit(1);
    }
    
    // Start interactive loop
    int result = interactive_loop(client);
    
    // Cleanup
    cleanup_and_exit(result);
    
    return 0;
}