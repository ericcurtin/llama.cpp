#pragma once

// Essential server components extracted for reuse
#include "../server/utils.hpp"

// Forward declare the server components we need
struct server_context;

// Essential handler type signatures
using chat_completion_handler = std::function<void(const httplib::Request&, httplib::Response&)>;
using models_handler = std::function<void(const httplib::Request&, httplib::Response&)>;

// Extract essential server setup functions
namespace llama_run_server {
    
    // Initialize server context with given params
    bool init_server_context(server_context*& ctx, const common_params& params);
    
    // Create the essential handlers we need for chat interface
    chat_completion_handler create_chat_completion_handler(server_context* ctx);
    models_handler create_models_handler(server_context* ctx);
    
    // Helper functions for response formatting
    void format_error_response(httplib::Response& res, const std::string& error, int code = 500);
    void format_json_response(httplib::Response& res, const json& data);
    
    // Clean up server context
    void cleanup_server_context(server_context* ctx);
}