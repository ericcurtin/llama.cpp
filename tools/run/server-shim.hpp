#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <memory>

namespace llama_run {

// Minimal shim structures to mimic httplib interface without actually creating sockets
struct Request {
    std::string method;
    std::string path;
    std::string body;
    std::unordered_map<std::string, std::string> headers;
    
    std::string get_header_value(const std::string& key) const {
        auto it = headers.find(key);
        return it != headers.end() ? it->second : "";
    }

    std::function<bool()> is_connection_closed = []() { return false; };
};

struct Response {
    int status = 200;
    std::string body;
    std::unordered_map<std::string, std::string> headers;
    
    void set_content(const std::string& content, const std::string& content_type) {
        body = content;
        set_header("Content-Type", content_type);
    }
    
    void set_header(const std::string& key, const std::string& value) {
        headers[key] = value;
    }
};

// Shim class that mimics httplib::Server interface
class Server {
public:
    using Handler = std::function<void(const Request&, Response&)>;
    
    enum class HandlerResponse {
        Handled,
        Unhandled
    };
    
    using PreRoutingHandler = std::function<HandlerResponse(const Request&, Response&)>;
    using ErrorHandler = std::function<void(const Request&, Response&)>;
    using ExceptionHandler = std::function<void(const Request&, Response&, std::exception_ptr)>;
    using Logger = std::function<void(const Request&, const Response&)>;

private:
    std::unordered_map<std::string, Handler> handlers;
    PreRoutingHandler pre_routing_handler;
    ErrorHandler error_handler;
    ExceptionHandler exception_handler;
    Logger logger;
    long read_timeout = 0;
    long write_timeout = 0;
    int address_family = 0;

public:
    // Mimic httplib::Server interface
    Server& Get(const std::string& pattern, Handler handler) {
        handlers["GET " + pattern] = handler;
        return *this;
    }
    
    Server& Post(const std::string& pattern, Handler handler) {
        handlers["POST " + pattern] = handler;
        return *this;
    }
    
    // No-op methods for socket-related operations
    bool bind_to_port(const std::string&, int) { return true; }
    int bind_to_any_port(const std::string&) { return 8080; }
    void listen_after_bind() { /* no-op */ }
    void wait_until_ready() { /* no-op */ }
    void set_address_family(int family) { address_family = family; }
    void set_read_timeout(long timeout) { read_timeout = timeout; }
    void set_write_timeout(long timeout) { write_timeout = timeout; }
    
    void set_pre_routing_handler(PreRoutingHandler handler) {
        pre_routing_handler = handler;
    }
    
    void set_error_handler(ErrorHandler handler) {
        error_handler = handler;
    }
    
    void set_exception_handler(ExceptionHandler handler) {
        exception_handler = handler;
    }
    
    void set_logger(Logger log) {
        logger = log;
    }
    
    // Method to call handlers directly (our main interface)
    Response call_handler(const std::string& method, const std::string& path, 
                         const std::string& body = "", 
                         const std::unordered_map<std::string, std::string>& headers = {}) {
        Request req;
        req.method = method;
        req.path = path;
        req.body = body;
        req.headers = headers;
        
        Response res;
        
        try {
            // Call pre-routing handler if set
            if (pre_routing_handler) {
                auto result = pre_routing_handler(req, res);
                if (result == HandlerResponse::Handled) {
                    return res;
                }
            }
            
            // Find and call the appropriate handler
            std::string key = method + " " + path;
            auto it = handlers.find(key);
            if (it != handlers.end()) {
                it->second(req, res);
            } else {
                // No handler found
                res.status = 404;
                if (error_handler) {
                    error_handler(req, res);
                }
            }
            
            // Call logger if set
            if (logger) {
                logger(req, res);
            }
            
        } catch (...) {
            if (exception_handler) {
                exception_handler(req, res, std::current_exception());
            } else {
                res.status = 500;
                res.body = "Internal Server Error";
            }
        }
        
        return res;
    }
};

} // namespace llama_run