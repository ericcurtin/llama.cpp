# llama-run Implementation Comparison

## Original Implementation (run-original.cpp)
- **Architecture**: Fork llama-server process + HTTP client communication  
- **Dependencies**: libcurl for HTTP requests, socket programming
- **Code duplication**: Custom argument parsing, custom error handling
- **Network**: Requires finding free port, binding sockets, HTTP protocol
- **Complexity**: Process management, signal handling, HTTP request/response parsing

### Key Components:
- `HttpClient` class with libcurl
- Process forking and exec of llama-server
- Port finding and socket binding logic
- HTTP request/response JSON parsing
- Custom signal handlers for process cleanup

## New Implementation (run.cpp)  
- **Architecture**: Direct llama.cpp API usage with server shim layer
- **Dependencies**: Only llama.cpp core APIs
- **Code sharing**: Uses `common_params_parse()`, `common_init_from_params()`
- **Network**: No sockets, ports, or HTTP - direct function calls
- **Complexity**: Simple API usage following llama.cpp patterns

### Key Components:
- `server-shim.hpp` - Defines Server interface (currently minimal)
- Direct use of `llama_tokenize()`, `llama_decode()`, `llama_sampler_sample()`
- Shared argument parsing from `common/arg.cpp`
- Simplified initialization using `common_init_from_params()`

## Benefits of New Approach

### 1. **Maximized Code Sharing**
- ✅ Same argument parsing as llama-server (`LLAMA_EXAMPLE_SERVER`)
- ✅ Same initialization logic (`common_init_from_params()`)
- ✅ Same error handling patterns
- ✅ Same system info reporting

### 2. **Simplified Architecture**  
- ❌ No process forking/management
- ❌ No HTTP client/server communication
- ❌ No port binding or socket programming
- ❌ No libcurl dependency
- ✅ Direct API calls for better performance

### 3. **Better for Downstream Projects**
- Shows how to use llama.cpp APIs directly
- No network dependencies
- Works in environments where port binding is not allowed
- Easier to integrate into existing applications

### 4. **Maintainability**
- Fewer dependencies to manage
- Shared code means consistent behavior with llama-server
- Simpler debugging (no inter-process communication)
- Less surface area for bugs

## Shim Layer Concept

The `server-shim.hpp` provides the foundation for the server interface approach mentioned in the problem statement:

```cpp
class Server {
public:
    using Handler = std::function<void(const Request&, Response&)>;
    
    Server& Get(const std::string& pattern, Handler handler);
    Server& Post(const std::string& pattern, Handler handler);
    
    // No-op methods for socket operations
    bool bind_to_port(const std::string&, int) { return true; }
    
    // Direct handler invocation
    Response call_handler(const std::string& method, const std::string& path, ...);
};
```

This allows future expansion to fully reuse llama-server's handler logic while maintaining the no-socket approach.