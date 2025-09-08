#pragma once

#ifdef LLAMA_USE_CURL

#include <curl/curl.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace llama_curl {

// RAII wrapper for curl_slist
struct curl_slist_ptr {
    struct curl_slist * ptr = nullptr;
    ~curl_slist_ptr() {
        if (ptr) {
            curl_slist_free_all(ptr);
        }
    }
};

// RAII wrapper for CURL handle
using curl_ptr = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>;

// Callback function types
using WriteCallback = std::function<size_t(void * data, size_t size, size_t nmemb)>;
using HeaderCallback = std::function<size_t(char * buffer, size_t size, size_t nitems)>;
using ProgressCallback = std::function<int(curl_off_t total_download, curl_off_t now_download, curl_off_t total_upload, curl_off_t now_upload)>;

// HTTP methods
enum class HttpMethod {
    GET,
    HEAD,
    POST,
    PUT,
    DELETE
};

// Configuration for CURL requests
struct CurlConfig {
    // Basic options
    std::string url;
    HttpMethod method = HttpMethod::GET;
    std::vector<std::string> headers;
    std::string user_agent = "llama-cpp";
    
    // Authentication
    std::string bearer_token;
    
    // Timeouts and limits
    long timeout = 0;           // CURLOPT_TIMEOUT, in seconds; 0 means no timeout
    long max_filesize = 0;      // CURLOPT_MAXFILESIZE, 0 means no limit
    
    // SSL options
    bool use_native_ca = true;  // Use system certificate store
    
    // Follow redirects
    bool follow_location = true;
    
    // Progress reporting
    bool show_progress = false;
    
    // Retry configuration
    int max_retry_attempts = 3;
    int retry_delay_seconds = 2;
    
    // Write data destination
    std::string output_file;        // If set, write to file
    std::vector<char> * output_buffer = nullptr;  // If set, write to buffer
    
    // Resume support
    curl_off_t resume_from = 0; // For resuming downloads
};

// Result of a CURL operation
struct CurlResult {
    CURLcode curl_code = CURLE_OK;
    long http_code = 0;
    std::string error_message;
    bool success = false;
};

// Main CURL wrapper class
class CurlClient {
public:
    CurlClient();
    ~CurlClient() = default;
    
    // Perform HTTP request with given configuration
    CurlResult perform(const CurlConfig & config);
    
    // Convenience methods for common operations
    CurlResult get(const std::string & url, const std::vector<std::string> & headers = {});
    CurlResult head(const std::string & url, const std::vector<std::string> & headers = {});
    CurlResult download_file(const std::string & url, const std::string & output_path, 
                            const std::vector<std::string> & headers = {},
                            bool show_progress = false);
    
    // Get content as string
    std::pair<long, std::vector<char>> get_content(const std::string & url, 
                                                   const std::vector<std::string> & headers = {},
                                                   long timeout = 0, long max_size = 0);

    // Make HEAD request with header callback support
    CurlResult head_with_headers(const std::string &              url,
                                 const std::vector<std::string> & headers      = {},
                                 const std::string &              bearer_token = "",
                                 HeaderCallback                   header_cb    = nullptr);

    // Set custom callbacks
    void set_write_callback(WriteCallback callback);
    void set_header_callback(HeaderCallback callback);
    void set_progress_callback(ProgressCallback callback);
    
private:
    curl_ptr curl_;
    curl_slist_ptr headers_;
    
    // Callbacks
    WriteCallback write_callback_;
    HeaderCallback header_callback_;
    ProgressCallback progress_callback_;
    
    // Store last CURL error for better error reporting
    CURLcode last_curl_error_ = CURLE_OK;
    
    // Internal methods
    void configure_curl(const CurlConfig & config);
    void setup_headers(const std::vector<std::string> & headers, const std::string & bearer_token);
    void setup_write_function(const CurlConfig & config);
    void setup_progress_function(const CurlConfig & config);
    bool perform_with_retry(const std::string & url, int max_attempts, int retry_delay_seconds, const char * method_name);
    
    // Static callback wrappers for CURL
    static size_t write_callback_wrapper(void * data, size_t size, size_t nmemb, void * userp);
    static size_t header_callback_wrapper(char * buffer, size_t size, size_t nitems, void * userp);
    static int progress_callback_wrapper(void * clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow);
    
    // File writing helper
    static size_t write_file_callback(void * data, size_t size, size_t nmemb, void * file);
    
    // Buffer writing helper
    static size_t write_buffer_callback(void * data, size_t size, size_t nmemb, void * buffer);
};

} // namespace llama_curl

#endif // LLAMA_USE_CURL