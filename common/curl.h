#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>

#ifdef LLAMA_USE_CURL
#include <curl/curl.h>
#endif

//
// CURL wrapper class to encapsulate all curl operations
//

struct common_curl_response {
    long http_code = 0;
    std::string etag;
    std::string last_modified;
    std::string accept_ranges;
};

struct common_curl_params {
    std::vector<std::string> headers;
    std::string bearer_token;
    long timeout = 0;              // in seconds, 0 means no timeout
    long max_size = 0;             // max response size, 0 means unlimited
    bool follow_redirects = true;
    bool show_progress = false;
    bool ssl_verify = true;
};

class common_curl {
public:
    common_curl();
    ~common_curl();

    // Disable copy constructor and assignment
    common_curl(const common_curl&) = delete;
    common_curl& operator=(const common_curl&) = delete;

    // Move constructor and assignment
    common_curl(common_curl&& other) noexcept;
    common_curl& operator=(common_curl&& other) noexcept;

    // Perform HEAD request to get metadata
    common_curl_response head_request(const std::string& url, const common_curl_params& params = {});

    // Download file to path with resume support
    bool download_file(const std::string& url, const std::string& path, const common_curl_params& params = {});

    // Get content as vector of chars
    std::pair<long, std::vector<char>> get_content(const std::string& url, const common_curl_params& params = {});

    // Retry wrapper for any curl operation
    bool perform_with_retry(const std::string& url, int max_attempts = 3, int retry_delay_seconds = 2, const char* method_name = "REQUEST");

private:
#ifdef LLAMA_USE_CURL
    CURL* curl;
    struct curl_slist* headers_list;

    void reset();
    void setup_common_options(const common_curl_params& params);
    void setup_headers(const common_curl_params& params);
    void cleanup_headers();
    CURLcode perform_internal();

    // Static callback functions
    static size_t header_callback(char* buffer, size_t size, size_t nitems, void* userdata);
    static size_t write_callback_file(void* data, size_t size, size_t nmemb, void* userdata);
    static size_t write_callback_memory(void* data, size_t size, size_t nmemb, void* userdata);
#endif
};

// Check if curl is available
bool common_curl_available();