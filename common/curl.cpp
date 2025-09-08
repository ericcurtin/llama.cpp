#include "curl.h"

#ifdef LLAMA_USE_CURL

#include "log.h"

#include <curl/easy.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <thread>

namespace llama_curl {

CurlClient::CurlClient() : curl_(curl_easy_init(), &curl_easy_cleanup) {
    if (!curl_) {
        LOG_ERR("Failed to initialize CURL\n");
    }
}

CurlResult CurlClient::perform(const CurlConfig & config) {
    CurlResult result;
    
    if (!curl_) {
        result.error_message = "CURL not initialized";
        return result;
    }
    
    // Reset CURL handle for reuse
    curl_easy_reset(curl_.get());
    
    // Configure CURL based on the provided config
    configure_curl(config);
    
    // Perform the request with retry logic
    bool success = perform_with_retry(config.url, config.max_retry_attempts, 
                                     config.retry_delay_seconds, 
                                     config.method == HttpMethod::HEAD ? "HEAD" : "GET");
    
    if (success) {
        result.curl_code = CURLE_OK;
        curl_easy_getinfo(curl_.get(), CURLINFO_RESPONSE_CODE, &result.http_code);
        result.success = true;
    } else {
        // Get the last curl error from perform_with_retry
        result.curl_code = last_curl_error_;
        result.error_message = curl_easy_strerror(last_curl_error_);
        result.success = false;
    }
    
    return result;
}

CurlResult CurlClient::get(const std::string & url, const std::vector<std::string> & headers) {
    CurlConfig config;
    config.url = url;
    config.method = HttpMethod::GET;
    config.headers = headers;
    return perform(config);
}

CurlResult CurlClient::head(const std::string & url, const std::vector<std::string> & headers) {
    CurlConfig config;
    config.url = url;
    config.method = HttpMethod::HEAD;
    config.headers = headers;
    return perform(config);
}

CurlResult CurlClient::download_file(const std::string & url, const std::string & output_path, 
                                    const std::vector<std::string> & headers, bool show_progress) {
    CurlConfig config;
    config.url = url;
    config.method = HttpMethod::GET;
    config.headers = headers;
    config.output_file = output_path;
    config.show_progress = show_progress;
    return perform(config);
}

std::pair<long, std::vector<char>> CurlClient::get_content(const std::string & url, 
                                                          const std::vector<std::string> & headers,
                                                          long timeout, long max_size) {
    std::vector<char> buffer;
    CurlConfig config;
    config.url = url;
    config.method = HttpMethod::GET;
    config.headers = headers;
    config.output_buffer = &buffer;
    config.timeout = timeout;
    config.max_filesize = max_size;
    
    CurlResult result = perform(config);
    
    if (!result.success) {
        throw std::runtime_error("error: cannot make GET request: " + result.error_message);
    }
    
    return { result.http_code, std::move(buffer) };
}

void CurlClient::set_write_callback(WriteCallback callback) {
    write_callback_ = callback;
}

void CurlClient::set_header_callback(HeaderCallback callback) {
    header_callback_ = callback;
}

void CurlClient::set_progress_callback(ProgressCallback callback) {
    progress_callback_ = callback;
}

void CurlClient::configure_curl(const CurlConfig & config) {
    // Basic options
    curl_easy_setopt(curl_.get(), CURLOPT_URL, config.url.c_str());
    curl_easy_setopt(curl_.get(), CURLOPT_FOLLOWLOCATION, config.follow_location ? 1L : 0L);
    curl_easy_setopt(curl_.get(), CURLOPT_DEFAULT_PROTOCOL, "https");
    
    // Method-specific options
    switch (config.method) {
        case HttpMethod::HEAD:
            curl_easy_setopt(curl_.get(), CURLOPT_NOBODY, 1L);
            break;
        case HttpMethod::GET:
            curl_easy_setopt(curl_.get(), CURLOPT_NOBODY, 0L);
            curl_easy_setopt(curl_.get(), CURLOPT_HTTPGET, 1L);
            break;
        case HttpMethod::POST:
            curl_easy_setopt(curl_.get(), CURLOPT_POST, 1L);
            break;
        case HttpMethod::PUT:
            curl_easy_setopt(curl_.get(), CURLOPT_UPLOAD, 1L);
            break;
        case HttpMethod::DELETE:
            curl_easy_setopt(curl_.get(), CURLOPT_CUSTOMREQUEST, "DELETE");
            break;
    }
    
    // Timeouts and limits
    if (config.timeout > 0) {
        curl_easy_setopt(curl_.get(), CURLOPT_TIMEOUT, config.timeout);
    }
    if (config.max_filesize > 0) {
        curl_easy_setopt(curl_.get(), CURLOPT_MAXFILESIZE, config.max_filesize);
    }
    
    // SSL options
#if defined(_WIN32)
    if (config.use_native_ca) {
        curl_easy_setopt(curl_.get(), CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA);
    }
#endif
    
    // Setup headers
    setup_headers(config.headers, config.bearer_token);
    
    // Setup write function
    setup_write_function(config);
    
    // Setup progress function
    setup_progress_function(config);
    
    // Resume support
    if (config.resume_from > 0) {
        curl_easy_setopt(curl_.get(), CURLOPT_RESUME_FROM_LARGE, config.resume_from);
    }
}

void CurlClient::setup_headers(const std::vector<std::string> & headers, const std::string & bearer_token) {
    // Reset headers
    headers_.ptr = nullptr;
    
    // Add default user agent
    headers_.ptr = curl_slist_append(headers_.ptr, "User-Agent: llama-cpp");
    
    // Add custom headers
    for (const auto & header : headers) {
        headers_.ptr = curl_slist_append(headers_.ptr, header.c_str());
    }
    
    // Add bearer token if provided
    if (!bearer_token.empty()) {
        std::string auth_header = "Authorization: Bearer " + bearer_token;
        headers_.ptr = curl_slist_append(headers_.ptr, auth_header.c_str());
    }
    
    curl_easy_setopt(curl_.get(), CURLOPT_HTTPHEADER, headers_.ptr);
}

void CurlClient::setup_write_function(const CurlConfig & config) {
    if (!config.output_file.empty()) {
        // Write to file - Note: file handling will be done externally for now
        curl_easy_setopt(curl_.get(), CURLOPT_WRITEFUNCTION, write_file_callback);
        // Note: We'll need to handle file opening in the callback or separately
    } else if (config.output_buffer) {
        // Write to buffer
        curl_easy_setopt(curl_.get(), CURLOPT_WRITEFUNCTION, write_buffer_callback);
        curl_easy_setopt(curl_.get(), CURLOPT_WRITEDATA, config.output_buffer);
    } else if (write_callback_) {
        // Use custom callback
        curl_easy_setopt(curl_.get(), CURLOPT_WRITEFUNCTION, write_callback_wrapper);
        curl_easy_setopt(curl_.get(), CURLOPT_WRITEDATA, this);
    }
}

void CurlClient::setup_progress_function(const CurlConfig & config) {
    if (config.show_progress || progress_callback_) {
        curl_easy_setopt(curl_.get(), CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl_.get(), CURLOPT_XFERINFOFUNCTION, progress_callback_wrapper);
        curl_easy_setopt(curl_.get(), CURLOPT_XFERINFODATA, this);
    } else {
        curl_easy_setopt(curl_.get(), CURLOPT_NOPROGRESS, 1L);
    }
}

bool CurlClient::perform_with_retry(const std::string & url, int max_attempts, int retry_delay_seconds, const char * method_name) {
    int remaining_attempts = max_attempts;
    
    while (remaining_attempts > 0) {
        LOG_INF("curl_perform_with_retry: %s %s (attempt %d of %d)...\n", 
                method_name, url.c_str(), max_attempts - remaining_attempts + 1, max_attempts);
        
        CURLcode res = curl_easy_perform(curl_.get());
        last_curl_error_ = res; // Store the last error
        
        if (res == CURLE_OK) {
            return true;
        }
        
        int exponential_backoff_delay = std::pow(retry_delay_seconds, max_attempts - remaining_attempts) * 1000;
        LOG_WRN("curl_perform_with_retry: curl_easy_perform() failed: %s, retrying after %d milliseconds...\n", 
                curl_easy_strerror(res), exponential_backoff_delay);
        
        remaining_attempts--;
        if (remaining_attempts == 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(exponential_backoff_delay));
    }
    
    LOG_ERR("curl_perform_with_retry: curl_easy_perform() failed after %d attempts\n", max_attempts);
    return false;
}

// Static callback wrappers
size_t CurlClient::write_callback_wrapper(void * data, size_t size, size_t nmemb, void * userp) {
    CurlClient * client = static_cast<CurlClient *>(userp);
    if (client->write_callback_) {
        return client->write_callback_(data, size, nmemb);
    }
    return size * nmemb; // Default: just consume the data
}

// Buffer writing helper
size_t CurlClient::write_buffer_callback(void * data, size_t size, size_t nmemb, void * buffer) {
    auto data_vec = static_cast<std::vector<char> *>(buffer);
    data_vec->insert(data_vec->end(), (char *)data, (char *)data + size * nmemb);
    return size * nmemb;
}

size_t CurlClient::header_callback_wrapper(char * buffer, size_t size, size_t nitems, void * userp) {
    CurlClient * client = static_cast<CurlClient *>(userp);
    if (client->header_callback_) {
        return client->header_callback_(buffer, size, nitems);
    }
    return nitems; // Default: just consume the headers
}

int CurlClient::progress_callback_wrapper(void * clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) {
    CurlClient * client = static_cast<CurlClient *>(clientp);
    if (client->progress_callback_) {
        return client->progress_callback_(dltotal, dlnow, ultotal, ulnow);
    }
    return 0; // Continue download
}

size_t CurlClient::write_file_callback(void * data, size_t size, size_t nmemb, void * file) {
    return fwrite(data, size, nmemb, static_cast<FILE *>(file));
}

} // namespace llama_curl

#endif // LLAMA_USE_CURL