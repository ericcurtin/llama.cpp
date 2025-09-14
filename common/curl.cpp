#include "curl.h"
#include "log.h"

#include <cstdio>
#include <cmath>
#include <chrono>
#include <thread>
#include <filesystem>
#include <regex>

#ifdef LLAMA_USE_CURL
#include <curl/curl.h>
#endif

#ifdef LLAMA_USE_CURL

bool common_curl_available() {
    return true;
}

common_curl::common_curl() : curl(nullptr), headers_list(nullptr) {
    curl = curl_easy_init();
    if (!curl) {
        LOG_ERR("%s: failed to initialize curl\n", __func__);
    }
}

common_curl::~common_curl() {
    cleanup_headers();
    if (curl) {
        curl_easy_cleanup(curl);
    }
}

common_curl::common_curl(common_curl&& other) noexcept 
    : curl(other.curl), headers_list(other.headers_list) {
    other.curl = nullptr;
    other.headers_list = nullptr;
}

common_curl& common_curl::operator=(common_curl&& other) noexcept {
    if (this != &other) {
        cleanup_headers();
        if (curl) {
            curl_easy_cleanup(curl);
        }
        
        curl = other.curl;
        headers_list = other.headers_list;
        
        other.curl = nullptr;
        other.headers_list = nullptr;
    }
    return *this;
}

void common_curl::reset() {
    if (curl) {
        curl_easy_reset(curl);
    }
    cleanup_headers();
}

void common_curl::cleanup_headers() {
    if (headers_list) {
        curl_slist_free_all(headers_list);
        headers_list = nullptr;
    }
}

void common_curl::setup_common_options(const common_curl_params& params) {
    if (!curl) return;

    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, params.follow_redirects ? 1L : 0L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, params.show_progress ? 0L : 1L);

#if defined(_WIN32)
    if (params.ssl_verify) {
        curl_easy_setopt(curl, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA);
    }
#endif

    if (params.timeout > 0) {
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, params.timeout);
    }

    if (params.max_size > 0) {
        curl_easy_setopt(curl, CURLOPT_MAXFILESIZE, params.max_size);
    }
}

void common_curl::setup_headers(const common_curl_params& params) {
    cleanup_headers();

    // Always add User-Agent
    headers_list = curl_slist_append(headers_list, "User-Agent: llama-cpp");

    // Add bearer token if provided
    if (!params.bearer_token.empty()) {
        std::string auth_header = "Authorization: Bearer " + params.bearer_token;
        headers_list = curl_slist_append(headers_list, auth_header.c_str());
    }

    // Add custom headers
    for (const auto& header : params.headers) {
        headers_list = curl_slist_append(headers_list, header.c_str());
    }

    if (headers_list) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers_list);
    }
}

CURLcode common_curl::perform_internal() {
    if (!curl) {
        return CURLE_FAILED_INIT;
    }
    return curl_easy_perform(curl);
}

size_t common_curl::header_callback(char* buffer, size_t /*size*/, size_t nitems, void* userdata) {
    auto* response = static_cast<common_curl_response*>(userdata);
    
    static std::regex header_regex("([^:]+): (.*)\r\n");
    static std::regex etag_regex("ETag", std::regex_constants::icase);
    static std::regex last_modified_regex("Last-Modified", std::regex_constants::icase);
    static std::regex accept_ranges_regex("Accept-Ranges", std::regex_constants::icase);
    
    std::string header(buffer, nitems);
    std::smatch match;
    
    if (std::regex_match(header, match, header_regex)) {
        const std::string& key = match[1];
        const std::string& value = match[2];
        
        if (std::regex_match(key, match, etag_regex)) {
            response->etag = value;
        } else if (std::regex_match(key, match, last_modified_regex)) {
            response->last_modified = value;
        } else if (std::regex_match(key, match, accept_ranges_regex)) {
            response->accept_ranges = value;
        }
    }
    
    return nitems;
}

size_t common_curl::write_callback_file(void* data, size_t size, size_t nmemb, void* userdata) {
    return std::fwrite(data, size, nmemb, static_cast<FILE*>(userdata));
}

size_t common_curl::write_callback_memory(void* data, size_t size, size_t nmemb, void* userdata) {
    auto* buffer = static_cast<std::vector<char>*>(userdata);
    buffer->insert(buffer->end(), static_cast<char*>(data), static_cast<char*>(data) + size * nmemb);
    return size * nmemb;
}

common_curl_response common_curl::head_request(const std::string& url, const common_curl_params& params) {
    common_curl_response response;
    
    if (!curl) {
        LOG_ERR("%s: curl not initialized\n", __func__);
        return response;
    }

    reset();
    setup_common_options(params);
    setup_headers(params);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);  // HEAD request
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &response);

    CURLcode res = perform_internal();
    if (res == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.http_code);
    } else {
        LOG_ERR("%s: curl_easy_perform() failed: %s\n", __func__, curl_easy_strerror(res));
    }

    return response;
}

bool common_curl::download_file(const std::string& url, const std::string& path, const common_curl_params& params) {
    if (!curl) {
        LOG_ERR("%s: curl not initialized\n", __func__);
        return false;
    }

    reset();
    setup_common_options(params);
    setup_headers(params);

    // Check if we need to resume
    size_t existing_size = 0;
    if (std::filesystem::exists(path)) {
        existing_size = std::filesystem::file_size(path);
        if (existing_size > 0) {
            curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(existing_size));
        }
    }

    // Open file for writing (append mode for resume support)
    FILE* file = fopen(path.c_str(), existing_size > 0 ? "ab" : "wb");
    if (!file) {
        LOG_ERR("%s: failed to open file for writing: %s\n", __func__, path.c_str());
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, file);

    CURLcode res = perform_internal();
    fclose(file);

    if (res != CURLE_OK) {
        LOG_ERR("%s: curl_easy_perform() failed: %s\n", __func__, curl_easy_strerror(res));
        return false;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    
    if (http_code < 200 || http_code >= 400) {
        LOG_ERR("%s: HTTP error: %ld\n", __func__, http_code);
        return false;
    }

    return true;
}

std::pair<long, std::vector<char>> common_curl::get_content(const std::string& url, const common_curl_params& params) {
    std::vector<char> buffer;
    long http_code = 0;

    if (!curl) {
        LOG_ERR("%s: curl not initialized\n", __func__);
        return {0, std::move(buffer)};
    }

    reset();
    setup_common_options(params);
    setup_headers(params);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback_memory);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);

    CURLcode res = perform_internal();
    if (res != CURLE_OK) {
        LOG_ERR("%s: curl_easy_perform() failed: %s\n", __func__, curl_easy_strerror(res));
        return {0, std::move(buffer)};
    }

    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    return {http_code, std::move(buffer)};
}

bool common_curl::perform_with_retry(const std::string& url, int max_attempts, int retry_delay_seconds, const char* method_name) {
    int remaining_attempts = max_attempts;

    while (remaining_attempts > 0) {
        LOG_INF("%s: %s %s (attempt %d of %d)...\n", __func__, method_name, url.c_str(), 
                max_attempts - remaining_attempts + 1, max_attempts);

        CURLcode res = perform_internal();
        if (res == CURLE_OK) {
            return true;
        }

        int exponential_backoff_delay = std::pow(retry_delay_seconds, max_attempts - remaining_attempts) * 1000;
        LOG_WRN("%s: curl_easy_perform() failed: %s, retrying after %d milliseconds...\n", 
                __func__, curl_easy_strerror(res), exponential_backoff_delay);

        remaining_attempts--;
        if (remaining_attempts == 0) break;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(exponential_backoff_delay));
    }

    LOG_ERR("%s: curl_easy_perform() failed after %d attempts\n", __func__, max_attempts);
    return false;
}

#else // !LLAMA_USE_CURL

bool common_curl_available() {
    return false;
}

common_curl::common_curl() {}
common_curl::~common_curl() {}
common_curl::common_curl(common_curl&& other) noexcept {}
common_curl& common_curl::operator=(common_curl&& other) noexcept { return *this; }

common_curl_response common_curl::head_request(const std::string& url, const common_curl_params& params) {
    LOG_ERR("error: built without CURL, cannot make HTTP requests\n");
    return {};
}

bool common_curl::download_file(const std::string& url, const std::string& path, const common_curl_params& params) {
    LOG_ERR("error: built without CURL, cannot download files\n");
    return false;
}

std::pair<long, std::vector<char>> common_curl::get_content(const std::string& url, const common_curl_params& params) {
    LOG_ERR("error: built without CURL, cannot get content\n");
    return {0, {}};
}

bool common_curl::perform_with_retry(const std::string& url, int max_attempts, int retry_delay_seconds, const char* method_name) {
    LOG_ERR("error: built without CURL, cannot perform HTTP requests\n");
    return false;
}

#endif // LLAMA_USE_CURL