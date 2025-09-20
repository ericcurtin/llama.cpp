#include "common.h"
#include "log.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <io.h>
#else
#    include <sys/file.h>
#    include <sys/ioctl.h>
#    include <unistd.h>
#endif

#if defined(LLAMA_USE_CURL)
#    include <curl/curl.h>
#endif

#include <signal.h>

#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

GGML_ATTRIBUTE_FORMAT(1, 2)
static int printe(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const int ret = vfprintf(stderr, fmt, args);
    va_end(args);

    return ret;
}

static std::string strftime_fmt(const char * fmt, const std::tm & tm) {
    std::ostringstream oss;
    oss << std::put_time(&tm, fmt);

    return oss.str();
}

// Forward declarations for helper functions and classes from run.cpp
struct progress_data {
    size_t                                file_size  = 0;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    bool                                  printed    = false;
};

static int get_terminal_width() {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return csbi.srWindow.Right - csbi.srWindow.Left + 1;
#else
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return w.ws_col;
#endif
}

class File {
  public:
    FILE * file = nullptr;

    FILE * open(const std::string & filename, const char * mode) {
        file = ggml_fopen(filename.c_str(), mode);

        return file;
    }

    int lock() {
        if (file) {
#    ifdef _WIN32
            fd    = _fileno(file);
            hFile = (HANDLE) _get_osfhandle(fd);
            if (hFile == INVALID_HANDLE_VALUE) {
                fd = -1;

                return 1;
            }

            OVERLAPPED overlapped = {};
            if (!LockFileEx(hFile, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, MAXDWORD, MAXDWORD,
                            &overlapped)) {
                fd = -1;

                return 1;
            }
#    else
            fd = fileno(file);
            if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
                fd = -1;

                return 1;
            }
#    endif
        }

        return 0;
    }

    std::string to_string() {
        fseek(file, 0, SEEK_END);
        const size_t size = ftell(file);
        fseek(file, 0, SEEK_SET);
        std::string out;
        out.resize(size);
        const size_t read_size = fread(&out[0], 1, size, file);
        if (read_size != size) {
            printe("Error reading file: %s", strerror(errno));
        }

        return out;
    }

    ~File() {
        if (fd >= 0) {
#    ifdef _WIN32
            if (hFile != INVALID_HANDLE_VALUE) {
                OVERLAPPED overlapped = {};
                UnlockFileEx(hFile, 0, MAXDWORD, MAXDWORD, &overlapped);
            }
#    else
            flock(fd, LOCK_UN);
#    endif
        }

        if (file) {
            fclose(file);
        }
    }

  private:
    int fd = -1;
#    ifdef _WIN32
    HANDLE hFile = nullptr;
#    endif
};

#ifdef LLAMA_USE_CURL
class HttpClient {
  public:
    int init(const std::string & url, const std::vector<std::string> & headers, const std::string & output_file,
             const bool progress, std::string * response_str = nullptr) {
        if (std::filesystem::exists(output_file)) {
            return 0;
        }

        std::string output_file_partial;

        if (!output_file.empty()) {
            output_file_partial = output_file + ".partial";
        }

        if (download(url, headers, output_file_partial, progress, response_str)) {
            return 1;
        }

        if (!output_file.empty()) {
            try {
                std::filesystem::rename(output_file_partial, output_file);
            } catch (const std::filesystem::filesystem_error & e) {
                printe("Failed to rename '%s' to '%s': %s\n", output_file_partial.c_str(), output_file.c_str(), e.what());
                return 1;
            }
        }

        return 0;
    }

    ~HttpClient() {
        if (chunk) {
            curl_slist_free_all(chunk);
        }

        if (curl) {
            curl_easy_cleanup(curl);
        }
    }

  private:
    CURL *              curl  = nullptr;
    struct curl_slist * chunk = nullptr;

    int download(const std::string & url, const std::vector<std::string> & headers, const std::string & output_file,
             const bool progress, std::string * response_str = nullptr) {
        curl = curl_easy_init();
        if (!curl) {
            return 1;
        }

        progress_data data;
        File          out;
        if (!output_file.empty()) {
            if (!out.open(output_file, "ab")) {
                printe("Failed to open file for writing\n");

                return 1;
            }

            if (out.lock()) {
                printe("Failed to exclusively lock file\n");

                return 1;
            }
        }

        set_write_options(response_str, out);
        data.file_size = set_resume_point(output_file);
        set_progress_options(progress, data);
        set_headers(headers);
        CURLcode res = perform(url);
        if (res != CURLE_OK){
            printe("Fetching resource '%s' failed: %s\n", url.c_str(), curl_easy_strerror(res));
            return 1;
        }

        return 0;
    }

    void set_write_options(std::string * response_str, const File & out) {
        if (response_str) {
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, capture_data);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, response_str);
        } else {
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, out.file);
        }
    }

    size_t set_resume_point(const std::string & output_file) {
        size_t file_size = 0;
        if (std::filesystem::exists(output_file)) {
            file_size = std::filesystem::file_size(output_file);
            curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(file_size));
        }

        return file_size;
    }

    void set_progress_options(bool progress, progress_data & data) {
        if (progress) {
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &data);
            curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, update_progress);
        }
    }

    void set_headers(const std::vector<std::string> & headers) {
        if (!headers.empty()) {
            if (chunk) {
                curl_slist_free_all(chunk);
                chunk = 0;
            }

            for (const auto & header : headers) {
                chunk = curl_slist_append(chunk, header.c_str());
            }

            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, chunk);
        }
    }

    CURLcode perform(const std::string & url) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_DEFAULT_PROTOCOL, "https");
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
#ifdef _WIN32
        curl_easy_setopt(curl, CURLOPT_SSL_OPTIONS, CURLSSLOPT_NATIVE_CA);
#endif
        return curl_easy_perform(curl);
    }

    static std::string human_readable_time(double seconds) {
        int hrs  = static_cast<int>(seconds) / 3600;
        int mins = (static_cast<int>(seconds) % 3600) / 60;
        int secs = static_cast<int>(seconds) % 60;

        if (hrs > 0) {
            return string_format("%dh %02dm %02ds", hrs, mins, secs);
        } else if (mins > 0) {
            return string_format("%dm %02ds", mins, secs);
        } else {
            return string_format("%ds", secs);
        }
    }

    static std::string human_readable_size(curl_off_t size) {
        static const char * suffix[] = { "B", "KB", "MB", "GB", "TB" };
        char                length   = sizeof(suffix) / sizeof(suffix[0]);
        int                 i        = 0;
        double              dbl_size = size;
        if (size > 1024) {
            for (i = 0; (size / 1024) > 0 && i < length - 1; i++, size /= 1024) {
                dbl_size = size / 1024.0;
            }
        }

        return string_format("%.2f %s", dbl_size, suffix[i]);
    }

    static int update_progress(void * ptr, curl_off_t total_to_download, curl_off_t now_downloaded, curl_off_t,
                               curl_off_t) {
        progress_data * data = static_cast<progress_data *>(ptr);
        if (total_to_download <= 0) {
            return 0;
        }

        total_to_download += data->file_size;
        const curl_off_t now_downloaded_plus_file_size = now_downloaded + data->file_size;
        const curl_off_t percentage      = calculate_percentage(now_downloaded_plus_file_size, total_to_download);
        std::string      progress_prefix = generate_progress_prefix(percentage);

        const double speed = calculate_speed(now_downloaded, data->start_time);
        const double tim   = (total_to_download - now_downloaded) / speed;
        std::string  progress_suffix =
            generate_progress_suffix(now_downloaded_plus_file_size, total_to_download, speed, tim);

        int         progress_bar_width = calculate_progress_bar_width(progress_prefix, progress_suffix);
        std::string progress_bar;
        generate_progress_bar(progress_bar_width, percentage, progress_bar);

        print_progress(progress_prefix, progress_bar, progress_suffix);
        data->printed = true;

        return 0;
    }

    static curl_off_t calculate_percentage(curl_off_t now_downloaded_plus_file_size, curl_off_t total_to_download) {
        return (now_downloaded_plus_file_size * 100) / total_to_download;
    }

    static std::string generate_progress_prefix(curl_off_t percentage) {
        return string_format("%3ld%% |", static_cast<long int>(percentage));
    }

    static double calculate_speed(curl_off_t now_downloaded, const std::chrono::steady_clock::time_point & start_time) {
        const auto                          now             = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed_seconds = now - start_time;

        return elapsed_seconds.count() > 0 ? static_cast<double>(now_downloaded) / elapsed_seconds.count() : 0;
    }

    static int calculate_progress_bar_width(const std::string & progress_prefix, const std::string & progress_suffix) {
        const int     terminal_width      = get_terminal_width();
        const int     available_width     = terminal_width - progress_prefix.length() - progress_suffix.length();
        constexpr int min_progress_width  = 10;
        constexpr int progress_bar_border = 3;  // For "| |"

        return std::max(min_progress_width, available_width - progress_bar_border);
    }

    static std::string generate_progress_suffix(curl_off_t now_downloaded_plus_file_size, curl_off_t total_to_download,
                                                double speed, double estimated_time) {
        return string_format("| %s/%s (%s/s, %s)", human_readable_size(now_downloaded_plus_file_size).c_str(),
                             human_readable_size(total_to_download).c_str(), human_readable_size(speed).c_str(),
                             human_readable_time(estimated_time).c_str());
    }

    static std::string generate_progress_bar(int progress_bar_width, curl_off_t percentage,
                                             std::string & progress_bar) {
        const curl_off_t pos = (percentage * progress_bar_width) / 100;
        for (int i = 0; i < progress_bar_width; ++i) {
            progress_bar.append((i < pos) ? "█" : " ");
        }

        return progress_bar;
    }

    static void print_progress(const std::string & progress_prefix, const std::string & progress_bar,
                               const std::string & progress_suffix) {
        printe("\r" LOG_CLR_TO_EOL "%s%s| %s", progress_prefix.c_str(), progress_bar.c_str(), progress_suffix.c_str());
    }
    // Function to write data to a file
    static size_t write_data(void * ptr, size_t size, size_t nmemb, void * stream) {
        FILE * out = static_cast<FILE *>(stream);
        return fwrite(ptr, size, nmemb, out);
    }

    // Function to capture data to a string
    static size_t capture_data(void * ptr, size_t size, size_t nmemb, void * userdata) {
        size_t total_size = size * nmemb;
        std::string * response_str = static_cast<std::string *>(userdata);
        response_str->append(static_cast<char *>(ptr), total_size);
        return total_size;
    }
};
#endif

class ModelDownloader {
  public:
#ifdef LLAMA_USE_CURL
    int download(const std::string & url, const std::string & output_file, const bool progress,
                 const std::vector<std::string> & headers = {}, std::string * response_str = nullptr) {
        HttpClient http;
        if (http.init(url, headers, output_file, progress, response_str)) {
            return 1;
        }

        return 0;
    }
#else
    int download(const std::string &, const std::string &, const bool, const std::vector<std::string> & = {},
                 std::string * = nullptr) {
        printe("%s: llama.cpp built without libcurl, downloading from an url not supported.\n", __func__);

        return 1;
    }
#endif

    // Helper function to handle model tag extraction and URL construction
    std::pair<std::string, std::string> extract_model_and_tag(std::string & model, const std::string & base_url) {
        std::string  model_tag = "latest";
        const size_t colon_pos = model.find(':');
        if (colon_pos != std::string::npos) {
            model_tag = model.substr(colon_pos + 1);
            model     = model.substr(0, colon_pos);
        }

        std::string url = base_url + model + "/manifests/" + model_tag;

        return { model, url };
    }

    // Helper function to download and parse the manifest
    int download_and_parse_manifest(const std::string & url, const std::vector<std::string> & headers,
                                    nlohmann::json & manifest) {
        std::string manifest_str;
        int         ret = download(url, "", false, headers, &manifest_str);
        if (ret) {
            return ret;
        }

        manifest = nlohmann::json::parse(manifest_str);

        return 0;
    }

    std::string get_model_endpoint() {
        const char * hf_endpoint = std::getenv("HF_ENDPOINT");
        if (hf_endpoint) {
            return std::string(hf_endpoint);
        }

        return "https://huggingface.co/";
    }

    int dl_from_endpoint(std::string & model_endpoint, std::string & model, const std::string & bn) {
        // Find the second occurrence of '/' after protocol string
        size_t pos = model.find('/');
        pos        = model.find('/', pos + 1);
        std::string              hfr, hff;
        std::vector<std::string> headers = { "User-Agent: llama-cpp", "Accept: application/json" };
        std::string              url;

        if (pos == std::string::npos) {
            auto [model_name, manifest_url] = extract_model_and_tag(model, model_endpoint + "v2/");
            hfr                             = model_name;

            nlohmann::json manifest;
            int            ret = download_and_parse_manifest(manifest_url, headers, manifest);
            if (ret) {
                return ret;
            }

            hff = manifest["ggufFile"]["rfilename"];
        } else {
            hfr = model.substr(0, pos);
            hff = model.substr(pos + 1);
        }

        url = model_endpoint + hfr + "/resolve/main/" + hff;

        return download(url, bn, true, headers);
    }

    int huggingface_dl(std::string & model, const std::string & bn) {
        std::string model_endpoint = get_model_endpoint();
        return dl_from_endpoint(model_endpoint, model, bn);
    }

    int ollama_dl(std::string & model, const std::string & bn) {
        const std::vector<std::string> headers = { "Accept: application/vnd.docker.distribution.manifest.v2+json" };
        if (model.find('/') == std::string::npos) {
            model = "library/" + model;
        }

        auto [model_name, manifest_url] = extract_model_and_tag(model, "https://registry.ollama.ai/v2/");
        nlohmann::json manifest;
        int            ret = download_and_parse_manifest(manifest_url, {}, manifest);
        if (ret) {
            return ret;
        }

        std::string layer;
        for (const auto & l : manifest["layers"]) {
            if (l["mediaType"] == "application/vnd.ollama.image.model") {
                layer = l["digest"];
                break;
            }
        }

        std::string blob_url = "https://registry.ollama.ai/v2/" + model_name + "/blobs/" + layer;

        return download(blob_url, bn, true, headers);
    }

    std::string basename(const std::string & path) {
        const size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos) {
            return path;
        }

        return path.substr(pos + 1);
    }

    int rm_until_substring(std::string & model_, const std::string & substring) {
        const std::string::size_type pos = model_.find(substring);
        if (pos == std::string::npos) {
            return 1;
        }

        model_ = model_.substr(pos + substring.size());  // Skip past the substring
        return 0;
    }

    int pull_huggingface(std::string model) {
        rm_until_substring(model, "hf.co/");
        rm_until_substring(model, "://");
        const std::string bn = basename(model);
        printf("Pulling HuggingFace model: %s\n", model.c_str());
        printf("Output file: %s\n", bn.c_str());
        int ret = huggingface_dl(model, bn);
        if (ret == 0) {
            printf("Successfully downloaded: %s\n", bn.c_str());
        } else {
            printe("Failed to download HuggingFace model: %s\n", model.c_str());
        }
        return ret;
    }

    int pull_docker_registry(std::string model) {
        rm_until_substring(model, "ollama.com/library/");
        rm_until_substring(model, "://");
        const std::string bn = basename(model);
        printf("Pulling Docker Registry model: %s\n", model.c_str());
        printf("Output file: %s\n", bn.c_str());
        int ret = ollama_dl(model, bn);
        if (ret == 0) {
            printf("Successfully downloaded: %s\n", bn.c_str());
        } else {
            printe("Failed to download Docker Registry model: %s\n", model.c_str());
        }
        return ret;
    }
};

static void print_usage() {
    printf(
        "Usage: llama-pull [options] model\n"
        "\n"
        "Options:\n"
        "  -hf, --huggingface     Pull model from HuggingFace\n"
        "  -dr, --docker-registry Pull model from Docker Registry (Ollama)\n"
        "  -h,  --help           Show this help message\n"
        "\n"
        "Examples:\n"
        "  llama-pull -hf microsoft/DialoGPT-medium\n"
        "  llama-pull -hf QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q2_K.gguf\n"
        "  llama-pull -dr llama3\n"
        "  llama-pull -dr granite-code:8b\n"
        "\n"
    );
}

int main(int argc, const char ** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    bool hf_mode = false;
    bool dr_mode = false;
    std::string model;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "-hf" || arg == "--huggingface") {
            hf_mode = true;
        } else if (arg == "-dr" || arg == "--docker-registry") {
            dr_mode = true;
        } else {
            model = arg;
        }
    }

    if (!hf_mode && !dr_mode) {
        printe("Error: Must specify either -hf or -dr\n");
        print_usage();
        return 1;
    }

    if (hf_mode && dr_mode) {
        printe("Error: Cannot specify both -hf and -dr\n");
        print_usage();
        return 1;
    }

    if (model.empty()) {
        printe("Error: No model specified\n");
        print_usage();
        return 1;
    }

    ModelDownloader downloader;

    if (hf_mode) {
        return downloader.pull_huggingface(model);
    } else if (dr_mode) {
        return downloader.pull_docker_registry(model);
    }

    return 1;
}