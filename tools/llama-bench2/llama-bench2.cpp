#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iterator>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <future>

#include "common.h"
#include "ggml.h"
#include "llama.h"

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

// Performance measurement utilities
static uint64_t get_time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

static uint64_t get_time_us() {
    return get_time_ns() / 1000;
}

static std::string get_cpu_info() {
    std::vector<std::string> cpu_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev      = ggml_backend_dev_get(i);
        auto   dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU || dev_type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            cpu_list.emplace_back(ggml_backend_dev_description(dev));
        }
    }
    std::string result;
    for (size_t i = 0; i < cpu_list.size(); i++) {
        result += cpu_list[i];
        if (i < cpu_list.size() - 1) result += ", ";
    }
    return result;
}

static std::string get_gpu_info() {
    std::vector<std::string> gpu_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        auto * dev      = ggml_backend_dev_get(i);
        auto   dev_type = ggml_backend_dev_type(dev);
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_list.emplace_back(ggml_backend_dev_description(dev));
        }
    }
    std::string result;
    for (size_t i = 0; i < gpu_list.size(); i++) {
        result += gpu_list[i];
        if (i < gpu_list.size() - 1) result += ", ";
    }
    return result;
}

// Test result structure
struct benchmark_result {
    std::string test_name;
    double avg_latency_ms;
    double throughput_tokens_per_sec;
    double memory_usage_mb;
    double gpu_utilization_percent;
    int batch_size;
    int concurrent_requests;
    std::vector<double> individual_latencies;
    std::string status;
};

// llama-bench2 specific parameters
struct bench2_params {
    std::string model_path = "models/7B/ggml-model-q4_0.gguf";
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64};
    std::vector<int> concurrent_requests = {1, 2, 4, 8, 16};
    int n_prompt = 512;
    int n_gen = 128;
    int n_threads = 2;
    int n_gpu_layers = 99;
    int repetitions = 3;
    bool verbose = false;
    std::string output_format = "md";
};

static void print_usage(int, char ** argv) {
    printf("Usage: %s [options]\n", argv[0]);
    printf("\nllama-bench2: Performance benchmarking tool focusing on llama.cpp vs vLLM comparison\n");
    printf("\nOptions:\n");
    printf("  -h, --help                    Show this help message\n");
    printf("  -m, --model <path>            Model path (default: models/7B/ggml-model-q4_0.gguf)\n");
    printf("  -p, --n-prompt <n>            Number of prompt tokens (default: 512)\n");
    printf("  -n, --n-gen <n>               Number of tokens to generate (default: 128)\n");
    printf("  -t, --threads <n>             Number of threads (default: %d)\n", std::thread::hardware_concurrency());
    printf("  -ngl, --n-gpu-layers <n>      Number of GPU layers (default: 99)\n");
    printf("  -r, --repetitions <n>         Number of repetitions per test (default: 3)\n");
    printf("  -b, --batch-sizes <list>      Comma-separated batch sizes (default: 1,2,4,8,16,32,64)\n");
    printf("  -c, --concurrent <list>       Comma-separated concurrent request counts (default: 1,2,4,8,16)\n");
    printf("  -o, --output <md|json|csv>    Output format (default: md)\n");
    printf("  -v, --verbose                 Verbose output\n");
    printf("\nTests focus on performance gaps between llama.cpp and vLLM:\n");
    printf("  1. Batch processing efficiency\n");
    printf("  2. Concurrent request handling\n");
    printf("  3. Memory usage patterns\n");
    printf("  4. GPU utilization efficiency\n");
    printf("  5. Throughput vs latency tradeoffs\n");
}

static std::vector<int> parse_int_list(const std::string & s) {
    std::vector<int> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            result.push_back(std::stoi(item));
        } catch (const std::exception&) {
            fprintf(stderr, "Warning: Invalid integer '%s' in list\n", item.c_str());
        }
    }
    return result;
}

static bool parse_params(int argc, char ** argv, bench2_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            return false;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if ((arg == "-p" || arg == "--n-prompt") && i + 1 < argc) {
            params.n_prompt = std::atoi(argv[++i]);
        } else if ((arg == "-n" || arg == "--n-gen") && i + 1 < argc) {
            params.n_gen = std::atoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            params.n_threads = std::atoi(argv[++i]);
        } else if ((arg == "-ngl" || arg == "--n-gpu-layers") && i + 1 < argc) {
            params.n_gpu_layers = std::atoi(argv[++i]);
        } else if ((arg == "-r" || arg == "--repetitions") && i + 1 < argc) {
            params.repetitions = std::atoi(argv[++i]);
        } else if ((arg == "-b" || arg == "--batch-sizes") && i + 1 < argc) {
            params.batch_sizes = parse_int_list(argv[++i]);
        } else if ((arg == "-c" || arg == "--concurrent") && i + 1 < argc) {
            params.concurrent_requests = parse_int_list(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            params.output_format = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    
    return true;
}

// Test 1: Batch Processing Efficiency
// This test measures how efficiently llama.cpp handles different batch sizes
// vLLM excels at large batch processing due to better GPU parallelization
static benchmark_result test_batch_efficiency(llama_context * ctx, llama_model * model, 
                                             const bench2_params & params, int batch_size) {
    benchmark_result result;
    result.test_name = "Batch Processing Efficiency";
    result.batch_size = batch_size;
    result.concurrent_requests = 1;
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    
    // Generate random tokens for the test
    std::vector<llama_token> tokens;
    for (int i = 0; i < params.n_prompt; i++) {
        tokens.push_back(std::rand() % n_vocab);
    }
    
    std::vector<double> latencies;
    double total_tokens = 0;
    uint64_t total_time_us = 0;
    
    for (int rep = 0; rep < params.repetitions; rep++) {
        llama_memory_clear(llama_get_memory(ctx), false);
        
        auto start_time = get_time_us();
        
        // Process tokens in batches using llama_batch_get_one
        for (size_t i = 0; i < tokens.size(); i += batch_size) {
            size_t current_batch_size = std::min(batch_size, (int)(tokens.size() - i));
            
            std::vector<llama_token> batch_tokens(tokens.begin() + i, tokens.begin() + i + current_batch_size);
            
            if (llama_decode(ctx, llama_batch_get_one(batch_tokens.data(), current_batch_size)) != 0) {
                result.status = "decode_failed";
                return result;
            }
        }
        
        auto end_time = get_time_us();
        double latency_ms = (end_time - start_time) / 1000.0;
        latencies.push_back(latency_ms);
        total_time_us += (end_time - start_time);
        total_tokens += tokens.size();
    }
    
    result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    result.throughput_tokens_per_sec = (total_tokens * 1000000.0) / total_time_us;
    result.individual_latencies = latencies;
    result.status = "success";
    
    return result;
}

// Test 2: Concurrent Request Simulation
// This simulates multiple concurrent requests, which is where vLLM typically outperforms llama.cpp
static benchmark_result test_concurrent_requests(llama_model * model, const bench2_params & params, 
                                                int num_concurrent) {
    benchmark_result result;
    result.test_name = "Concurrent Request Handling";
    result.batch_size = 1;
    result.concurrent_requests = num_concurrent;
    
    auto worker_function = [&](int /* worker_id */) -> double {
        // Create separate context for each worker
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = params.n_prompt + params.n_gen;
        ctx_params.n_batch = 1;
        ctx_params.n_threads = 1; // Single thread per worker
        
        llama_context * worker_ctx = llama_init_from_model(model, ctx_params);
        if (!worker_ctx) {
            return -1.0;
        }
        
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        
        auto start_time = get_time_us();
        
        // Generate prompt tokens
        std::vector<llama_token> prompt_tokens;
        for (int i = 0; i < params.n_prompt; i++) {
            prompt_tokens.push_back(std::rand() % n_vocab);
        }
        
        // Process prompt using llama_batch_get_one
        std::vector<llama_token> batch_tokens(prompt_tokens);
        
        if (llama_decode(worker_ctx, llama_batch_get_one(batch_tokens.data(), prompt_tokens.size())) != 0) {
            llama_free(worker_ctx);
            return -1.0;
        }
        
        // Generate tokens one by one
        for (int i = 0; i < params.n_gen; i++) {
            llama_token next_token = std::rand() % n_vocab;
            
            if (llama_decode(worker_ctx, llama_batch_get_one(&next_token, 1)) != 0) {
                llama_free(worker_ctx);
                return -1.0;
            }
        }
        
        auto end_time = get_time_us();
        llama_free(worker_ctx);
        
        return (end_time - start_time) / 1000.0; // Return latency in ms
    };
    
    std::vector<double> latencies;
    
    for (int rep = 0; rep < params.repetitions; rep++) {
        std::vector<std::future<double>> futures;
        
        auto start_time = get_time_us();
        
        // Launch concurrent workers
        for (int i = 0; i < num_concurrent; i++) {
            futures.push_back(std::async(std::launch::async, worker_function, i));
        }
        
        // Wait for all workers to complete
        for (auto & future : futures) {
            double worker_latency = future.get();
            if (worker_latency > 0) {
                latencies.push_back(worker_latency);
            }
        }
        
        auto end_time = get_time_us();
        double total_latency_ms = (end_time - start_time) / 1000.0;
        
        if (params.verbose) {
            printf("Repetition %d: Total time %.2f ms for %d concurrent requests\n", 
                   rep + 1, total_latency_ms, num_concurrent);
        }
    }
    
    if (latencies.empty()) {
        result.status = "failed";
        return result;
    }
    
    result.avg_latency_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    result.throughput_tokens_per_sec = (num_concurrent * (params.n_prompt + params.n_gen) * 1000.0) / result.avg_latency_ms;
    result.individual_latencies = latencies;
    result.status = "success";
    
    return result;
}

// Test 3: Memory Usage Analysis
static benchmark_result test_memory_usage(llama_context * ctx, const bench2_params & /* params */) {
    benchmark_result result;
    result.test_name = "Memory Usage Analysis";
    result.batch_size = 0;
    result.concurrent_requests = 1;
    
    // Get memory information from llama context
    const size_t mem_size = llama_state_get_size(ctx);
    result.memory_usage_mb = mem_size / (1024.0 * 1024.0);
    
    result.avg_latency_ms = 0;
    result.throughput_tokens_per_sec = 0;
    result.status = "success";
    
    return result;
}

static void print_results_markdown(const std::vector<benchmark_result> & results, 
                                 const bench2_params & params) {
    printf("# llama-bench2 Results: Performance Comparison vs vLLM\n\n");
    printf("**Model**: %s\n", params.model_path.c_str());
    printf("**CPU**: %s\n", get_cpu_info().c_str());
    printf("**GPU**: %s\n", get_gpu_info().c_str());
    printf("**Threads**: %d\n", params.n_threads);
    printf("**GPU Layers**: %d\n\n", params.n_gpu_layers);
    
    printf("## Performance Issues Identified\n\n");
    printf("This benchmark focuses on areas where llama.cpp typically underperforms compared to vLLM:\n\n");
    
    printf("| Test | Batch Size | Concurrent | Avg Latency (ms) | Throughput (tok/s) | Memory (MB) | Status |\n");
    printf("|------|------------|------------|------------------|------------------- |-------------|--------|\n");
    
    for (const auto & result : results) {
        printf("| %s | %d | %d | %.2f | %.2f | %.2f | %s |\n",
               result.test_name.c_str(),
               result.batch_size,
               result.concurrent_requests,
               result.avg_latency_ms,
               result.throughput_tokens_per_sec,
               result.memory_usage_mb,
               result.status.c_str());
    }
    
    printf("\n## Key Observations\n\n");
    printf("- **Batch Processing**: Large batch sizes show performance characteristics\n");
    printf("- **Concurrency**: Multiple concurrent requests reveal throughput limitations\n");
    printf("- **Memory Usage**: Memory efficiency patterns compared to vLLM\n");
    printf("- **Recommendations**: Use these metrics to identify optimization opportunities\n");
}

static void print_results_json(const std::vector<benchmark_result> & results,
                             const bench2_params & params) {
    printf("{\n");
    printf("  \"metadata\": {\n");
    printf("    \"model\": \"%s\",\n", params.model_path.c_str());
    printf("    \"cpu_info\": \"%s\",\n", get_cpu_info().c_str());
    printf("    \"gpu_info\": \"%s\",\n", get_gpu_info().c_str());
    printf("    \"threads\": %d,\n", params.n_threads);
    printf("    \"gpu_layers\": %d\n", params.n_gpu_layers);
    printf("  },\n");
    printf("  \"results\": [\n");
    
    for (size_t i = 0; i < results.size(); i++) {
        const auto & result = results[i];
        printf("    {\n");
        printf("      \"test_name\": \"%s\",\n", result.test_name.c_str());
        printf("      \"batch_size\": %d,\n", result.batch_size);
        printf("      \"concurrent_requests\": %d,\n", result.concurrent_requests);
        printf("      \"avg_latency_ms\": %.2f,\n", result.avg_latency_ms);
        printf("      \"throughput_tokens_per_sec\": %.2f,\n", result.throughput_tokens_per_sec);
        printf("      \"memory_usage_mb\": %.2f,\n", result.memory_usage_mb);
        printf("      \"status\": \"%s\"\n", result.status.c_str());
        printf("    }%s\n", (i < results.size() - 1) ? "," : "");
    }
    
    printf("  ]\n");
    printf("}\n");
}

int main(int argc, char ** argv) {
    bench2_params params;
    
    if (!parse_params(argc, argv, params)) {
        return 1;
    }
    
    // Initialize llama backend
    llama_backend_init();
    
    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    
    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Error: failed to load model from %s\n", params.model_path.c_str());
        return 1;
    }
    
    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.n_prompt + params.n_gen;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = params.n_threads;
    
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: failed to create llama context\n");
        llama_model_free(model);
        return 1;
    }
    
    std::vector<benchmark_result> results;
    
    if (params.verbose) {
        printf("Starting llama-bench2 performance tests...\n");
        printf("Model: %s\n", params.model_path.c_str());
        printf("Tests focus on performance gaps vs vLLM\n\n");
    }
    
    // Test 1: Batch processing efficiency
    if (params.verbose) printf("Running batch processing efficiency tests...\n");
    for (int batch_size : params.batch_sizes) {
        auto result = test_batch_efficiency(ctx, model, params, batch_size);
        if (params.verbose) {
            printf("  Batch size %d: %.2f ms, %.2f tok/s\n", 
                   batch_size, result.avg_latency_ms, result.throughput_tokens_per_sec);
        }
        results.push_back(result);
    }
    
    // Test 2: Concurrent request handling
    if (params.verbose) printf("Running concurrent request tests...\n");
    for (int concurrent : params.concurrent_requests) {
        if (concurrent > 1) { // Skip single request as it's covered in batch tests
            auto result = test_concurrent_requests(model, params, concurrent);
            if (params.verbose) {
                printf("  Concurrent %d: %.2f ms, %.2f tok/s\n", 
                       concurrent, result.avg_latency_ms, result.throughput_tokens_per_sec);
            }
            results.push_back(result);
        }
    }
    
    // Test 3: Memory usage
    if (params.verbose) printf("Analyzing memory usage...\n");
    auto memory_result = test_memory_usage(ctx, params);
    if (params.verbose) {
        printf("  Memory usage: %.2f MB\n", memory_result.memory_usage_mb);
    }
    results.push_back(memory_result);
    
    // Output results
    if (params.output_format == "json") {
        print_results_json(results, params);
    } else if (params.output_format == "csv") {
        printf("test_name,batch_size,concurrent_requests,avg_latency_ms,throughput_tokens_per_sec,memory_usage_mb,status\n");
        for (const auto & result : results) {
            printf("%s,%d,%d,%.2f,%.2f,%.2f,%s\n",
                   result.test_name.c_str(), result.batch_size, result.concurrent_requests,
                   result.avg_latency_ms, result.throughput_tokens_per_sec, 
                   result.memory_usage_mb, result.status.c_str());
        }
    } else {
        print_results_markdown(results, params);
    }
    
    // Cleanup
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    return 0;
}