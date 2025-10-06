#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <memory>

namespace llama {

/**
 * Performance metrics collection and monitoring
 */
class performance_monitor {
public:
    struct metrics {
        // Throughput metrics
        std::atomic<uint64_t> total_tokens_processed{0};
        std::atomic<uint64_t> total_requests_processed{0};
        std::atomic<uint64_t> total_batches_processed{0};
        
        // Latency metrics (in microseconds)
        std::atomic<uint64_t> total_processing_time_us{0};
        std::atomic<uint64_t> total_queuing_time_us{0};
        std::atomic<uint64_t> total_batch_preparation_time_us{0};
        
        // Resource utilization
        std::atomic<float> avg_gpu_utilization{0.0f};
        std::atomic<float> avg_memory_utilization{0.0f};
        std::atomic<float> avg_cpu_utilization{0.0f};
        
        // Quality metrics
        std::atomic<uint64_t> cache_hits{0};
        std::atomic<uint64_t> cache_misses{0};
        std::atomic<uint64_t> memory_allocations{0};
        std::atomic<uint64_t> memory_deallocations{0};
        
        // Error tracking
        std::atomic<uint64_t> failed_requests{0};
        std::atomic<uint64_t> timeout_requests{0};
        std::atomic<uint64_t> oom_errors{0};
    };
    
    struct performance_snapshot {
        std::chrono::steady_clock::time_point timestamp;
        double tokens_per_second;
        double requests_per_second;
        double avg_latency_ms;
        double avg_queue_time_ms;
        double avg_batch_size;
        float gpu_utilization;
        float memory_utilization;
        float cache_hit_rate;
        size_t active_sequences;
        size_t pending_requests;
    };
    
    performance_monitor() : start_time_(std::chrono::steady_clock::now()) {}
    
    // Record metrics
    void record_request_completed(uint32_t tokens, uint64_t processing_time_us, uint64_t queue_time_us) {
        metrics_.total_tokens_processed += tokens;
        metrics_.total_requests_processed += 1;
        metrics_.total_processing_time_us += processing_time_us;
        metrics_.total_queuing_time_us += queue_time_us;
    }
    
    void record_batch_processed(uint32_t batch_size, uint64_t preparation_time_us) {
        metrics_.total_batches_processed += 1;
        metrics_.total_batch_preparation_time_us += preparation_time_us;
        
        std::lock_guard<std::mutex> lock(history_mutex_);
        recent_batch_sizes_.push_back(batch_size);
        if (recent_batch_sizes_.size() > 100) {
            recent_batch_sizes_.erase(recent_batch_sizes_.begin());
        }
    }
    
    void record_cache_event(bool hit) {
        if (hit) {
            metrics_.cache_hits += 1;
        } else {
            metrics_.cache_misses += 1;
        }
    }
    
    void record_memory_allocation() {
        metrics_.memory_allocations += 1;
    }
    
    void record_memory_deallocation() {
        metrics_.memory_deallocations += 1;
    }
    
    void record_error(const std::string& error_type) {
        if (error_type == "timeout") {
            metrics_.timeout_requests += 1;
        } else if (error_type == "oom") {
            metrics_.oom_errors += 1;
        } else {
            metrics_.failed_requests += 1;
        }
    }
    
    void update_resource_utilization(float gpu_util, float memory_util, float cpu_util) {
        // Simple exponential moving average
        const float alpha = 0.1f;
        float current_gpu = metrics_.avg_gpu_utilization.load();
        float current_memory = metrics_.avg_memory_utilization.load();
        float current_cpu = metrics_.avg_cpu_utilization.load();
        
        metrics_.avg_gpu_utilization = current_gpu * (1 - alpha) + gpu_util * alpha;
        metrics_.avg_memory_utilization = current_memory * (1 - alpha) + memory_util * alpha;
        metrics_.avg_cpu_utilization = current_cpu * (1 - alpha) + cpu_util * alpha;
    }
    
    // Get current performance snapshot
    performance_snapshot get_current_snapshot(size_t active_seqs, size_t pending_reqs) const {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        performance_snapshot snapshot;
        snapshot.timestamp = now;
        
        if (elapsed > 0) {
            snapshot.tokens_per_second = double(metrics_.total_tokens_processed) / elapsed;
            snapshot.requests_per_second = double(metrics_.total_requests_processed) / elapsed;
        } else {
            snapshot.tokens_per_second = 0.0;
            snapshot.requests_per_second = 0.0;
        }
        
        // Calculate average latencies
        uint64_t total_requests = metrics_.total_requests_processed;
        if (total_requests > 0) {
            snapshot.avg_latency_ms = double(metrics_.total_processing_time_us) / total_requests / 1000.0;
            snapshot.avg_queue_time_ms = double(metrics_.total_queuing_time_us) / total_requests / 1000.0;
        } else {
            snapshot.avg_latency_ms = 0.0;
            snapshot.avg_queue_time_ms = 0.0;
        }
        
        // Calculate average batch size
        {
            std::lock_guard<std::mutex> lock(history_mutex_);
            if (!recent_batch_sizes_.empty()) {
                uint64_t sum = 0;
                for (auto size : recent_batch_sizes_) {
                    sum += size;
                }
                snapshot.avg_batch_size = double(sum) / recent_batch_sizes_.size();
            } else {
                snapshot.avg_batch_size = 0.0;
            }
        }
        
        snapshot.gpu_utilization = metrics_.avg_gpu_utilization;
        snapshot.memory_utilization = metrics_.avg_memory_utilization;
        
        // Calculate cache hit rate
        uint64_t total_cache_accesses = metrics_.cache_hits + metrics_.cache_misses;
        if (total_cache_accesses > 0) {
            snapshot.cache_hit_rate = float(metrics_.cache_hits) / total_cache_accesses;
        } else {
            snapshot.cache_hit_rate = 0.0f;
        }
        
        snapshot.active_sequences = active_seqs;
        snapshot.pending_requests = pending_reqs;
        
        return snapshot;
    }
    
    // Get performance recommendations based on current metrics
    std::vector<std::string> get_optimization_recommendations() const {
        std::vector<std::string> recommendations;
        
        auto snapshot = get_current_snapshot(0, 0);
        
        // GPU utilization recommendations
        if (snapshot.gpu_utilization < 0.6f) {
            recommendations.push_back("Low GPU utilization - consider increasing batch size");
        } else if (snapshot.gpu_utilization > 0.95f) {
            recommendations.push_back("High GPU utilization - consider reducing batch size to improve latency");
        }
        
        // Memory recommendations
        if (snapshot.memory_utilization > 0.9f) {
            recommendations.push_back("High memory utilization - consider enabling memory optimization");
        }
        
        // Cache performance recommendations
        if (snapshot.cache_hit_rate < 0.8f) {
            recommendations.push_back("Low cache hit rate - consider increasing cache size or improving cache policies");
        }
        
        // Latency recommendations
        if (snapshot.avg_latency_ms > 200.0) {
            recommendations.push_back("High latency detected - consider optimizing batch composition or reducing batch size");
        }
        
        // Queue time recommendations
        if (snapshot.avg_queue_time_ms > 50.0) {
            recommendations.push_back("High queue times - consider improving request scheduling or adding more processing capacity");
        }
        
        // Batch size recommendations
        if (snapshot.avg_batch_size < 4.0) {
            recommendations.push_back("Small average batch size - consider increasing batching timeout to improve efficiency");
        } else if (snapshot.avg_batch_size > 32.0) {
            recommendations.push_back("Large average batch size - monitor for potential latency increases");
        }
        
        return recommendations;
    }
    
    // Reset all metrics
    void reset() {
        // Reset individual atomic fields rather than the whole struct
        metrics_.total_tokens_processed = 0;
        metrics_.total_requests_processed = 0;
        metrics_.total_batches_processed = 0;
        metrics_.total_processing_time_us = 0;
        metrics_.total_queuing_time_us = 0;
        metrics_.total_batch_preparation_time_us = 0;
        metrics_.avg_gpu_utilization = 0.0f;
        metrics_.avg_memory_utilization = 0.0f;
        metrics_.avg_cpu_utilization = 0.0f;
        metrics_.cache_hits = 0;
        metrics_.cache_misses = 0;
        metrics_.memory_allocations = 0;
        metrics_.memory_deallocations = 0;
        metrics_.failed_requests = 0;
        metrics_.timeout_requests = 0;
        metrics_.oom_errors = 0;
        
        start_time_ = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(history_mutex_);
        recent_batch_sizes_.clear();
    }
    
    // Get raw metrics
    const metrics& get_metrics() const {
        return metrics_;
    }

private:
    metrics metrics_;
    std::chrono::steady_clock::time_point start_time_;
    mutable std::mutex history_mutex_;
    std::vector<uint32_t> recent_batch_sizes_;
};

/**
 * RAII class for automatic performance measurement
 */
class performance_timer {
public:
    performance_timer(performance_monitor& monitor, const std::string& operation)
        : monitor_(monitor), operation_(operation) {
        start_time_ = std::chrono::steady_clock::now();
    }
    
    ~performance_timer() {
        auto end_time = std::chrono::steady_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_).count();
        
        // Record the timing based on operation type
        if (operation_ == "request_processing") {
            // This would need to be integrated with request tracking
        } else if (operation_ == "batch_preparation") {
            monitor_.record_batch_processed(1, duration_us);
        }
    }

private:
    performance_monitor& monitor_;
    std::string operation_;
    std::chrono::steady_clock::time_point start_time_;
};

// Convenience macro for performance timing
#define PERF_TIMER(monitor, operation) \
    performance_timer timer(monitor, operation)

} // namespace llama