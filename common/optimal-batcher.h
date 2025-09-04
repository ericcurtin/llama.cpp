#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
#include <unordered_map>

namespace llama {

/**
 * Metadata for optimized batch construction
 */
struct batch_candidate {
    int seq_id;
    size_t token_count;
    size_t context_length;
    bool is_prompt;  // vs continuation
    float compute_intensity;  // estimated compute cost
    int priority;
    std::chrono::steady_clock::time_point arrival_time;
    
    batch_candidate(int id, size_t tokens, size_t ctx_len, bool prompt, int prio)
        : seq_id(id), token_count(tokens), context_length(ctx_len), 
          is_prompt(prompt), priority(prio) {
        arrival_time = std::chrono::steady_clock::now();
        // Estimate compute intensity (prompt processing is more expensive)
        compute_intensity = is_prompt ? tokens * 2.0f : tokens * 1.0f;
    }
};

/**
 * Advanced batching strategy for optimal GPU utilization
 */
class optimal_batcher {
public:
    struct config {
        size_t max_batch_size;
        size_t max_tokens_per_batch;
        std::chrono::milliseconds max_wait_time;
        float prompt_continuation_ratio;  // Max 30% prompts in batch
        bool enable_dynamic_batching;
        bool prioritize_fairness;
        
        config() : max_batch_size(64), max_tokens_per_batch(4096), 
                  max_wait_time(50), prompt_continuation_ratio(0.3f),
                  enable_dynamic_batching(true), prioritize_fairness(true) {}
    };
    
    explicit optimal_batcher(const config& cfg = config()) : config_(cfg) {}
    
    /**
     * Create optimal batch from candidates using multiple strategies
     */
    std::vector<int> create_optimal_batch(std::vector<batch_candidate>& candidates) {
        if (candidates.empty()) return {};
        
        // Sort candidates for optimal batching
        if (config_.prioritize_fairness) {
            sort_by_fairness(candidates);
        } else {
            sort_by_efficiency(candidates);
        }
        
        std::vector<int> batch;
        batch.reserve(config_.max_batch_size);
        
        size_t total_tokens = 0;
        size_t prompt_count = 0;
        size_t continuation_count = 0;
        float total_compute = 0.0f;
        
        auto now = std::chrono::steady_clock::now();
        
        for (const auto& candidate : candidates) {
            if (batch.size() >= config_.max_batch_size) break;
            
            // Check token limit
            if (total_tokens + candidate.token_count > config_.max_tokens_per_batch) {
                // Try to fit smaller candidates
                continue;
            }
            
            // Check prompt/continuation ratio
            size_t would_be_prompts = prompt_count + (candidate.is_prompt ? 1 : 0);
            float prompt_ratio = float(would_be_prompts) / (batch.size() + 1);
            
            if (candidate.is_prompt && prompt_ratio > config_.prompt_continuation_ratio) {
                continue;  // Too many prompts in batch
            }
            
            // Check if candidate has been waiting too long (fairness)
            auto wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - candidate.arrival_time);
            bool urgent = wait_time > config_.max_wait_time;
            
            // Apply batching strategy
            bool should_include = should_include_in_batch(
                candidate, batch, total_tokens, total_compute, urgent
            );
            
            if (should_include) {
                batch.push_back(candidate.seq_id);
                total_tokens += candidate.token_count;
                total_compute += candidate.compute_intensity;
                
                if (candidate.is_prompt) {
                    prompt_count++;
                } else {
                    continuation_count++;
                }
            }
        }
        
        return batch;
    }
    
    /**
     * Predict optimal batch size based on current GPU memory and compute capacity
     */
    size_t predict_optimal_batch_size(size_t available_memory_mb, 
                                     size_t avg_sequence_length) const {
        // Simplified heuristic - in practice this would consider:
        // - GPU memory constraints
        // - Model size and attention mechanism
        // - Current GPU utilization
        // - Historical performance data
        
        size_t memory_constrained_size = available_memory_mb / (avg_sequence_length * 4); // rough estimate
        size_t compute_optimal_size = std::min(config_.max_batch_size, size_t(64));
        
        return std::min(memory_constrained_size, compute_optimal_size);
    }
    
    /**
     * Dynamic adjustment of batch parameters based on performance feedback
     */
    void update_performance_feedback(float throughput_tokens_per_sec,
                                   float gpu_utilization,
                                   float avg_latency_ms) {
        performance_history_.push_back({throughput_tokens_per_sec, gpu_utilization, avg_latency_ms});
        
        // Keep only recent history
        if (performance_history_.size() > 100) {
            performance_history_.erase(performance_history_.begin());
        }
        
        // Adapt parameters based on performance
        if (config_.enable_dynamic_batching) {
            adapt_batch_parameters();
        }
    }

private:
    struct performance_sample {
        float throughput;
        float gpu_util;
        float latency;
    };
    
    config config_;
    std::vector<performance_sample> performance_history_;
    
    void sort_by_fairness(std::vector<batch_candidate>& candidates) {
        // Sort by arrival time (FIFO) with priority boost
        std::sort(candidates.begin(), candidates.end(), 
                 [](const batch_candidate& a, const batch_candidate& b) {
                     if (a.priority != b.priority) {
                         return a.priority < b.priority;  // Higher priority first
                     }
                     return a.arrival_time < b.arrival_time;  // FIFO within priority
                 });
    }
    
    void sort_by_efficiency(std::vector<batch_candidate>& candidates) {
        // Sort by compute efficiency and batching compatibility
        std::sort(candidates.begin(), candidates.end(),
                 [](const batch_candidate& a, const batch_candidate& b) {
                     // Group by type (prompt vs continuation) for better batching
                     if (a.is_prompt != b.is_prompt) {
                         return !a.is_prompt;  // Continuations first for better batching
                     }
                     
                     // Within same type, prefer higher compute intensity
                     return a.compute_intensity > b.compute_intensity;
                 });
    }
    
    bool should_include_in_batch(const batch_candidate& candidate,
                                const std::vector<int>& current_batch,
                                size_t current_tokens,
                                float current_compute,
                                bool urgent) const {
        if (urgent) return true;  // Always include urgent requests
        
        // Check if adding this candidate improves batch efficiency
        float new_compute = current_compute + candidate.compute_intensity;
        float compute_density = new_compute / (current_batch.size() + 1);
        
        // Prefer batches with similar compute intensity for better GPU utilization
        if (!current_batch.empty()) {
            float avg_compute = current_compute / current_batch.size();
            float intensity_ratio = candidate.compute_intensity / avg_compute;
            
            // Avoid mixing very different compute intensities
            if (intensity_ratio < 0.5f || intensity_ratio > 2.0f) {
                return false;
            }
        }
        
        return true;
    }
    
    void adapt_batch_parameters() {
        if (performance_history_.size() < 10) return;
        
        // Calculate recent average performance
        float avg_throughput = 0.0f;
        float avg_gpu_util = 0.0f;
        float avg_latency = 0.0f;
        
        size_t recent_samples = std::min(size_t(10), performance_history_.size());
        for (size_t i = performance_history_.size() - recent_samples; 
             i < performance_history_.size(); ++i) {
            avg_throughput += performance_history_[i].throughput;
            avg_gpu_util += performance_history_[i].gpu_util;
            avg_latency += performance_history_[i].latency;
        }
        
        avg_throughput /= recent_samples;
        avg_gpu_util /= recent_samples;
        avg_latency /= recent_samples;
        
        // Adaptive adjustments
        if (avg_gpu_util < 0.7f && avg_latency < 100.0f) {
            // Low GPU utilization, can increase batch size
            config_.max_batch_size = std::min(config_.max_batch_size + 4, size_t(128));
        } else if (avg_latency > 200.0f) {
            // High latency, reduce batch size
            config_.max_batch_size = std::max(config_.max_batch_size - 2, size_t(8));
        }
        
        // Adjust wait time based on throughput
        if (avg_throughput < 1000.0f) {
            // Low throughput, be more aggressive with batching
            config_.max_wait_time = std::chrono::milliseconds(
                std::min(config_.max_wait_time.count() + 10, 100L)
            );
        }
    }
};

} // namespace llama