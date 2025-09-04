#pragma once

#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <unordered_set>
#include <memory>

namespace llama {

/**
 * Priority levels for server tasks
 */
enum class task_priority : int {
    HIGH = 0,      // Completion requests
    MEDIUM = 1,    // Embedding requests  
    LOW = 2,       // Background tasks
    BATCH = 3      // Batched operations
};

/**
 * Enhanced task with priority and scheduling metadata
 */
struct enhanced_task {
    int id;
    task_priority priority;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point deadline;
    size_t estimated_tokens;
    size_t batch_affinity;  // For grouping compatible tasks
    void* data;  // Original task data
    
    enhanced_task(int task_id, task_priority prio, void* task_data, 
                  size_t tokens = 0, size_t affinity = 0) 
        : id(task_id), priority(prio), estimated_tokens(tokens), 
          batch_affinity(affinity), data(task_data) {
        created_at = std::chrono::steady_clock::now();
        // Set deadline based on priority
        auto timeout = std::chrono::milliseconds(
            priority == task_priority::HIGH ? 5000 :
            priority == task_priority::MEDIUM ? 10000 : 30000
        );
        deadline = created_at + timeout;
    }
    
    // For priority queue ordering (lower priority value = higher priority)
    bool operator<(const enhanced_task& other) const {
        if (priority != other.priority) {
            return static_cast<int>(priority) > static_cast<int>(other.priority);
        }
        // Within same priority, prefer tasks that can be batched together
        if (batch_affinity != other.batch_affinity) {
            return batch_affinity < other.batch_affinity;
        }
        // Finally, FIFO within same priority and affinity
        return created_at > other.created_at;
    }
    
    bool is_expired() const {
        return std::chrono::steady_clock::now() > deadline;
    }
};

/**
 * High-performance task scheduler with priority queues and batching optimization
 */
class enhanced_scheduler {
public:
    explicit enhanced_scheduler(size_t max_batch_size = 32, 
                               std::chrono::milliseconds batch_timeout = std::chrono::milliseconds(10))
        : max_batch_size_(max_batch_size), batch_timeout_(batch_timeout), running_(true) {}
    
    ~enhanced_scheduler() {
        stop();
    }
    
    // Add a task to the scheduler
    void enqueue(enhanced_task task) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Remove expired tasks to keep queue clean
        cleanup_expired_tasks();
        
        task_queue_.push(std::move(task));
        lock.unlock();
        queue_cv_.notify_one();
    }
    
    // Get the next batch of tasks for processing
    // Returns empty vector if no tasks available within timeout
    std::vector<enhanced_task> get_batch(std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for tasks or timeout
        if (task_queue_.empty()) {
            queue_cv_.wait_for(lock, timeout, [this] { 
                return !task_queue_.empty() || !running_; 
            });
        }
        
        if (!running_) {
            return {};
        }
        
        std::vector<enhanced_task> batch;
        auto now = std::chrono::steady_clock::now();
        
        // Strategy: Build optimal batch based on priority and affinity
        if (!task_queue_.empty()) {
            auto first_task = std::move(const_cast<enhanced_task&>(task_queue_.top()));
            task_queue_.pop();
            
            if (!first_task.is_expired()) {
                batch.push_back(std::move(first_task));
                
                // Try to add compatible tasks to the batch
                std::vector<enhanced_task> remaining_tasks;
                
                while (!task_queue_.empty() && batch.size() < max_batch_size_) {
                    auto candidate = std::move(const_cast<enhanced_task&>(task_queue_.top()));
                    task_queue_.pop();
                    
                    if (candidate.is_expired()) {
                        continue; // Skip expired tasks
                    }
                    
                    // Check if candidate is compatible with batch
                    bool compatible = true;
                    if (!batch.empty()) {
                        const auto& first = batch[0];
                        // Same priority and affinity are preferred for batching
                        compatible = (candidate.priority == first.priority) &&
                                   (candidate.batch_affinity == first.batch_affinity);
                    }
                    
                    if (compatible) {
                        batch.push_back(std::move(candidate));
                    } else {
                        remaining_tasks.push_back(std::move(candidate));
                    }
                }
                
                // Put non-compatible tasks back in queue
                for (auto& task : remaining_tasks) {
                    task_queue_.push(std::move(task));
                }
            }
        }
        
        return batch;
    }
    
    // Get a single high-priority task immediately
    std::unique_ptr<enhanced_task> get_urgent() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        if (task_queue_.empty()) {
            return nullptr;
        }
        
        auto task = std::move(const_cast<enhanced_task&>(task_queue_.top()));
        task_queue_.pop();
        
        if (task.is_expired()) {
            return nullptr;
        }
        
        // Only return if it's high priority
        if (task.priority == task_priority::HIGH) {
            return std::make_unique<enhanced_task>(std::move(task));
        }
        
        // Put it back if not urgent
        task_queue_.push(std::move(task));
        return nullptr;
    }
    
    // Get current queue size
    size_t size() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return task_queue_.size();
    }
    
    // Stop the scheduler
    void stop() {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        running_ = false;
        queue_cv_.notify_all();
    }
    
    // Get statistics
    struct stats {
        size_t total_tasks;
        size_t high_priority_tasks;
        size_t expired_tasks;
        double avg_wait_time_ms;
    };
    
    stats get_stats() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        stats result = {};
        result.total_tasks = task_queue_.size();
        
        // Count by priority (would need to iterate through queue)
        // This is expensive, so implement only if needed for monitoring
        
        return result;
    }

private:
    void cleanup_expired_tasks() {
        // Remove expired tasks from the front of queue
        std::vector<enhanced_task> valid_tasks;
        
        while (!task_queue_.empty()) {
            auto task = std::move(const_cast<enhanced_task&>(task_queue_.top()));
            task_queue_.pop();
            
            if (!task.is_expired()) {
                valid_tasks.push_back(std::move(task));
            }
        }
        
        // Put valid tasks back
        for (auto& task : valid_tasks) {
            task_queue_.push(std::move(task));
        }
    }
    
    std::priority_queue<enhanced_task> task_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    size_t max_batch_size_;
    std::chrono::milliseconds batch_timeout_;
    bool running_;
};

} // namespace llama