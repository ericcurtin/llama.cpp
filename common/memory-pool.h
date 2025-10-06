#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <queue>
#include <functional>

namespace llama {

/**
 * High-performance memory pool for frequently allocated objects
 * Reduces allocation overhead during high-throughput scenarios
 */
template<typename T>
class memory_pool {
public:
    explicit memory_pool(size_t initial_size = 64, size_t max_size = 1024)
        : max_pool_size(max_size) {
        std::lock_guard<std::mutex> lock(pool_mutex);
        for (size_t i = 0; i < initial_size; ++i) {
            pool.push(std::make_unique<T>());
        }
    }

    ~memory_pool() = default;

    // Get an object from the pool or allocate a new one
    std::unique_ptr<T> acquire() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (!pool.empty()) {
            auto obj = std::move(pool.front());
            pool.pop();
            return obj;
        }
        return std::make_unique<T>();
    }

    // Return an object to the pool for reuse
    void release(std::unique_ptr<T> obj) {
        if (!obj) return;
        
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (pool.size() < max_pool_size) {
            // Reset object to clean state if it has a reset method
            // Note: This would require SFINAE or static_assert in C++17
            // For now, we'll assume objects don't need explicit reset
            pool.push(std::move(obj));
        }
        // If pool is full, let the object be destroyed
    }

    // Get current pool size (for monitoring)
    size_t size() const {
        std::lock_guard<std::mutex> lock(pool_mutex);
        return pool.size();
    }

    // Clear the pool
    void clear() {
        std::lock_guard<std::mutex> lock(pool_mutex);
        while (!pool.empty()) {
            pool.pop();
        }
    }

private:
    std::queue<std::unique_ptr<T>> pool;
    mutable std::mutex pool_mutex;
    size_t max_pool_size;
};

/**
 * RAII wrapper for pooled objects
 * Automatically returns objects to pool when destroyed
 */
template<typename T>
class pooled_object {
public:
    pooled_object(std::unique_ptr<T> obj, memory_pool<T>* pool) 
        : object(std::move(obj)), pool_ptr(pool) {}

    ~pooled_object() {
        if (object && pool_ptr) {
            pool_ptr->release(std::move(object));
        }
    }

    // Move constructor
    pooled_object(pooled_object&& other) noexcept 
        : object(std::move(other.object)), pool_ptr(other.pool_ptr) {
        other.pool_ptr = nullptr;
    }

    // Move assignment
    pooled_object& operator=(pooled_object&& other) noexcept {
        if (this != &other) {
            if (object && pool_ptr) {
                pool_ptr->release(std::move(object));
            }
            object = std::move(other.object);
            pool_ptr = other.pool_ptr;
            other.pool_ptr = nullptr;
        }
        return *this;
    }

    // Disable copy
    pooled_object(const pooled_object&) = delete;
    pooled_object& operator=(const pooled_object&) = delete;

    T* operator->() { return object.get(); }
    const T* operator->() const { return object.get(); }
    T& operator*() { return *object; }
    const T& operator*() const { return *object; }
    T* get() { return object.get(); }
    const T* get() const { return object.get(); }

private:
    std::unique_ptr<T> object;
    memory_pool<T>* pool_ptr;
};

/**
 * Factory function to create pooled objects
 */
template<typename T>
pooled_object<T> make_pooled(memory_pool<T>& pool) {
    return pooled_object<T>(pool.acquire(), &pool);
}

} // namespace llama