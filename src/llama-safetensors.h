#pragma once

#include "llama-impl.h"
#include "llama-mmap.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// Safetensors file reader for llama.cpp
// Implements a minimal parser for the safetensors format
// Reference: https://huggingface.co/docs/safetensors/index

struct llama_safetensors_tensor {
    std::string dtype;
    std::vector<size_t> shape;
    size_t data_offset_start;
    size_t data_offset_end;
    
    size_t data_size() const {
        return data_offset_end - data_offset_start;
    }
    
    size_t nelements() const {
        size_t n = 1;
        for (size_t dim : shape) {
            n *= dim;
        }
        return n;
    }
};

struct llama_safetensors_metadata {
    std::map<std::string, std::string> metadata;
    std::map<std::string, llama_safetensors_tensor> tensors;
    size_t data_offset_start;
};

class llama_safetensors_file {
public:
    llama_safetensors_file(const std::string & fname);
    ~llama_safetensors_file();

    const llama_safetensors_metadata & get_metadata() const { return metadata; }
    const llama_safetensors_tensor * get_tensor(const std::string & name) const;
    
    // Read tensor data directly
    void read_tensor_data(const std::string & name, void * data, size_t size) const;
    
    // Get raw file data for memory mapping
    const uint8_t * get_data() const;
    size_t get_size() const;

private:
    llama_file file;
    llama_safetensors_metadata metadata;
    
    void parse_metadata();
};

// Utility functions
ggml_type safetensors_dtype_to_ggml_type(const std::string & dtype);
bool is_safetensors_file(const std::string & fname);