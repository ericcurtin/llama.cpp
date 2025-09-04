#include "llama-safetensors.h"
#include "llama-io.h"

#include "../vendor/nlohmann/json.hpp"

#include <cstring>
#include <stdexcept>

using json = nlohmann::json;

llama_safetensors_file::llama_safetensors_file(const std::string & fname) : file(fname.c_str(), "rb") {
    parse_metadata();
}

llama_safetensors_file::~llama_safetensors_file() = default;

void llama_safetensors_file::parse_metadata() {
    // Read the first 8 bytes to get metadata length
    uint64_t metadata_len;
    file.read_raw(&metadata_len, sizeof(metadata_len));
    
    // Safetensors uses little-endian
    // TODO: handle endianness conversion if needed
    
    if (metadata_len > 100 * 1024 * 1024) { // 100MB sanity check
        throw std::runtime_error("Safetensors metadata too large: " + std::to_string(metadata_len));
    }
    
    // Read metadata JSON
    std::vector<char> metadata_buf(metadata_len);
    file.read_raw(metadata_buf.data(), metadata_len);
    
    std::string metadata_str(metadata_buf.begin(), metadata_buf.end());
    json metadata_json = json::parse(metadata_str);
    
    // Calculate data start offset (header + metadata)
    metadata.data_offset_start = 8 + metadata_len;
    
    // Parse tensors and metadata
    for (auto & [key, value] : metadata_json.items()) {
        if (key == "__metadata__") {
            // Store general metadata
            for (auto & [meta_key, meta_value] : value.items()) {
                if (meta_value.is_string()) {
                    metadata.metadata[meta_key] = meta_value.get<std::string>();
                }
            }
        } else {
            // Parse tensor metadata
            llama_safetensors_tensor tensor;
            
            if (!value.contains("dtype") || !value.contains("shape") || !value.contains("data_offsets")) {
                throw std::runtime_error("Invalid tensor metadata for: " + key);
            }
            
            tensor.dtype = value["dtype"].get<std::string>();
            tensor.shape = value["shape"].get<std::vector<size_t>>();
            
            auto data_offsets = value["data_offsets"].get<std::vector<size_t>>();
            if (data_offsets.size() != 2) {
                throw std::runtime_error("Invalid data_offsets for tensor: " + key);
            }
            
            tensor.data_offset_start = metadata.data_offset_start + data_offsets[0];
            tensor.data_offset_end = metadata.data_offset_start + data_offsets[1];
            
            metadata.tensors[key] = tensor;
        }
    }
}

const llama_safetensors_tensor * llama_safetensors_file::get_tensor(const std::string & name) const {
    auto it = metadata.tensors.find(name);
    return it != metadata.tensors.end() ? &it->second : nullptr;
}

void llama_safetensors_file::read_tensor_data(const std::string & name, void * data, size_t size) const {
    const auto * tensor = get_tensor(name);
    if (!tensor) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    if (size != tensor->data_size()) {
        throw std::runtime_error("Size mismatch for tensor " + name + 
                                ": expected " + std::to_string(tensor->data_size()) + 
                                ", got " + std::to_string(size));
    }
    
    file.seek(tensor->data_offset_start, SEEK_SET);
    file.read_raw(data, size);
}

const uint8_t * llama_safetensors_file::get_data() const {
    // This method should only be used with memory mapping, 
    // for now just throw an error
    throw std::runtime_error("get_data() not implemented for safetensors - use read_tensor_data() instead");
}

size_t llama_safetensors_file::get_size() const {
    return file.size();
}

ggml_type safetensors_dtype_to_ggml_type(const std::string & dtype) {
    // Map safetensors dtypes to ggml types
    if (dtype == "F32") return GGML_TYPE_F32;
    if (dtype == "F16") return GGML_TYPE_F16;
    if (dtype == "BF16") return GGML_TYPE_BF16;
    if (dtype == "I32") return GGML_TYPE_I32;
    if (dtype == "I16") return GGML_TYPE_I16;
    if (dtype == "I8") return GGML_TYPE_I8;
    if (dtype == "U32") return GGML_TYPE_I32; // Map to signed for now
    if (dtype == "U16") return GGML_TYPE_I16; // Map to signed for now  
    if (dtype == "U8") return GGML_TYPE_I8;   // Map to signed for now
    
    throw std::runtime_error("Unsupported safetensors dtype: " + dtype);
}

bool is_safetensors_file(const std::string & fname) {
    try {
        llama_file file(fname.c_str(), "rb");
        
        // Read first 8 bytes for metadata length
        uint64_t metadata_len;
        file.read_raw(&metadata_len, sizeof(metadata_len));
        
        // Basic sanity checks
        if (metadata_len == 0 || metadata_len > 100 * 1024 * 1024) {
            return false;
        }
        
        // Check if file is large enough to contain the metadata
        if (file.size() < 8 + metadata_len) {
            return false;
        }
        
        // Try to read and parse a small part of the metadata as JSON
        std::vector<char> sample_buf(std::min(metadata_len, static_cast<uint64_t>(1024)));
        file.read_raw(sample_buf.data(), sample_buf.size());
        
        std::string sample_str(sample_buf.begin(), sample_buf.end());
        
        // Check if it starts like JSON metadata
        return sample_str.find('{') == 0 && sample_str.find('"') != std::string::npos;
        
    } catch (...) {
        return false;
    }
}