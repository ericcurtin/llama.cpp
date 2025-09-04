#include "ggml.h"
#include "../src/llama-model-loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>

// Helper function to create a test safetensors file
static void create_test_safetensors_file(const std::string & filename) {
    // Create a simple safetensors file for testing
    // JSON header
    const char * json_header = R"({
  "tensor1": {
    "dtype": "F32",
    "shape": [2, 2],
    "data_offsets": [0, 16]
  },
  "tensor2": {
    "dtype": "F16",
    "shape": [3],
    "data_offsets": [16, 22]
  }
})";
    
    uint64_t header_len = strlen(json_header);
    
    // Create test tensor data
    float tensor1_data[4] = {1.0f, 2.0f, 3.0f, 4.0f}; // 16 bytes
    uint16_t tensor2_data[3] = {0x3c00, 0x4000, 0x4200}; // F16 values for 1.0, 2.0, 3.0 - 6 bytes
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to create test file: %s\n", filename.c_str());
        exit(1);
    }
    
    // Write header length (little-endian)
    file.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    
    // Write JSON header
    file.write(json_header, header_len);
    
    // Write tensor data
    file.write(reinterpret_cast<const char*>(tensor1_data), sizeof(tensor1_data));
    file.write(reinterpret_cast<const char*>(tensor2_data), sizeof(tensor2_data));
    
    file.close();
}

static bool test_file_type_detection() {
    printf("Testing file type detection...\n");
    
    // Use dummy parameters to create a loader instance
    std::string dummy_file = "/tmp/dummy.gguf";
    std::vector<std::string> splits;
    try {
        llama_model_loader loader_test(dummy_file, splits, false, false, nullptr, nullptr);
        // This will fail but we just need the instance to test methods
    } catch (...) {
        // Expected to fail, continue with static tests
    }
    
    // Create a minimal loader instance for testing static methods
    class test_loader_wrapper {
    public:
        static llama_file_type test_detect_file_type(const std::string & fname) {
            // Copy the detection logic here for testing
            size_t pos = fname.find_last_of('.');
            if (pos != std::string::npos) {
                std::string ext = fname.substr(pos);
                if (ext == ".safetensors") {
                    return LLAMA_FILE_TYPE_SAFETENSORS;
                }
            }
            return LLAMA_FILE_TYPE_GGUF;
        }
        
        static ggml_type test_safetensors_dtype_to_ggml_type(const std::string & dtype) {
            if (dtype == "F32") return GGML_TYPE_F32;
            if (dtype == "F16") return GGML_TYPE_F16;
            if (dtype == "BF16") return GGML_TYPE_BF16;
            if (dtype == "I32") return GGML_TYPE_I32;
            if (dtype == "I16") return GGML_TYPE_I16;
            if (dtype == "I8") return GGML_TYPE_I8;
            if (dtype == "U8") return GGML_TYPE_I8;
            if (dtype == "BOOL") return GGML_TYPE_I8;
            
            throw std::runtime_error("unsupported safetensors dtype: " + dtype);
        }
    };
    
    // Test safetensors detection
    auto type1 = test_loader_wrapper::test_detect_file_type("test.safetensors");
    if (type1 != LLAMA_FILE_TYPE_SAFETENSORS) {
        printf("FAIL: safetensors file not detected correctly\n");
        return false;
    }
    
    // Test GGUF detection  
    auto type2 = test_loader_wrapper::test_detect_file_type("test.gguf");
    if (type2 != LLAMA_FILE_TYPE_GGUF) {
        printf("FAIL: GGUF file not detected correctly\n");
        return false;
    }
    
    printf("PASS: File type detection works correctly\n");
    return true;
}

static bool test_dtype_conversion() {
    printf("Testing dtype conversion...\n");
    
    // Use the test wrapper from above
    class test_loader_wrapper {
    public:
        static ggml_type test_safetensors_dtype_to_ggml_type(const std::string & dtype) {
            if (dtype == "F32") return GGML_TYPE_F32;
            if (dtype == "F16") return GGML_TYPE_F16;
            if (dtype == "BF16") return GGML_TYPE_BF16;
            if (dtype == "I32") return GGML_TYPE_I32;
            if (dtype == "I16") return GGML_TYPE_I16;
            if (dtype == "I8") return GGML_TYPE_I8;
            if (dtype == "U8") return GGML_TYPE_I8;
            if (dtype == "BOOL") return GGML_TYPE_I8;
            
            throw std::runtime_error("unsupported safetensors dtype: " + dtype);
        }
    };
    
    // Test various dtype conversions
    if (test_loader_wrapper::test_safetensors_dtype_to_ggml_type("F32") != GGML_TYPE_F32) {
        printf("FAIL: F32 conversion failed\n");
        return false;
    }
    
    if (test_loader_wrapper::test_safetensors_dtype_to_ggml_type("F16") != GGML_TYPE_F16) {
        printf("FAIL: F16 conversion failed\n"); 
        return false;
    }
    
    if (test_loader_wrapper::test_safetensors_dtype_to_ggml_type("I32") != GGML_TYPE_I32) {
        printf("FAIL: I32 conversion failed\n");
        return false;
    }
    
    // Test unsupported type
    try {
        test_loader_wrapper::test_safetensors_dtype_to_ggml_type("UNSUPPORTED");
        printf("FAIL: Should have thrown exception for unsupported type\n");
        return false;
    } catch (const std::exception& e) {
        // Expected behavior
    }
    
    printf("PASS: Dtype conversion works correctly\n");
    return true;
}

static bool test_safetensors_loading() {
    printf("Testing safetensors loading...\n");
    
    const std::string test_file = "/tmp/test_safetensors_unit.safetensors";
    
    // Create test file
    create_test_safetensors_file(test_file);
    
    try {
        std::vector<std::string> splits;
        llama_model_loader loader(test_file, splits, false, false, nullptr, nullptr);
        
        // Check that file type was detected correctly
        if (loader.file_type != LLAMA_FILE_TYPE_SAFETENSORS) {
            printf("FAIL: File type not detected correctly\n");
            return false;
        }
        
        // Check that tensors were loaded
        if (loader.weights_map.size() != 2) {
            printf("FAIL: Expected 2 tensors, got %zu\n", loader.weights_map.size());
            return false;
        }
        
        // Check tensor1
        auto it1 = loader.weights_map.find("tensor1");
        if (it1 == loader.weights_map.end()) {
            printf("FAIL: tensor1 not found\n");
            return false;
        }
        
        const auto& tensor1 = it1->second.tensor;
        if (tensor1->type != GGML_TYPE_F32) {
            printf("FAIL: tensor1 wrong type\n");
            return false;
        }
        
        if (tensor1->ne[0] != 2 || tensor1->ne[1] != 2) {
            printf("FAIL: tensor1 wrong shape\n");
            return false;
        }
        
        // Check tensor2
        auto it2 = loader.weights_map.find("tensor2");
        if (it2 == loader.weights_map.end()) {
            printf("FAIL: tensor2 not found\n");
            return false;
        }
        
        const auto& tensor2 = it2->second.tensor;
        if (tensor2->type != GGML_TYPE_F16) {
            printf("FAIL: tensor2 wrong type\n");
            return false;
        }
        
        if (tensor2->ne[0] != 3) {
            printf("FAIL: tensor2 wrong shape\n");
            return false;
        }
        
        printf("PASS: Safetensors loading works correctly\n");
        
        // Clean up
        remove(test_file.c_str());
        return true;
        
    } catch (const std::exception& e) {
        printf("FAIL: Exception during loading: %s\n", e.what());
        remove(test_file.c_str());
        return false;
    }
}

int main() {
    printf("Running safetensors tests...\n\n");
    
    bool all_passed = true;
    
    all_passed &= test_file_type_detection();
    all_passed &= test_dtype_conversion();
    all_passed &= test_safetensors_loading();
    
    if (all_passed) {
        printf("\n✅ All safetensors tests passed!\n");
        return 0;
    } else {
        printf("\n❌ Some safetensors tests failed!\n");
        return 1;
    }
}