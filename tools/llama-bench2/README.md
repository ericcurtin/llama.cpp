# llama-bench2

Performance benchmarking tool specifically designed to measure the performance gaps between llama.cpp and vLLM.

## Overview

`llama-bench2` focuses on identifying and measuring the specific performance characteristics where llama.cpp typically differs from vLLM, particularly:

1. **Batch Processing Efficiency** - How well llama.cpp handles different batch sizes
2. **Concurrent Request Handling** - Performance with multiple simultaneous requests  
3. **Memory Usage Patterns** - Memory efficiency compared to vLLM
4. **GPU Utilization** - How effectively GPU resources are used
5. **Throughput vs Latency Tradeoffs** - Performance characteristics under different loads

## Usage

```bash
./llama-bench2 [options]
```

### Options

- `-h, --help` - Show help message
- `-m, --model <path>` - Model path (default: models/7B/ggml-model-q4_0.gguf)
- `-p, --n-prompt <n>` - Number of prompt tokens (default: 512)
- `-n, --n-gen <n>` - Number of tokens to generate (default: 128)
- `-t, --threads <n>` - Number of threads (default: auto-detect)
- `-ngl, --n-gpu-layers <n>` - Number of GPU layers (default: 99)
- `-r, --repetitions <n>` - Number of repetitions per test (default: 3)
- `-b, --batch-sizes <list>` - Comma-separated batch sizes (default: 1,2,4,8,16,32,64)
- `-c, --concurrent <list>` - Comma-separated concurrent request counts (default: 1,2,4,8,16)
- `-o, --output <md|json|csv>` - Output format (default: md)
- `-v, --verbose` - Verbose output

### Examples

Basic benchmark:
```bash
./llama-bench2 -m models/llama-7b-q4_0.gguf
```

Test specific batch sizes:
```bash
./llama-bench2 -b 1,8,32,128 -o json
```

Test concurrent request handling:
```bash
./llama-bench2 -c 1,2,4,8,16 -v
```

## Tests Performed

### 1. Batch Processing Efficiency
Tests how efficiently llama.cpp processes different batch sizes. vLLM typically excels at larger batch sizes due to better GPU parallelization.

**Metrics:**
- Average latency per batch
- Throughput (tokens/second)
- Memory usage

### 2. Concurrent Request Handling
Simulates multiple concurrent inference requests to measure how well llama.cpp handles parallel workloads compared to vLLM's request batching capabilities.

**Metrics:**
- Average request latency
- Overall throughput
- Resource utilization

### 3. Memory Usage Analysis
Analyzes memory consumption patterns to identify efficiency differences compared to vLLM.

**Metrics:**
- Memory footprint
- Memory efficiency per token

## Understanding the Results

The benchmark results help identify:

- **Optimal batch sizes** for your hardware configuration
- **Concurrency limits** where performance degrades
- **Memory bottlenecks** that may limit scalability
- **Performance gaps** compared to vLLM's expected performance

### Performance Recommendations

Based on the results, you can:
- Optimize batch sizes for your use case
- Configure appropriate concurrency limits
- Identify hardware bottlenecks
- Compare against vLLM's expected performance characteristics

## Output Formats

### Markdown (default)
Human-readable table format with analysis and recommendations.

### JSON
Machine-readable format for integration with other tools:
```json
{
  "metadata": {
    "model": "path/to/model.gguf",
    "cpu_info": "CPU details",
    "gpu_info": "GPU details"
  },
  "results": [
    {
      "test_name": "Batch Processing Efficiency",
      "batch_size": 8,
      "avg_latency_ms": 125.5,
      "throughput_tokens_per_sec": 64.2
    }
  ]
}
```

### CSV
Comma-separated values for data analysis and plotting.

## Interpreting Performance Gaps

The benchmark helps identify where llama.cpp may have performance disadvantages compared to vLLM:

- **Large batch processing**: vLLM typically performs better with large batch sizes
- **High concurrency**: vLLM's request batching usually outperforms multiple separate contexts
- **Memory efficiency**: vLLM often uses memory more efficiently for serving workloads
- **GPU utilization**: vLLM may achieve higher GPU utilization rates

These insights can guide optimization efforts and help set realistic performance expectations.