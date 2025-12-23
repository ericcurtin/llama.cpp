#!/bin/bash

# llama-bench2 Demo Script
# This script demonstrates how to use llama-bench2 to analyze performance gaps vs vLLM

echo "=== llama-bench2: Performance Gap Analysis Tool ==="
echo ""
echo "This tool focuses on identifying performance differences between llama.cpp and vLLM."
echo ""

# Show help
echo "1. Basic usage (help):"
./build/bin/llama-bench2 --help
echo ""

echo "2. Example usage scenarios:"
echo ""

echo "Basic benchmark with default settings:"
echo "  ./llama-bench2 -m /path/to/model.gguf"
echo ""

echo "Test specific batch sizes (vLLM excels at large batches):"
echo "  ./llama-bench2 -b 1,8,32,128 -o json"
echo ""

echo "Test concurrent request handling (vLLM's strength):"
echo "  ./llama-bench2 -c 1,2,4,8,16 -v"
echo ""

echo "Comprehensive performance analysis:"
echo "  ./llama-bench2 -b 1,2,4,8,16,32,64 -c 1,2,4,8 -r 5 -o csv"
echo ""

echo "Quick memory usage check:"
echo "  ./llama-bench2 -b 1 -c 1 -r 1"
echo ""

echo "=== Expected Performance Patterns vs vLLM ==="
echo ""
echo "The benchmark helps identify these common patterns:"
echo "- Batch processing: vLLM typically performs better with large batch sizes"
echo "- Concurrency: vLLM's request batching usually outperforms separate contexts"
echo "- Memory efficiency: vLLM often uses memory more efficiently for serving"
echo "- GPU utilization: vLLM may achieve higher GPU utilization rates"
echo ""

echo "=== Output Formats ==="
echo ""
echo "Markdown (default): Human-readable tables with analysis"
echo "JSON: Machine-readable format for integration"
echo "CSV: Data format for analysis and plotting"
echo ""

echo "Note: This demo requires a valid GGUF model file to run actual tests."
echo "The tool will gracefully handle missing models and show appropriate error messages."