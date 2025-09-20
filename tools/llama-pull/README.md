# llama-pull - Model Download Tool

A command-line tool for downloading AI models from HuggingFace and Docker Registry for use with llama.cpp.

## Usage

```bash
# Download from HuggingFace
llama-pull -hf <user>/<model>[:<quant>]

# Download from Docker Registry  
llama-pull -dr [<repo>/]<model>[:<quant>]
```

## Options

- `-hf, --hf-repo REPO` - Download model from HuggingFace repository
- `-dr, --docker-repo REPO` - Download model from Docker registry  
- `--hf-token TOKEN` - HuggingFace token for private repositories
- `--offline` - Offline mode, don't attempt downloads
- `-h, --help` - Show help message

## Examples

```bash
# Download a HuggingFace model
llama-pull -hf microsoft/DialoGPT-medium

# Download a Docker model (ai/ repo is default)
llama-pull -dr gemma3

# Download with specific quantization
llama-pull -hf bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M

# Use offline mode (only use cached models)
llama-pull -dr gemma3 --offline
```

## Model Storage

Downloaded models are stored in the standard llama.cpp cache directory:
- Linux/macOS: `~/.cache/llama.cpp/`
- The models can then be used with other llama.cpp tools

## Requirements

- Built with `LLAMA_USE_CURL=ON` (default) for download functionality
- Internet connection for downloading (unless using `--offline` mode)