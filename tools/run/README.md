# llama.cpp/tools/run

The purpose of this tool is to demonstrate a minimal usage of llama.cpp for running models interactively.

This tool uses llama.cpp directly without any HTTP server or network communication, making it suitable for environments where port binding is not permitted and providing a cleaner integration example for downstream projects.

```bash
llama-run -m model.gguf
```

```bash
Usage: build/bin/llama-run [options]

This tool provides an interactive chat interface using llama.cpp directly.

Common options:
  -h, --help                  Show this help
  -m,    --model FNAME        model path (required)
  -c, --ctx-size N            Context size (default: 2048)
  -n, --predict N             Number of tokens to predict (default: -1, unlimited)
  -t, --threads N             Number of threads
  -ngl, --n-gpu-layers N      Number of layers to offload to GPU (default: 99)
  --temp N                    Temperature for sampling (default: 0.8)
  --top-k N                   Top-k sampling (default: 40)
  --top-p N                   Top-p sampling (default: 0.9)
```
