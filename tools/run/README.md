# llama.cpp/tools/run

The purpose of this tool is to demonstrate a minimal usage of llama.cpp for running models with an interactive chat interface.

This implementation uses the llama.cpp server infrastructure with a shim layer approach, maximizing code sharing with llama-server while avoiding network dependencies.

## Usage

```bash
llama-run -m path/to/model.gguf
```

```bash
Usage: llama-run [server-options]

This tool provides an interactive chat interface using shared llama.cpp server infrastructure.
All options are passed through to the llama server configuration.

Common options:
  -h, --help                  Show this help
  -m,    --model FNAME        model path
  -hf,   -hfr, --hf-repo      <user>/<model>[:quant] Hugging Face model repository
  -c, --ctx-size N            Context size
  -n, --predict N             Number of tokens to predict
  -t, --threads N             Number of threads

For all server options, run: llama-server --help
```

## Architecture

This tool demonstrates how to use llama.cpp in downstream projects by:

1. **Sharing argument parsing** with llama-server using `common_params_parse()`
2. **Reusing initialization logic** with `common_init_from_params()`  
3. **Using direct API calls** instead of HTTP requests
4. **Avoiding network dependencies** (no sockets, ports, or HTTP)

## Key Benefits

- **Code Sharing**: Uses the same infrastructure as llama-server
- **No Network**: Works in environments where port binding is not permitted
- **Simpler**: Direct API usage without HTTP client/server complexity
- **Educational**: Shows best practices for llama.cpp integration

## Implementation Details

The implementation uses a "shim layer" approach where:
- Server infrastructure is available but socket operations are no-ops
- Handler functions can be called directly instead of through HTTP
- All llama-server arguments and functionality are preserved
- Interactive loop uses direct llama.cpp APIs for token generation
