# OCI/Docker Registry Integration

llama.cpp supports pulling models directly from OCI-compliant registries such as Docker Hub. This feature uses the [go-containerregistry](https://github.com/google/go-containerregistry) library to handle registry authentication and image pulling.

## Features

- Pull GGUF models from Docker Hub and other OCI registries
- Automatic authentication using Docker credentials (via `docker login`)
- Support for private registries with authentication
- Caching of downloaded models

## Prerequisites

- Go 1.24 or later (for building from source)
- Docker credentials configured (for private registries)

## Usage

### Pulling Public Models

To pull a public model from Docker Hub:

```bash
./llama-cli --docker-repo ai/smollm2:135M-Q4_0
```

By default, models are pulled from the `ai/` namespace on Docker Hub. If no namespace is specified, `ai/` is assumed:

```bash
# These are equivalent:
./llama-cli --docker-repo gemma3
./llama-cli --docker-repo ai/gemma3
```

### Pulling Private Models

For private models or registries requiring authentication, first authenticate using Docker:

```bash
docker login
# Or for a specific registry:
docker login registry.example.com
```

Then pull the model:

```bash
./llama-cli --docker-repo myuser/private-model:Q4_K_M
```

### Custom Registries

You can also pull from custom OCI registries by specifying the full registry URL:

```bash
./llama-cli --docker-repo registry.example.com/namespace/model:tag
```

## How It Works

1. The `--docker-repo` (or `-dr`) flag specifies the OCI image reference
2. llama.cpp uses the Go-based OCI library to:
   - Parse the image reference
   - Authenticate using Docker credentials (if available)
   - Fetch the manifest from the registry
   - Identify and download the GGUF layer
3. The model is cached locally for future use

## Image Format

Models must be packaged as OCI images with a GGUF layer. The layer should have one of these media types:
- `application/vnd.docker.ai.gguf.v3`
- Any media type containing "gguf"

## Authentication

Authentication is handled automatically using the same credentials as the Docker CLI:
- Credentials are stored in `~/.docker/config.json`
- Use `docker login` to authenticate
- Supports credential helpers and authentication providers

## Caching

Downloaded models are cached in the standard llama.cpp cache directory:
- Linux/macOS: `~/.cache/llama.cpp/`
- Windows: `%LOCALAPPDATA%\llama.cpp\`

## Building with OCI Support

OCI support is automatically enabled if Go is available during build:

```bash
cmake -B build
cmake --build build
```

If Go is not found, a warning will be displayed and OCI functionality will be unavailable.

## Troubleshooting

### Authentication Issues

If you encounter authentication errors:
1. Ensure you're logged in: `docker login`
2. Verify credentials: Check `~/.docker/config.json`
3. For private registries, specify the full registry URL

### Network Issues

If downloads fail:
1. Check your internet connection
2. Verify the registry is accessible
3. Try pulling a test image with Docker: `docker pull <image>`

### Build Issues

If OCI support is not available:
1. Ensure Go 1.24 or later is installed: `go version`
2. Rebuild the project: `cmake --build build --clean-first`
3. Check CMake output for Go-related warnings
