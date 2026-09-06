# imgen: text-to-image generation

`imgen` runs diffusion transformer image models on ggml. Supported architectures:

| `general.architecture` | Model | Text encoder | VAE |
|---|---|---|---|
| `qwen_image` | [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image-2512) | Qwen2.5-VL-7B-Instruct | Wan2.1 (`qwen_image_vae.safetensors`) |
| `lumina2` | [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Qwen3-4B | Flux (`ae.safetensors`) |

The DiT GGUFs published by [unsloth](https://huggingface.co/unsloth) and [city96's ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) tooling only contain the transformer. The text encoder is a regular llama.cpp GGUF and the VAE is loaded from the ComfyUI safetensors files. When you pass `-hf` (or `-m` with a supported DiT), both are downloaded automatically:

```sh
llama-imgen-cli -hf unsloth/Z-Image-Turbo-GGUF -p "a cat wearing a top hat" -o cat.png
llama-imgen-cli -hf unsloth/Qwen-Image-2512-GGUF:Q4_K_M -p "a cat wearing a top hat" --steps 30 -o cat.png
```

To pick the companions yourself:

```sh
llama-imgen-cli -m z-image-turbo-Q4_K_M.gguf --text-encoder Qwen3-4B-Q8_0.gguf --vae ae.safetensors -p "..."
llama-imgen-cli -m z-image-turbo-Q4_K_M.gguf --text-encoder-hf unsloth/Qwen3-4B-GGUF:Q4_K_M --vae-hf Comfy-Org/z_image_turbo -p "..."
```

Options: `--width`, `--height` (multiples of 16, default 1024), `--steps` (default 8 for Z-Image, 50 for Qwen-Image), `--cfg-scale` (default 1.0 for Z-Image-Turbo, 4.0 for Qwen-Image; values <= 1 skip the unconditional pass), `--negative-prompt`, `-s` seed.

## Server

`llama-server -hf unsloth/Z-Image-Turbo-GGUF` loads the text encoder as the main model and exposes the OpenAI images endpoint:

```sh
curl http://localhost:8080/v1/images/generations -d '{"prompt": "a cat wearing a top hat", "size": "1024x1024"}' \
  | jq -r '.data[0].b64_json' | base64 -d > cat.png
```

Extra request fields: `steps`, `guidance_scale`, `negative_prompt`, `seed`, `width`, `height`. Only `response_format: "b64_json"` is supported. `/props` reports `modalities.image_generation: true`.

## Implementation notes

- Sampling is flow matching Euler with the diffusers schedules: `shift=3` for Z-Image, dynamic shift with `shift_terminal=0.02` for Qwen-Image.
- Text conditioning: Qwen-Image uses the final norm output of Qwen2.5-VL with the first 34 template tokens dropped, Z-Image uses `hidden_states[-2]` of Qwen3 (captured with `cb_eval`).
- 3-axis RoPE is applied with precomputed cos/sin tables, since the per-axis frequency bases differ from `ggml_rope_multi`.
- The Wan VAE is a causal 3D conv network. A single frame only sees the last temporal slice of each kernel, so the decoder runs as a 2D network.
- BF16 weights are converted to F16 on load; quantized DiT tensors are used as-is.
