ARG UBUNTU_VERSION=26.04

FROM ubuntu:$UBUNTU_VERSION AS build

# Install build tools
RUN apt update && apt install -y git build-essential cmake wget xz-utils python3

# Install cURL and Vulkan SDK dependencies (excluding glslc, which we'll build from source)
RUN apt install -y libcurl4-openssl-dev curl \
    libxcb-xinput0 libxcb-xinerama0 libxcb-cursor-dev libvulkan-dev

# Build shaderc (glslc) from source with latest glslang and SPIRV-Tools
# This ensures support for GL_EXT_integer_dot_product extension
RUN cd /tmp && \
    git clone --depth 1 https://github.com/google/shaderc.git && \
    cd shaderc && \
    python3 ./utils/git-sync-deps && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DSHADERC_SKIP_TESTS=ON \
          -DSHADERC_SKIP_EXAMPLES=ON \
          .. && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/shaderc

# Build it
WORKDIR /app

COPY . .

RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

## Base image
FROM ubuntu:$UBUNTU_VERSION AS base

RUN apt-get update \
    && apt-get install -y libgomp1 curl libvulkan1 mesa-vulkan-drivers \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

COPY --from=build /app/lib/ /app

### Full
FROM base AS full

COPY --from=build /app/full /app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    git \
    python3 \
    python3-pip \
    python3-wheel \
    && pip install --break-system-packages --upgrade setuptools \
    && pip install --break-system-packages -r requirements.txt \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

ENTRYPOINT ["/app/tools.sh"]

### Light, CLI only
FROM base AS light

COPY --from=build /app/full/llama-cli /app

WORKDIR /app

ENTRYPOINT [ "/app/llama-cli" ]

### Server, Server only
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/full/llama-server /app

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
