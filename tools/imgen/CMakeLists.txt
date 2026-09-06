# imgen: text-to-image generation with diffusion transformers

add_library(imgen
            imgen.cpp
            imgen.h
            imgen-impl.h
            imgen-load.cpp
            imgen-dit.cpp
            imgen-vae.cpp
            imgen-text.cpp
            )

target_link_libraries     (imgen PUBLIC ggml llama PRIVATE vendor::nlohmann)
target_include_directories(imgen PUBLIC .)
target_compile_features   (imgen PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties(imgen PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(TARGET llama-imgen-cli)
add_executable(${TARGET} imgen-cli.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common imgen ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TARGET} RUNTIME)
