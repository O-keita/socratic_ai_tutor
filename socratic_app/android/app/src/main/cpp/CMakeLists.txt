cmake_minimum_required(VERSION 3.22.1)
project(chat LANGUAGES C CXX)

# Path to llama.cpp source (cloned with the local_llama_cpp_dart fork).
# From: socratic_app/android/app/src/main/cpp/
# To:   local_llama_cpp_dart/src/llama.cpp/
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../local_llama_cpp_dart/src/llama.cpp")

# ---------- CPU-only, baseline ARM64, no OpenMP ----------
set(GGML_OPENCL       OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN       OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE       OFF CACHE BOOL "" FORCE)
set(GGML_CPU_AARCH64  OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP       OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL        OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    ON  CACHE BOOL "" FORCE)

# Build llama.cpp (produces libllama.so, libggml.so, etc.)
add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)

# ---------- Our thin wrapper ----------
add_library(chat SHARED libchat.cc)

target_include_directories(chat PRIVATE
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(chat PRIVATE llama ggml log)

set_target_properties(chat PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
