# libchat — Thin C API for LLM Chat Inference on Android

A minimal C wrapper around [llama.cpp](https://github.com/ggml-org/llama.cpp) that compiles from source into your Android APK via CMake. Designed for Flutter apps using `dart:ffi`, but works with any language that can call C functions.

## Why?

Existing Flutter packages for on-device LLM inference (`llama_cpp_dart`, `llamadart`, `llama_flutter_android`) have recurring issues:

- **Missing native sources** — pub.dev strips git submodules, so `llama_cpp_dart` ships without the llama.cpp source code
- **Hardcoded GPU dependencies** — packages enable OpenCL/Vulkan by default, pulling in `libomp.so` or GPU libraries that don't exist on most devices
- **Pre-built binaries with wrong CPU targets** — `llamadart` ships ARMv8.2+dotprod `.so` files that crash with `SIGILL` on older ARM64 CPUs (Cortex-A53/A55)
- **Broken isolate support** — some packages can't run inference off the UI thread

**libchat** solves all of these by compiling llama.cpp directly from source during your Gradle build, giving you full control over CPU targets, GPU features, and the C API surface.

## Architecture

```
Flutter (Dart)
    │
    │  dart:ffi
    ▼
libchat.so          ← 4-function C API (this wrapper)
    │
    │  links against
    ▼
libllama.so         ← llama.cpp compiled from source
libggml.so          ← ggml tensor library
```

## API

```c
// libchat.h

// Load a GGUF model. Returns an opaque session pointer, or NULL on failure.
// CPU-only (n_gpu_layers = 0).
chat_session * chat_create(const char * model_path, int n_ctx, int n_threads);

// Generate a complete response for a user message.
// Manages conversation history and chat template formatting internally.
// Returns a malloc'd string — caller MUST pass it to chat_string_free().
// Returns NULL on error.
char * chat_generate(chat_session * session, const char * user_message);

// Free a string returned by chat_generate().
void chat_string_free(char * str);

// Destroy the session — frees model, context, sampler, and message history.
void chat_destroy(chat_session * session);
```

### Key behaviors

- **Chat template auto-detection**: Uses `llama_model_chat_template()` to read the template from the GGUF metadata. Works with ChatML, Llama, Qwen, and any other template the model was trained with.
- **Conversation history**: The `chat_session` struct tracks all messages internally. Each `chat_generate()` call applies the chat template to the full history, then tokenizes only the delta (new tokens since the last call). This means multi-turn conversations work automatically.
- **Non-streaming**: Returns the full response as a single string. For Flutter, run it in `Isolate.run()` to keep the UI responsive.

## Setup

### Prerequisites

1. A local clone of llama.cpp source code:
   ```bash
   git clone --depth 1 https://github.com/ggml-org/llama.cpp /path/to/llama.cpp
   ```

2. Android NDK (comes with Flutter's Android toolchain)

### File structure

Place these files in your Flutter project:

```
your_app/
├── android/app/
│   ├── build.gradle.kts          # Modified (add externalNativeBuild)
│   └── src/main/cpp/
│       ├── CMakeLists.txt         # Build config
│       ├── libchat.h              # C API header
│       └── libchat.cc             # Implementation (~235 lines)
└── lib/services/
    └── llm_service.dart           # Dart FFI bindings
```

### Step 1: CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.22.1)
project(chat LANGUAGES C CXX)

# Point to your llama.cpp source checkout.
set(LLAMA_CPP_DIR "/path/to/llama.cpp")

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
```

**Important CMake flags:**
- `GGML_CPU_AARCH64=OFF` — disables ARM-specific SIMD (dotprod, i8mm, SVE). ~30% slower but runs on **every** ARM64 device.
- `GGML_OPENMP=OFF` — avoids `libomp.so` dependency that doesn't ship on Android.
- `GGML_VULKAN=OFF` / `GGML_OPENCL=OFF` — CPU-only, saves ~40 MB in APK size.
- `GGML_NATIVE=OFF` — don't auto-detect host CPU features (we're cross-compiling).

### Step 2: build.gradle.kts

Add the `externalNativeBuild` block to your app-level `build.gradle.kts`:

```kotlin
android {
    // ... existing config ...

    // Native C++ build — compiles llama.cpp + libchat into the APK.
    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
        }
    }

    defaultConfig {
        // ... existing config ...

        // Only build for arm64 (llama.cpp doesn't support 32-bit ARM well).
        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_PLATFORM=android-26",
                    "-DCMAKE_BUILD_TYPE=Release"
                )
                abiFilters += "arm64-v8a"
            }
        }
    }
}
```

### Step 3: Dart FFI bindings

```dart
import 'dart:ffi';
import 'dart:isolate';
import 'package:ffi/ffi.dart';

// FFI type definitions matching libchat.h
typedef _ChatCreateNative = Pointer<Void> Function(Pointer<Utf8>, Int32, Int32);
typedef _ChatCreateDart = Pointer<Void> Function(Pointer<Utf8>, int, int);

typedef _ChatGenerateNative = Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>);
typedef _ChatGenerateDart = Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>);

typedef _ChatStringFreeNative = Void Function(Pointer<Utf8>);
typedef _ChatStringFreeDart = void Function(Pointer<Utf8>);

typedef _ChatDestroyNative = Void Function(Pointer<Void>);
typedef _ChatDestroyDart = void Function(Pointer<Void>);

class LlmService {
  late final _ChatCreateDart _chatCreate;
  late final _ChatDestroyDart _chatDestroy;
  int _sessionAddress = 0;

  void _loadBindings() {
    final lib = DynamicLibrary.open('libchat.so');
    _chatCreate = lib.lookupFunction<_ChatCreateNative, _ChatCreateDart>('chat_create');
    _chatDestroy = lib.lookupFunction<_ChatDestroyNative, _ChatDestroyDart>('chat_destroy');
  }

  /// Load a GGUF model. Call once at app start.
  bool loadModel(String modelPath, {int nCtx = 2048, int nThreads = 4}) {
    _loadBindings();
    final pathPtr = modelPath.toNativeUtf8();
    final session = _chatCreate(pathPtr, nCtx, nThreads);
    malloc.free(pathPtr);
    if (session.address == 0) return false;
    _sessionAddress = session.address;
    return true;
  }

  /// Generate a response. Runs in a background isolate.
  Future<String?> generate(String userMessage) async {
    return Isolate.run(() {
      // Re-resolve FFI in the isolate (Dart isolates don't share state).
      final lib = DynamicLibrary.open('libchat.so');
      final gen = lib.lookupFunction<_ChatGenerateNative, _ChatGenerateDart>('chat_generate');
      final free = lib.lookupFunction<_ChatStringFreeNative, _ChatStringFreeDart>('chat_string_free');

      final session = Pointer<Void>.fromAddress(_sessionAddress);
      final msgPtr = userMessage.toNativeUtf8();
      final resultPtr = gen(session, msgPtr);
      malloc.free(msgPtr);

      if (resultPtr.address == 0) return null;
      final text = resultPtr.toDartString();
      free(resultPtr);
      return text;
    });
  }

  /// Free all native resources.
  void dispose() {
    if (_sessionAddress != 0) {
      _chatDestroy(Pointer<Void>.fromAddress(_sessionAddress));
      _sessionAddress = 0;
    }
  }
}
```

**Key pattern**: `DynamicLibrary.open('libchat.so')` must be called again inside `Isolate.run()` because Dart isolates don't share memory. The session pointer is passed as an `int` address and reconstructed with `Pointer<Void>.fromAddress()`.

### Step 4: Build

```bash
cd your_app
flutter build apk --release
```

The first build compiles llama.cpp from source (~45 seconds). Subsequent builds are incremental.

## How it works internally

### chat_create

1. Calls `ggml_backend_load_all()` to register CPU backend
2. Loads the GGUF model with `n_gpu_layers = 0` (CPU-only)
3. Creates a llama context with the specified `n_ctx` and `n_threads`
4. Sets up a sampler chain: min_p(0.05) → temperature(0.6) → dist sampling
5. Returns an opaque `chat_session` pointer

### chat_generate

1. Appends the user message to the session's message history
2. Reads the chat template from the model's GGUF metadata (`llama_model_chat_template`)
3. Applies the template to the full conversation via `llama_chat_apply_template`
4. Computes the delta (new text since last call) and tokenizes only that
5. Runs the decode loop token-by-token until end-of-generation
6. Appends the assistant response to history and returns a malloc'd copy

### chat_destroy

Frees all strdup'd message contents, the sampler, context, model, and the session struct itself.

## Supported models

Any GGUF model that llama.cpp supports, including:

- **Qwen3** (0.6B, 1.7B, 4B, etc.)
- **SmolLM2** (135M, 360M, 1.7B)
- **Llama 3** (1B, 3B, 8B)
- **Phi-3/4** (mini, small)
- **Gemma 2** (2B, 9B)
- **Mistral** (7B)

The chat template is auto-detected from the GGUF metadata, so you don't need to configure it per model.

## Performance notes

- **Model loading**: 2-5 seconds for a 230-460 MB Q4_K_M model on mid-range phones
- **Inference**: ~10-30 tokens/second for 0.6B models on ARM64 (CPU-only, 4 threads)
- **APK size**: ~60 MB overhead for the compiled llama.cpp + ggml libraries
- **RAM**: ~500 MB for a 460 MB Q4_K_M model (model size + KV cache)

## Crash-loop protection (recommended)

On-device LLM inference can crash the app (OOM during model load, SIGILL on incompatible CPUs). Use SharedPreferences flags to detect and block repeat crashes:

```dart
// Set before model load, clear after success
await prefs.setBool('llm_model_loading', true);
// ... load model ...
await prefs.remove('llm_model_loading');

// Set before inference, clear after success
await prefs.setBool('llm_inference_running', true);
// ... run inference ...
await prefs.remove('llm_inference_running');

// On next app start, if flag is still set → previous attempt crashed
if (prefs.getBool('llm_model_loading') == true) {
  // Skip local model, route to remote API instead
}
```

## License

libchat is a thin wrapper. The heavy lifting is done by [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT license).
