# Socratic AI Tutor - Mobile Frontend

Flutter-based mobile application that provides a **hybrid** Socratic tutoring experience. Intelligently alternates between local on-device inference and remote cloud-based inference depending on connectivity and hardware.

## Architecture

- **Local Engine**: Custom `libchat` C API wrapping llama.cpp, compiled from source via CMake into the APK. Called from Dart via `dart:ffi`.
- **Remote Engine**: Connects to the FastAPI backend for inference when online.
- **Orchestration**: `HybridTutorService` handles seamless switching between modes.

## On-Device Inference

The app uses a custom thin C wrapper (`libchat`) around llama.cpp instead of third-party packages. See `android/app/src/main/cpp/LIBCHAT.md` for full documentation.

```
Dart (llm_service.dart) → dart:ffi → libchat.so → llama.cpp
```

## Getting Started

```bash
flutter pub get
flutter run
```

For release builds (ARM64 only):
```bash
flutter build apk --release
```
