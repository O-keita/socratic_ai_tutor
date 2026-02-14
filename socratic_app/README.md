# Socratic AI Tutor - Mobile Frontend

This is a Flutter-based mobile application that provides a **hybrid** Socratic tutoring experience. It intelligently alternates between local, on-device inference (powered by GGUF models) and remote cloud-based inference depending on the user's connectivity and hardware capabilities.

## Architecture

The app uses a **Hybrid Infrastructure**:
- **Local Engine**: Powered by `llama_flutter_android` using a quantized Qwen3-0.6B model.
- **Remote Engine**: Connects to the FastAPI backend for higher-fidelity reasoning when online.
- **Orchestration**: Managed by `HybridTutorService` which handles seamless switching between modes.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.
