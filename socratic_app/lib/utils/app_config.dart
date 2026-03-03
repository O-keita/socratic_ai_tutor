import 'dart:io';
import 'package:flutter/foundation.dart';

/// Centralized backend URL configuration.
///
/// Override at build time with:
///   flutter run --dart-define=BACKEND_URL=http://192.168.1.100:8000
///
/// This is the single source of truth for the backend URL.
/// All services (auth, remote LLM, courses) import this instead of
/// each duplicating their own platform-detection logic.
class AppConfig {
  AppConfig._();

  /// Production URL
  static const String productionUrl = 'https://socratic.hx-ai.org';

  // ── Model configuration (single source of truth) ──────────────────────
  // Update these when switching to a new model version.

  /// Local filename used to save/load the GGUF model on device.
  static const String modelFileName = 'socratic-model.gguf';

  /// Direct download URL for the GGUF model file.
  static const String modelDownloadUrl =
      'https://huggingface.co/Omar-keita/DSML-Socatic-qwen3-0.6B/resolve/main/socratic-qwen3-0.5B-Q4_K_M_offline.gguf';

  /// Human-readable model name shown in the UI.
  static const String modelDisplayName = 'Qwen3-0.6B';

  /// Version of the model this app build ships with.
  /// Seeded into SharedPreferences on first download so update checks work.
  static const String bundledModelVersion = '1.0';

  /// The base URL for the FastAPI backend.
  /// Priority: --dart-define=BACKEND_URL > platform default
  ///
  /// Android (emulator or device) always uses the production URL.
  /// Desktop debug builds fall back to localhost:8000 for local development.
  /// Override at any time with: flutter run --dart-define=BACKEND_URL=http://10.0.2.2:8000
  static String get backendUrl {
    const envUrl = String.fromEnvironment('BACKEND_URL');
    if (envUrl.isNotEmpty) return envUrl;

    if (kIsWeb) return 'http://localhost:8000';
    try {
      if (Platform.isAndroid || Platform.isIOS) return productionUrl;
    } catch (_) {}

    // Desktop debug — use local backend if running, else production
    return kReleaseMode ? productionUrl : 'http://localhost:8000';
  }
}
