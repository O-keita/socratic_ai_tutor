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
