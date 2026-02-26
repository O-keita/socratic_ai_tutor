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

  /// Production Heroku URL
  static const String productionUrl = 'https://socratic-ai-tutor-api-7f5ba58f5ff8.herokuapp.com';

  /// The base URL for the FastAPI backend.
  /// Priority: --dart-define=BACKEND_URL > release mode > platform default
  static String get backendUrl {
    const envUrl = String.fromEnvironment('BACKEND_URL');
    if (envUrl.isNotEmpty) return envUrl;

    // Use Heroku in release mode automatically
    if (kReleaseMode) return productionUrl;

    if (kIsWeb) return 'http://localhost:8000';
    try {
      if (Platform.isAndroid) return 'http://10.0.2.2:8000';
    } catch (_) {}
    return 'http://localhost:8000';
  }
}
