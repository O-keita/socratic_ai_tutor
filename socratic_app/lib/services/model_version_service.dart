import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../utils/app_config.dart';

/// Tracks the installed model version and checks the backend for updates.
///
/// The backend's `GET /model/version` endpoint returns the latest version,
/// download URL, and release notes. This service compares that with the
/// version stored in SharedPreferences to determine if an update is available.
class ModelVersionService extends ChangeNotifier {
  static final ModelVersionService _instance = ModelVersionService._internal();
  factory ModelVersionService() => _instance;
  ModelVersionService._internal();

  static const String _prefKey = 'model_version';

  bool _updateAvailable = false;
  String? _latestVersion;
  String? _latestDownloadUrl;
  String? _releaseNotes;
  String? _latestDisplayName;

  bool get updateAvailable => _updateAvailable;
  String? get latestVersion => _latestVersion;
  String? get latestDownloadUrl => _latestDownloadUrl;
  String? get releaseNotes => _releaseNotes;
  String? get latestDisplayName => _latestDisplayName;

  /// Read the currently installed model version from SharedPreferences.
  Future<String?> getInstalledVersion() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_prefKey);
  }

  /// Persist the version after a successful model download.
  Future<void> setInstalledVersion(String version) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefKey, version);
  }

  /// Ask the backend for the latest model version and compare with local.
  /// Returns true if an update is available.
  Future<bool> checkForUpdate() async {
    try {
      final response = await http
          .get(Uri.parse('${AppConfig.backendUrl}/model/version'))
          .timeout(const Duration(seconds: 5));

      if (response.statusCode != 200) return false;

      final data = jsonDecode(response.body) as Map<String, dynamic>;
      _latestVersion = data['version'] as String?;
      _latestDownloadUrl = data['download_url'] as String?;
      _releaseNotes = data['release_notes'] as String?;
      _latestDisplayName = data['display_name'] as String?;

      final installed = await getInstalledVersion();

      // Only flag an update if the user already has a model and versions differ.
      // If installed is null (no model yet), ModelSetupScreen handles first download.
      _updateAvailable =
          _latestVersion != null &&
          installed != null &&
          installed != _latestVersion;

      notifyListeners();
      debugPrint(
        'ModelVersionService: installed=$installed latest=$_latestVersion '
        'updateAvailable=$_updateAvailable',
      );
      return _updateAvailable;
    } catch (e) {
      debugPrint('ModelVersionService: check failed: $e');
      return false;
    }
  }

  /// Clear the update flag (e.g. after the user completes the update).
  void clearUpdateFlag() {
    _updateAvailable = false;
    notifyListeners();
  }
}
