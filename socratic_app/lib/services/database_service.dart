import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import '../models/user.dart';

/// File-based user store.
///
/// Stores users as a JSON file at:
///   <appSupportDir>/auth/users.json
///
/// Schema: { "<userId>": { ...userFields, "passwordHash": "..." } }
///
/// This replaces the previous sqflite implementation, which caused
/// MissingPluginException on some Android builds.
class DatabaseService {
  static final DatabaseService _instance = DatabaseService._internal();
  factory DatabaseService() => _instance;
  DatabaseService._internal();

  // In-memory cache; loaded lazily on first access.
  Map<String, dynamic>? _cache;

  Future<File> get _file async {
    try {
      final dir = await getApplicationSupportDirectory();
      final authDir = Directory(p.join(dir.path, 'auth'));
      if (!authDir.existsSync()) authDir.createSync(recursive: true);
      return File(p.join(authDir.path, 'users.json'));
    } catch (e) {
      // Fallback to temp directory on platforms where support dir is unavailable
      final dir = await getTemporaryDirectory();
      return File(p.join(dir.path, 'socratic_auth_users.json'));
    }
  }

  Future<Map<String, dynamic>> _load() async {
    if (_cache != null) return _cache!;
    try {
      final f = await _file;
      if (f.existsSync()) {
        final raw = jsonDecode(f.readAsStringSync());
        _cache = Map<String, dynamic>.from(raw as Map);
        return _cache!;
      }
    } catch (e) {
      debugPrint('DatabaseService: load error (non-fatal): $e');
    }
    _cache = {};
    return _cache!;
  }

  Future<void> _save(Map<String, dynamic> data) async {
    try {
      _cache = data;
      final f = await _file;
      f.writeAsStringSync(jsonEncode(data));
    } catch (e) {
      debugPrint('DatabaseService: save error (non-fatal): $e');
    }
  }

  /// Upsert a user record. Merges [passwordHash] into the stored map if provided.
  Future<void> saveUser(User user, {String? passwordHash}) async {
    final store = await _load();
    final map = user.toMap();
    if (passwordHash != null) map['passwordHash'] = passwordHash;
    store[user.id] = map;
    await _save(store);
    debugPrint('DatabaseService: saved user ${user.username}');
  }

  /// Retrieve a user by ID. Returns null if not found.
  Future<User?> getUser(String id) async {
    final store = await _load();
    final entry = store[id];
    if (entry == null) return null;
    try {
      return User.fromMap(Map<String, dynamic>.from(entry as Map));
    } catch (e) {
      debugPrint('DatabaseService: getUser parse error: $e');
      return null;
    }
  }

  /// Find a user by email or username (case-insensitive).
  /// Returns the raw map including [passwordHash], or null if not found.
  Future<Map<String, dynamic>?> getLocalCredentials(String identifier) async {
    final id = identifier.toLowerCase();
    final store = await _load();
    for (final entry in store.values) {
      final m = Map<String, dynamic>.from(entry as Map);
      final email = (m['email'] as String? ?? '').toLowerCase();
      final username = (m['username'] as String? ?? '').toLowerCase();
      if (email == id || username == id) return m;
    }
    return null;
  }
}
