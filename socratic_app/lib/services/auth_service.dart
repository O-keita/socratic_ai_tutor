import 'dart:convert';
import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:uuid/uuid.dart';
import 'package:dio/dio.dart';

import '../models/user.dart';
import '../utils/app_config.dart';
import 'database_service.dart';

class AuthService extends ChangeNotifier {
  final _storage = const FlutterSecureStorage();
  final _dbService = DatabaseService();
  final _dio = Dio(BaseOptions(
    connectTimeout: const Duration(seconds: 5),
    receiveTimeout: const Duration(seconds: 5),
  ));

  // Single source of truth for backend URL
  String get _baseUrl => AppConfig.backendUrl;

  User? _currentUser;
  bool _isLoading = false;
  bool _isInitialized = false;

  User? get currentUser => _currentUser;
  bool get isLoading => _isLoading;
  bool get isInitialized => _isInitialized;
  bool get isAuthenticated => _currentUser != null;

  AuthService() {
    _loadSession();
  }

  Future<void> _loadSession() async {
    _isLoading = true;
    notifyListeners();

    try {
      final userId = await _storage.read(key: 'user_id');
      if (userId != null) {
        // Try database first
        try {
          _currentUser = await _dbService.getUser(userId);
        } catch (e) {
          debugPrint('DB session load failed, trying secure storage: $e');
        }
        // Fall back to secure storage backup
        if (_currentUser == null) {
          final userJson = await _storage.read(key: 'offline_user_json');
          if (userJson != null) {
            try {
              _currentUser = User.fromMap(
                  jsonDecode(userJson) as Map<String, dynamic>);
            } catch (e) {
              debugPrint('Secure storage session restore failed: $e');
            }
          }
        }
      }
    } catch (e) {
      debugPrint('Error loading session: $e');
    } finally {
      _isLoading = false;
      _isInitialized = true;
      notifyListeners();
    }
  }

  String _hashPassword(String password) {
    final bytes = utf8.encode(password);
    return sha256.convert(bytes).toString();
  }

  /// Persist credentials to secure storage as a reliable offline fallback.
  /// This is used when sqflite is unavailable (e.g. plugin not linked).
  Future<void> _saveCredentialsSecurely(User user, String passwordHash) async {
    try {
      await _storage.write(key: 'offline_user_json', value: jsonEncode(user.toMap()));
      await _storage.write(key: 'offline_password_hash', value: passwordHash);
    } catch (e) {
      debugPrint('Secure credential backup failed (non-fatal): $e');
    }
  }

  Future<Map<String, dynamic>?> _getCredentialsFromSecureStorage(
      String identifier) async {
    try {
      final userJson = await _storage.read(key: 'offline_user_json');
      final hash = await _storage.read(key: 'offline_password_hash');
      if (userJson == null || hash == null) return null;
      final userMap = jsonDecode(userJson) as Map<String, dynamic>;
      final email = (userMap['email'] as String? ?? '').toLowerCase();
      final username = (userMap['username'] as String? ?? '').toLowerCase();
      if (identifier == email || identifier == username) {
        return {...userMap, 'passwordHash': hash};
      }
    } catch (e) {
      debugPrint('Secure credential read failed: $e');
    }
    return null;
  }

  Future<bool> register(String username, String email, String password) async {
    _isLoading = true;
    notifyListeners();

    try {
      String? remoteId;

      // Try online registration first
      try {
        final response = await _dio
            .post('$_baseUrl/register', data: {
              'username': username,
              'email': email,
              'password': password,
            })
            .timeout(const Duration(seconds: 10));

        remoteId = response.data['id'];
        debugPrint('Remote registration successful: $remoteId');
      } on DioException catch (e) {
        final message = e.response?.data?['detail'] ?? e.message;
        debugPrint('Remote registration failed/unavailable: $message');

        // If it's a conflict (400) or validation (422), stop and inform user
        if (e.response?.statusCode == 400 || e.response?.statusCode == 422) {
          _isLoading = false;
          notifyListeners();
          throw Exception(message);
        }
        // Other network errors: fall back to local registration
      } catch (e) {
        debugPrint('Non-Dio error during remote registration: $e');
      }

      // Local implementation
      final userId = remoteId ?? const Uuid().v4();
      final newUser = User(
        id: userId,
        username: username,
        email: email,
        createdAt: DateTime.now(),
        isSynced: remoteId != null,
      );

      final pwHash = _hashPassword(password);
      try {
        await _dbService.saveUser(newUser, passwordHash: pwHash);
      } catch (e) {
        debugPrint('Error saving user locally (non-fatal): $e');
      }
      await _saveCredentialsSecurely(newUser, pwHash);

      _currentUser = newUser;
      await _storage.write(key: 'user_id', value: userId);

      _isLoading = false;
      notifyListeners();
      return true;
    } catch (e) {
      debugPrint('Auth registration error: $e');
      _isLoading = false;
      notifyListeners();
      return false;
    }
  }

  Future<bool> login(String identifier, String password) async {
    _isLoading = true;
    notifyListeners();
    final sanitizedIdentifier = identifier.trim().toLowerCase();
    debugPrint('Login attempt for: $sanitizedIdentifier');

    try {
      // 1. Try online first
      try {
        final response = await _dio.post('$_baseUrl/login', data: {
          'email': sanitizedIdentifier,
          'password': password,
        });

        final loginData = response.data;
        debugPrint('Online login successful: ${loginData['username']}');
        await _storage.write(key: 'auth_token', value: loginData['token']);

        final newUser = User(
          id: loginData['id'],
          username: loginData['username'],
          email: sanitizedIdentifier.contains('@')
              ? sanitizedIdentifier
              : 'synced@user.com',
          createdAt: DateTime.now(),
          isSynced: true,
        );

        final pwHash = _hashPassword(password);
        try {
          await _dbService.saveUser(newUser, passwordHash: pwHash);
        } catch (e) {
          debugPrint('Error saving user locally (non-fatal): $e');
        }
        await _saveCredentialsSecurely(newUser, pwHash);

        _currentUser = newUser;
        await _storage.write(key: 'user_id', value: newUser.id);

        _isLoading = false;
        notifyListeners();
        return true;
      } on DioException catch (e) {
        final message = e.response?.data?['detail'] ?? e.message;
        debugPrint('Remote login unavailable or failed: $message');

        // Explicit auth errors (401, 403) — don't fall back to local
        if (e.response?.statusCode != null) {
          _isLoading = false;
          notifyListeners();
          throw Exception(message);
        }
        // Connection problem — proceed to local fallback
        debugPrint('Network error, attempting local login fallback...');
      } catch (e) {
        debugPrint('Unexpected error during online login: $e');
      }

      // 2. Local fallback — try database first, then secure storage backup
      debugPrint('Attempting local login for $sanitizedIdentifier');
      Map<String, dynamic>? localCreds;
      try {
        localCreds = await _dbService.getLocalCredentials(sanitizedIdentifier);
      } catch (e) {
        debugPrint('Local database query failed: $e');
      }
      // If database failed or had no record, check secure storage backup
      localCreds ??= await _getCredentialsFromSecureStorage(sanitizedIdentifier);

      if (localCreds != null) {
        final storedHash = localCreds['passwordHash'];
        if (_hashPassword(password) == storedHash) {
          debugPrint('Local login successful');
          _currentUser = User.fromMap(localCreds);
          await _storage.write(key: 'user_id', value: _currentUser!.id);
          _isLoading = false;
          notifyListeners();
          return true;
        } else {
          debugPrint('Local password mismatch');
          throw Exception('Incorrect password');
        }
      }

      _isLoading = false;
      notifyListeners();
      throw Exception('Account not found. Please check your credentials or register.');
    } catch (e) {
      debugPrint('Auth login error: $e');
      _isLoading = false;
      notifyListeners();
      rethrow;
    }
  }

  /// Returns the stored JWT, or null if not logged in / token missing.
  Future<String?> getToken() async {
    try {
      return await _storage.read(key: 'auth_token');
    } catch (_) {
      return null;
    }
  }

  Future<void> logout() async {
    await _storage.delete(key: 'user_id');
    await _storage.delete(key: 'auth_token');
    _currentUser = null;
    notifyListeners();
  }
}
