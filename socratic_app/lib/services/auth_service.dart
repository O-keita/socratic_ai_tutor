import 'dart:convert';
import 'dart:io' show Platform;
import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:uuid/uuid.dart';
import 'package:dio/dio.dart';

import '../models/user.dart';
import 'database_service.dart';

class AuthService extends ChangeNotifier {
  final _storage = const FlutterSecureStorage();
  final _dbService = DatabaseService();
  final _dio = Dio(BaseOptions(
    connectTimeout: const Duration(seconds: 5),
    receiveTimeout: const Duration(seconds: 5),
  ));
  
  // Use the same base URL logic as ApiService
  String get _baseUrl {
    if (kIsWeb) return 'http://localhost:8000';
    try {
      if (Platform.isAndroid) return 'http://10.0.2.2:8000';
    } catch (_) {}
    return 'http://localhost:8000';
  }
  
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
        _currentUser = await _dbService.getUser(userId);
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
    var bytes = utf8.encode(password);
    return sha256.convert(bytes).toString();
  }

  Future<bool> register(String username, String email, String password) async {
    _isLoading = true;
    notifyListeners();

    try {
      String? remoteId;
      
      // Try online registration first
      try {
        final response = await _dio.post('$_baseUrl/register', data: {
          'username': username,
          'email': email,
          'password': password,
        }).timeout(const Duration(seconds: 10)); // Total timeout for registration
        
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
        // Other network errors: we'll fallback to local registration
      } catch (e) {
        debugPrint('Non-Dio error during remote registration: $e');
      }

      // Local Implementation
      final userId = remoteId ?? const Uuid().v4();
      final newUser = User(
        id: userId,
        username: username,
        email: email,
        createdAt: DateTime.now(),
        isSynced: remoteId != null,
      );

      final passwordHash = _hashPassword(password);
      try {
        await _dbService.saveUser(newUser, passwordHash: passwordHash);
      } catch (e) {
        debugPrint('Error saving user locally (non-fatal): $e');
      }
      
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
      // 1. Try Online First
      final loginUrl = '$_baseUrl/login';
      debugPrint('Attempting online login to $loginUrl');
      
      try {
        final response = await _dio.post(loginUrl, data: {
          'email': sanitizedIdentifier,
          'password': password,
        });
        
        final loginData = response.data;
        debugPrint('Online login successful: ${loginData['username']}');
        await _storage.write(key: 'auth_token', value: loginData['token']);
        
        final newUser = User(
          id: loginData['id'],
          username: loginData['username'],
          email: sanitizedIdentifier.contains('@') ? sanitizedIdentifier : 'synced@user.com',
          createdAt: DateTime.now(),
          isSynced: true,
        );
        
        try {
          await _dbService.saveUser(newUser, passwordHash: _hashPassword(password));
        } catch (e) {
          debugPrint('Error saving user locally (non-fatal): $e');
        }
        
        _currentUser = newUser;
        await _storage.write(key: 'user_id', value: newUser.id);
        
        _isLoading = false;
        notifyListeners();
        return true;
      } on DioException catch (e) {
        final message = e.response?.data?['detail'] ?? e.message;
        debugPrint('Remote login unavailable or failed: $message');
        
        // If it's an explicit auth error (400, 401, 403), don't fallback to local
        if (e.response?.statusCode != null) {
          _isLoading = false;
          notifyListeners();
          throw Exception(message);
        }
        // If it's a connection problem, proceed to local fallback
        debugPrint('Network error during online login, attempting local fallback...');
      } catch (e) {
        debugPrint('Unexpected error during online login: $e');
      }

      // 2. Local fallback
      debugPrint('Attempting local login fallback for $sanitizedIdentifier');
      Map<String, dynamic>? localCreds;
      try {
        localCreds = await _dbService.getLocalCredentials(sanitizedIdentifier);
      } catch (e) {
        debugPrint('Local database query failed: $e');
      }

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
      
      // If we reach here, neither online nor local login worked
      throw Exception('Account not found. Please check your credentials or register.');
    } catch (e) {
      debugPrint('Auth login error final catch: $e');
      _isLoading = false;
      notifyListeners();
      rethrow;
    }
  }

  Future<void> logout() async {
    await _storage.delete(key: 'user_id');
    _currentUser = null;
    notifyListeners();
  }
}
