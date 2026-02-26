import 'dart:async';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/message.dart';
import 'tutor_engine.dart';
import 'llm_service.dart';
import 'remote_llm_service.dart';

enum TutorMode { auto, online, offline }

enum EngineStatus { online, offline, connecting }

/// Orchestrates routing between Local and Remote LLM engines.
class HybridTutorService implements TutorEngine {
  final SocraticLlmService _localEngine = SocraticLlmService();
  final RemoteLlmService _remoteEngine = RemoteLlmService();

  TutorMode mode = TutorMode.auto;
  EngineStatus _currentStatus = EngineStatus.offline;
  final StreamController<EngineStatus> _statusController =
      StreamController<EngineStatus>.broadcast();

  Stream<EngineStatus> get statusStream => _statusController.stream;
  EngineStatus get currentStatus => _currentStatus;

  static final HybridTutorService _instance = HybridTutorService._internal();
  factory HybridTutorService() => _instance;
  HybridTutorService._internal() {
    _loadMode();
    _updateStatus();
  }

  Future<void> _loadMode() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final savedMode = prefs.getString('tutor_mode');
      if (savedMode != null) {
        mode = TutorMode.values.firstWhere(
          (e) => e.toString() == savedMode,
          orElse: () => TutorMode.auto,
        );
        _updateStatus();
      }
    } catch (e) {
      debugPrint('HybridTutorService: Error loading mode: $e');
    }
  }

  Future<void> _updateStatus() async {
    final engine = await _getBestEngine();
    _currentStatus =
        engine is RemoteLlmService ? EngineStatus.online : EngineStatus.offline;
    _statusController.add(_currentStatus);
  }

  @override
  bool get isReady => _localEngine.isReady || _remoteEngine.isReady;

  @override
  bool get isGenerating =>
      _localEngine.isGenerating || _remoteEngine.isGenerating;

  @override
  Future<bool> initialize() async {
    _currentStatus = EngineStatus.connecting;
    _statusController.add(_currentStatus);

    // Initialize both engines in parallel so neither blocks the other.
    // Local engine is capped at 5 seconds — on x86_64 emulators the ARM64-only
    // native library crashes the plugin channel, causing path_provider retries
    // that would otherwise hang for ~20 seconds.
    final results = await Future.wait([
      _localEngine.initialize().timeout(
        const Duration(seconds: 5),
        onTimeout: () {
          debugPrint('HybridTutorService: Local engine init timed out (unsupported architecture?)');
          return false;
        },
      ).catchError((e) {
        debugPrint('HybridTutorService: Local engine init error: $e');
        return false;
      }),
      _remoteEngine.initialize().catchError((e) {
        debugPrint('HybridTutorService: Remote engine init error: $e');
        return false;
      }),
    ]);

    await _updateStatus();
    return results[0] || results[1];
  }

  void setMode(TutorMode newMode) {
    mode = newMode;
    _updateStatus();
    _saveMode(newMode);
  }

  Future<void> _saveMode(TutorMode newMode) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('tutor_mode', newMode.toString());
    } catch (e) {
      debugPrint('HybridTutorService: Error saving mode: $e');
    }
  }

  Future<TutorEngine> _getBestEngine() async {
    // llamadart supports all platforms, so no platform restrictions needed.
    if (mode == TutorMode.offline) return _localEngine;
    if (mode == TutorMode.online) return _remoteEngine;

    // Auto mode: check connectivity
    final connectivity = await Connectivity().checkConnectivity();
    if (connectivity.contains(ConnectivityResult.none)) {
      return _localEngine;
    }

    // Has network — verify the server actually responds (3-second cap)
    final remoteAlive = await _remoteEngine
        .initialize()
        .timeout(
          const Duration(seconds: 3),
          onTimeout: () => false,
        );
    return remoteAlive ? _remoteEngine : _localEngine;
  }

  @override
  Stream<String> generateResponse(
    String prompt, {
    List<Message>? history,
    int maxTokens = 150,
  }) async* {
    final engine = await _getBestEngine();
    _currentStatus = engine is RemoteLlmService ? EngineStatus.online : EngineStatus.offline;
    _statusController.add(_currentStatus);
    yield* engine.generateResponse(prompt, history: history, maxTokens: maxTokens);
  }

  @override
  Future<LLMResponse> generateSocraticResponse(
    String prompt, {
    List<Message>? history,
    int maxTokens = 150,
  }) async {
    final engine = await _getBestEngine();
    _currentStatus = engine is RemoteLlmService ? EngineStatus.online : EngineStatus.offline;
    _statusController.add(_currentStatus);
    return engine.generateSocraticResponse(
      prompt,
      history: history,
      maxTokens: maxTokens,
    );
  }
}
