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
    // Lightweight status check — does NOT initialize engines.
    if (mode == TutorMode.offline) {
      _currentStatus = EngineStatus.offline;
    } else if (mode == TutorMode.online) {
      _currentStatus = EngineStatus.online;
    } else {
      // Auto: report based on which engine is currently ready.
      if (_remoteEngine.isReady) {
        _currentStatus = EngineStatus.online;
      } else if (_localEngine.isReady) {
        _currentStatus = EngineStatus.offline;
      }
      // If neither is ready, keep whatever status we had (offline by default).
    }
    _statusController.add(_currentStatus);
  }

  @override
  bool get isReady => _localEngine.isReady || _remoteEngine.isReady;

  @override
  bool get isGenerating =>
      _localEngine.isGenerating || _remoteEngine.isGenerating;

  @override
  Future<bool> initialize({bool force = false}) async {
    _currentStatus = EngineStatus.connecting;
    _statusController.add(_currentStatus);

    // Only initialize the engine we actually need.  Loading the local GGUF
    // model can consume 400-800 MB of RAM — on budget devices this OOM-kills
    // the app and causes a crash loop.  So we check connectivity first and
    // only fall back to local when remote isn't reachable.

    if (mode == TutorMode.offline) {
      // User explicitly wants local — initialize it.
      final ok = await _initLocalWithTimeout();
      await _updateStatus();
      return ok;
    }

    if (mode == TutorMode.online) {
      final ok = await _remoteEngine.initialize().catchError((e) {
        debugPrint('HybridTutorService: Remote engine init error: $e');
        return false;
      });
      await _updateStatus();
      return ok;
    }

    // Auto mode: try remote first (cheap — just a ping), only load local
    // model if remote is unreachable.
    final remoteOk = await _remoteEngine.initialize().timeout(
      const Duration(seconds: 3),
      onTimeout: () => false,
    ).catchError((e) {
      debugPrint('HybridTutorService: Remote engine init error: $e');
      return false;
    });

    if (remoteOk) {
      await _updateStatus();
      return true;
    }

    // Remote failed — try local as fallback.
    final localOk = await _initLocalWithTimeout();
    await _updateStatus();
    return localOk;
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
    if (mode == TutorMode.offline) {
      await _ensureLocalInitialized();
      return _localEngine;
    }
    if (mode == TutorMode.online) return _remoteEngine;

    // Auto mode: check connectivity
    final connectivity = await Connectivity().checkConnectivity();
    if (connectivity.contains(ConnectivityResult.none)) {
      await _ensureLocalInitialized();
      return _localEngine;
    }

    // Has network — verify the server actually responds (3-second cap)
    final remoteAlive = await _remoteEngine
        .initialize()
        .timeout(
          const Duration(seconds: 3),
          onTimeout: () => false,
        );
    if (remoteAlive) return _remoteEngine;

    // Remote unreachable — fall back to local.
    await _ensureLocalInitialized();
    return _localEngine;
  }

  /// Initialize local engine with a timeout.  If the timeout fires, the
  /// in-progress native `loadModel()` is **aborted** to free memory and
  /// prevent an OOM crash on budget devices.
  Future<bool> _initLocalWithTimeout() async {
    // Don't attempt if a previous OOM crash was detected — the crash flag in
    // SharedPreferences blocks auto-retry to prevent crash loops on low-RAM
    // devices.  The user must clear it explicitly via Settings → Reset Model.
    if (await _localEngine.wasKilledDuringLoad) {
      debugPrint(
          'HybridTutorService: Skipping local init — previous OOM crash detected. '
          'User must retry from Settings.');
      return false;
    }

    // Don't attempt if previous inference crashed the process (e.g. SIGILL on
    // Exynos 1330 mixed-core CPUs).  Fatal signals leave the inference flag set
    // across the crash.  Route to remote instead.
    if (await _localEngine.wasKilledDuringInference) {
      debugPrint(
          'HybridTutorService: Skipping local init — previous inference crashed '
          '(SIGILL/fatal signal). Routing to remote engine.');
      return false;
    }
    if (_localEngine.initializationFailed) {
      _localEngine.resetInitializationFailure();
    }
    final ok = await _localEngine.initialize().timeout(
      const Duration(seconds: 60),
      onTimeout: () {
        debugPrint('HybridTutorService: Local engine init timed out (60s) — aborting to free memory');
        _localEngine.abortLoad();
        return false;
      },
    ).catchError((e) {
      debugPrint('HybridTutorService: Local engine init error: $e');
      return false;
    });
    return ok;
  }

  /// Lazily initialize the local engine only when needed.
  Future<void> _ensureLocalInitialized() async {
    if (_localEngine.isReady) return;
    await _initLocalWithTimeout();
  }

  @override
  void resetConversation() {
    // Only the local engine holds native conversation state.
    // Remote engine is stateless per-request (history sent via JSON).
    _localEngine.resetConversation();
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
