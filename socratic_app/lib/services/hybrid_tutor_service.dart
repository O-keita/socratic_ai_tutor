import 'dart:async';
import 'package:connectivity_plus/connectivity_plus.dart';
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
  final StreamController<EngineStatus> _statusController = StreamController<EngineStatus>.broadcast();
  
  Stream<EngineStatus> get statusStream => _statusController.stream;
  EngineStatus get currentStatus => _currentStatus;
  
  static final HybridTutorService _instance = HybridTutorService._internal();
  factory HybridTutorService() => _instance;
  HybridTutorService._internal() {
    // Initial status
    _updateStatus();
  }

  Future<void> _updateStatus() async {
    final engine = await _getBestEngine();
    if (engine is RemoteLlmService) {
      _currentStatus = EngineStatus.online;
    } else {
      _currentStatus = EngineStatus.offline;
    }
    _statusController.add(_currentStatus);
  }

  @override
  bool get isReady => _localEngine.isReady || _remoteEngine.isReady;

  @override
  bool get isGenerating => _localEngine.isGenerating || _remoteEngine.isGenerating;

  @override
  Future<bool> initialize() async {
    _currentStatus = EngineStatus.connecting;
    _statusController.add(_currentStatus);
    
    // Initialize both engines in parallel. 
    // This prevents the local engine (which might hang on incompatible hardware) 
    // from blocking the remote engine check.
    final List<bool> results = await Future.wait([
      _localEngine.initialize().catchError((e) {
        print('HybridTutorService: Local engine init error: $e');
        return false;
      }),
      _remoteEngine.initialize().catchError((e) {
        print('HybridTutorService: Remote engine init error: $e');
        return false;
      }),
    ]);
    
    final localOk = results[0];
    final remoteOk = results[1];
    
    await _updateStatus();
    return localOk || remoteOk;
  }

  void setMode(TutorMode newMode) {
    mode = newMode;
    _updateStatus();
  }

  Future<TutorEngine> _getBestEngine() async {
    if (mode == TutorMode.offline) return _localEngine;
    if (mode == TutorMode.online) return _remoteEngine;
    
    // Auto mode: check connectivity
    final List<ConnectivityResult> connectivityResult = await (Connectivity().checkConnectivity());
    if (connectivityResult.contains(ConnectivityResult.none)) {
      return _localEngine;
    }
    
    // If online, try remote, but check if it's actually responding
    // Using a quick check to avoid hangs
    final remoteAlive = await _remoteEngine.initialize();
    return remoteAlive ? _remoteEngine : _localEngine;
  }

  @override
  Stream<String> generateResponse(String prompt, {List<Message>? history, int maxTokens = 150}) async* {
    await _updateStatus();
    final engine = await _getBestEngine();
    yield* engine.generateResponse(prompt, history: history, maxTokens: maxTokens);
  }

  @override
  Future<String> generateSocraticResponse(String prompt, {List<Message>? history, int maxTokens = 150}) async {
    await _updateStatus();
    final engine = await _getBestEngine();
    return engine.generateSocraticResponse(prompt, history: history, maxTokens: maxTokens);
  }
}
