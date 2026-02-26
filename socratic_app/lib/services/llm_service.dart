import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:llamadart/llamadart.dart';
import '../models/message.dart' as model;
import 'tutor_engine.dart';

/// Local LLM inference service using llamadart (llama.cpp).
///
/// Works on all platforms: Android ARM64, Android x86_64 (emulators), iOS,
/// macOS, Linux, Windows. No architecture restrictions.
///
/// Flow:
///   1. App installs without the model (~350 MB GGUF).
///   2. User downloads the model via Settings → Manage Local Model.
///   3. [initialize()] loads the GGUF file into [LlamaEngine].
///   4. [generateResponse()] streams tokens via [LlamaEngine.create()].
class SocraticLlmService implements TutorEngine {
  static final SocraticLlmService _instance = SocraticLlmService._internal();
  factory SocraticLlmService() => _instance;
  SocraticLlmService._internal();

  /// Local filename for the downloaded model.
  /// Must match [ModelSetupScreen.modelFileName].
  static const String modelFileName = 'socratic-model.gguf';

  bool _isInitialized = false;
  bool _isGenerating = false;
  bool _isInitializing = false;
  bool _initializationFailed = false;
  String _initializationError = '';

  LlamaEngine? _engine;
  String? _modelPath;

  final String _systemPrompt =
      'You are a Socratic AI tutor specializing in data science and machine learning. '
      'Your sole teaching method is guided questioning.\n\n'
      'STRICT RULES:\n'
      '- NEVER give direct answers or explanations\n'
      '- ALWAYS respond with exactly ONE focused guiding question\n'
      '- Lead the student to discover the answer themselves\n\n'
      'Example:\n'
      'User: "What is a neural network?"\n'
      'Tutor: "What simplified system in nature processes information through '
      'interconnected nodes, and how might computers replicate that structure?"';

  @override
  bool get isReady => _isInitialized && _engine != null && !_isGenerating;

  @override
  bool get isGenerating => _isGenerating;

  bool get initializationFailed => _initializationFailed;
  String get initializationError => _initializationError;

  /// Load the GGUF model into memory.
  ///
  /// Returns [true] if the engine is ready. Returns [false] (without throwing) if
  /// the model has not been downloaded or [LlamaEngine.loadModel] fails.
  @override
  Future<bool> initialize() async {
    if (_isInitialized) return true;
    if (_isInitializing) return false;
    if (_initializationFailed) return false;

    _isInitializing = true;
    try {
      final dir = await getApplicationSupportDirectory()
          .timeout(const Duration(seconds: 5));
      _modelPath = p.join(dir.path, modelFileName);

      if (!await File(_modelPath!).exists()) {
        throw Exception(
          'Model file not found. Please download it from '
          'Settings → Manage Local Model.',
        );
      }

      final sizeMb =
          (await File(_modelPath!).length() / 1024 / 1024).toStringAsFixed(1);
      debugPrint('LLMService: Loading $sizeMb MB model from $_modelPath');

      _engine = LlamaEngine(LlamaBackend());
      await _engine!.loadModel(_modelPath!);

      _isInitialized = true;
      debugPrint('LLMService: ✅ Engine ready');
      return true;
    } catch (e, stack) {
      _initializationFailed = true;
      _initializationError = e.toString().split('\n').first;
      _engine = null;
      debugPrint('LLMService: ❌ Init failed: $e\n$stack');
      return false;
    } finally {
      _isInitializing = false;
    }
  }

  /// Stream a Socratic response token-by-token.
  ///
  /// Uses [LlamaEngine.create()] (OpenAI-style stateless API) — the full
  /// conversation history is passed on every call so the engine has context.
  ///
  /// [enableThinking] is set to `false` to suppress Qwen3's think blocks and
  /// return answers immediately without internal chain-of-thought.
  @override
  Stream<String> generateResponse(
    String userPrompt, {
    List<model.Message>? history,
    String? systemPrompt,
    int maxTokens = 150,
    double temperature = 0.6,
    double topP = 0.85,
    int topK = 30,
    double repeatPenalty = 1.1,
  }) async* {
    if (!_isInitialized || _engine == null) {
      // If initialization previously failed, reset and retry — the model may
      // have been downloaded since the last attempt.
      if (_initializationFailed) {
        resetInitializationFailure();
      }
      final ok = await initialize();
      if (!ok) {
        yield 'Failed to initialize model. $_initializationError';
        return;
      }
    }

    if (_isGenerating) {
      yield '[Please wait for the current response to finish…]';
      return;
    }

    _isGenerating = true;
    try {
      // Build the full conversation (system + history + current user message)
      final messages = <LlamaChatMessage>[
        LlamaChatMessage.fromText(
          role: LlamaChatRole.system,
          text: systemPrompt ?? _systemPrompt,
        ),
      ];

      if (history != null && history.isNotEmpty) {
        // Pass last 6 messages to keep context without overflowing
        final recent =
            history.length > 6 ? history.sublist(history.length - 6) : history;
        for (final msg in recent) {
          messages.add(
            LlamaChatMessage.fromText(
              role: msg.isUser ? LlamaChatRole.user : LlamaChatRole.assistant,
              text: msg.text,
            ),
          );
        }
      }
      messages.add(
        LlamaChatMessage.fromText(
          role: LlamaChatRole.user,
          text: userPrompt,
        ),
      );

      final params = GenerationParams(
        maxTokens: maxTokens,
        temp: temperature,
        topK: topK,
        topP: topP,
        penalty: repeatPenalty,
      );

      // enableThinking: false → suppress Qwen3 <think> blocks for faster responses
      // delta.content already contains only the answer text (thinking is in delta.thinking)
      await for (final chunk in _engine!.create(
        messages,
        params: params,
        enableThinking: false,
      )) {
        final content = chunk.choices.first.delta.content;
        if (content != null && content.isNotEmpty) {
          yield content;
        }
      }
    } catch (e, stack) {
      debugPrint('LLMService: ❌ Generation error: $e\n$stack');
      yield '[Error during generation. Please try again.]';
    } finally {
      _isGenerating = false;
    }
  }

  /// Non-streaming convenience wrapper — collects the full response.
  @override
  Future<LLMResponse> generateSocraticResponse(
    String userQuestion, {
    List<model.Message>? history,
    int maxTokens = 150,
  }) async {
    final buf = StringBuffer();
    await for (final token in generateResponse(
      userQuestion,
      history: history,
      maxTokens: maxTokens,
    )) {
      buf.write(token);
    }
    final text = buf.toString().trim();
    return LLMResponse(
      text.isEmpty
          ? 'What aspects of this topic would you like to explore further?'
          : text,
    );
  }

  /// Allow re-attempting initialization after a failure.
  void resetInitializationFailure() {
    _initializationFailed = false;
    _initializationError = '';
    _isInitialized = false;
    debugPrint('LLMService: Reset. Call initialize() to retry.');
  }

  /// Release native resources. Call on app shutdown or when swapping models.
  Future<void> dispose() async {
    _isGenerating = false;
    try {
      await _engine?.dispose();
    } catch (e) {
      debugPrint('LLMService: Error disposing engine: $e');
    }
    _engine = null;
    _isInitialized = false;
    _initializationFailed = false;
    _modelPath = null;
    debugPrint('LLMService: Disposed');
  }
}
