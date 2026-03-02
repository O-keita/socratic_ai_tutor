import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:ffi/ffi.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/message.dart' as model;
import '../utils/app_config.dart';
import 'tutor_engine.dart';

// ---------------------------------------------------------------------------
// FFI type definitions matching libchat.h
// ---------------------------------------------------------------------------

/// Opaque pointer to the native chat_session struct.
typedef ChatSessionPtr = Pointer<Void>;

// chat_create(const char * model_path, int n_ctx, int n_threads) -> chat_session *
typedef _ChatCreateNative = Pointer<Void> Function(
    Pointer<Utf8>, Int32, Int32);
typedef _ChatCreateDart = Pointer<Void> Function(
    Pointer<Utf8>, int, int);

// chat_generate(chat_session *, const char * user_message) -> char *
typedef _ChatGenerateNative = Pointer<Utf8> Function(
    Pointer<Void>, Pointer<Utf8>);
typedef _ChatGenerateDart = Pointer<Utf8> Function(
    Pointer<Void>, Pointer<Utf8>);

// chat_string_free(char *)
typedef _ChatStringFreeNative = Void Function(Pointer<Utf8>);
typedef _ChatStringFreeDart = void Function(Pointer<Utf8>);

// chat_reset(chat_session *)
typedef _ChatResetNative = Void Function(Pointer<Void>);
typedef _ChatResetDart = void Function(Pointer<Void>);

// chat_destroy(chat_session *)
typedef _ChatDestroyNative = Void Function(Pointer<Void>);
typedef _ChatDestroyDart = void Function(Pointer<Void>);

// ---------------------------------------------------------------------------
// Isolate message types for running inference off the UI thread
// ---------------------------------------------------------------------------

class _GenerateRequest {
  final int sessionAddress;
  final String userMessage;
  _GenerateRequest(this.sessionAddress, this.userMessage);
}

class _GenerateResult {
  final String? text;
  final String? error;
  _GenerateResult({this.text, this.error});
}

// ---------------------------------------------------------------------------
// SocraticLlmService
// ---------------------------------------------------------------------------

/// Local LLM inference service using our custom libchat C API (llama.cpp).
///
/// The native library is compiled from source via CMake during the Gradle
/// build and bundled into the APK as `libchat.so`.
class SocraticLlmService implements TutorEngine {
  static final SocraticLlmService _instance = SocraticLlmService._internal();
  factory SocraticLlmService() => _instance;
  SocraticLlmService._internal();

  static String get modelFileName => AppConfig.modelFileName;

  bool _isInitialized = false;
  bool _isGenerating = false;
  bool _isInitializing = false;
  bool _initializationFailed = false;
  String _initializationError = '';

  /// Native session pointer (address stored as int for isolate transfer).
  int _sessionAddress = 0;

  /// FFI function pointers — resolved once on first init.
  /// Only chat_create and chat_destroy run on the main isolate.
  /// chat_generate and chat_string_free are resolved inside the worker isolate.
  static _ChatCreateDart? _chatCreate;
  static _ChatResetDart? _chatReset;
  static _ChatDestroyDart? _chatDestroy;

  final String _systemPrompt =
      'You are a Socratic AI tutor specializing in data science and machine learning.\n\n'
      'RULES:\n'
      '1. ALWAYS begin your response with a thinking block containing your reasoning.\n'
      '2. For conceptual questions: respond with ONE guiding question. NEVER give direct answers. '
      'If the student is stuck, give a small hint before your question.\n'
      '3. For code questions: guide the student to write the code themselves through Socratic questioning.\n'
      '4. For casual messages (greetings, thanks, chitchat): respond warmly and naturally.\n\n'
      'Always start with a thinking block. This is mandatory.';

  @override
  bool get isReady => _isInitialized && _sessionAddress != 0 && !_isGenerating;

  @override
  bool get isGenerating => _isGenerating;

  bool get initializationFailed => _initializationFailed;
  String get initializationError => _initializationError;

  static const _prefLoadingKey = 'llm_model_loading';
  static const _prefInferenceKey = 'llm_inference_running';

  Future<bool> get wasKilledDuringLoad async {
    try {
      final prefs = await SharedPreferences.getInstance();
      return prefs.getBool(_prefLoadingKey) ?? false;
    } catch (_) {
      return false;
    }
  }

  Future<void> clearCrashFlag() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(_prefLoadingKey);
    } catch (_) {}
  }

  Future<bool> get wasKilledDuringInference async {
    try {
      final prefs = await SharedPreferences.getInstance();
      return prefs.getBool(_prefInferenceKey) ?? false;
    } catch (_) {
      return false;
    }
  }

  Future<void> clearInferenceCrashFlag() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(_prefInferenceKey);
    } catch (_) {}
  }

  /// Resolve the FFI function pointers from libchat.so.
  void _ensureBindings() {
    if (_chatCreate != null) return;

    final lib = DynamicLibrary.open('libchat.so');

    _chatCreate = lib
        .lookupFunction<_ChatCreateNative, _ChatCreateDart>('chat_create');
    _chatReset = lib
        .lookupFunction<_ChatResetNative, _ChatResetDart>('chat_reset');
    _chatDestroy = lib
        .lookupFunction<_ChatDestroyNative, _ChatDestroyDart>('chat_destroy');

    debugPrint('LLMService: FFI bindings loaded from libchat.so');
  }

  @override
  Future<bool> initialize({bool force = false}) async {
    if (_isInitialized) return true;
    if (_isInitializing) return false;
    if (_initializationFailed) return false;

    if (!force && await wasKilledDuringLoad) {
      debugPrint('LLMService: Skipping init — previous load crashed.');
      _initializationFailed = true;
      _initializationError =
          'Model loading previously crashed the app. '
          'Tap "Retry local model" in Settings, or use Online mode.';
      return false;
    }

    if (!force && await wasKilledDuringInference) {
      debugPrint('LLMService: Skipping init — previous inference crashed.');
      _initializationFailed = true;
      _initializationError =
          'Local inference crashed on this device (incompatible CPU). '
          'Use Online mode or tap "Reset Model" in Settings.';
      return false;
    }

    _isInitializing = true;
    try {
      _ensureBindings();

      final dir = await getApplicationSupportDirectory()
          .timeout(const Duration(seconds: 5));
      final modelPath = p.join(dir.path, modelFileName);

      if (!await File(modelPath).exists()) {
        throw Exception(
          'Model file not found at $modelPath. '
          'Please download it from Settings → Manage Local Model.',
        );
      }

      final sizeMb =
          (await File(modelPath).length() / 1024 / 1024).toStringAsFixed(1);
      debugPrint('LLMService: Loading $sizeMb MB model from $modelPath');

      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool(_prefLoadingKey, true);

      // Load the model (this is CPU-heavy — runs synchronously but is
      // typically < 3 seconds for a 230 MB Q4 model).
      final pathPtr = modelPath.toNativeUtf8();
      final session = _chatCreate!(pathPtr, 2048, 4);
      malloc.free(pathPtr);

      if (session.address == 0) {
        throw Exception('Native chat_create returned null');
      }

      _sessionAddress = session.address;

      await prefs.remove(_prefLoadingKey);

      _isInitialized = true;
      debugPrint('LLMService: Engine ready (libchat)');
      return true;
    } catch (e, stack) {
      _initializationFailed = true;
      var errMsg = e.toString().split('\n').first;
      errMsg = errMsg.replaceAll(RegExp(r'^(Exception:\s*)+'), '');
      if (errMsg.length > 120) errMsg = '${errMsg.substring(0, 117)}...';
      _initializationError = errMsg;
      debugPrint('LLMService: Init failed: $e\n$stack');
      try {
        final prefs = await SharedPreferences.getInstance();
        await prefs.remove(_prefLoadingKey);
      } catch (_) {}
      return false;
    } finally {
      _isInitializing = false;
    }
  }

  /// Run inference in a background isolate so the UI stays responsive.
  static _GenerateResult _runGenerate(_GenerateRequest req) {
    try {
      final lib = DynamicLibrary.open('libchat.so');
      final generate = lib.lookupFunction<_ChatGenerateNative,
          _ChatGenerateDart>('chat_generate');
      final stringFree = lib.lookupFunction<_ChatStringFreeNative,
          _ChatStringFreeDart>('chat_string_free');

      final session = Pointer<Void>.fromAddress(req.sessionAddress);
      final msgPtr = req.userMessage.toNativeUtf8();
      final resultPtr = generate(session, msgPtr);
      malloc.free(msgPtr);

      if (resultPtr.address == 0) {
        return _GenerateResult(error: 'Native chat_generate returned null');
      }

      final text = resultPtr.toDartString();
      stringFree(resultPtr);
      return _GenerateResult(text: text);
    } catch (e) {
      return _GenerateResult(error: e.toString());
    }
  }

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
    if (!_isInitialized || _sessionAddress == 0) {
      if (_initializationFailed) {
        yield _initializationError.isNotEmpty
            ? 'Model unavailable: $_initializationError'
            : 'Model not initialized. Go to Settings → Reset Model or switch to Online mode.';
        return;
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
    SharedPreferences? prefs;
    try {
      prefs = await SharedPreferences.getInstance();
      await prefs.setBool(_prefInferenceKey, true);

      // Note: conversation history is managed natively inside the chat_session.
      // For the first message, the model's built-in chat template handles
      // system prompt injection. For lesson-scoped chats with a custom system
      // prompt, we prepend it as context in the user message.
      String prompt = userPrompt;
      if (systemPrompt != null && systemPrompt != _systemPrompt) {
        prompt = '[Context: $systemPrompt]\n\n$userPrompt';
      }

      // Run the blocking FFI call in a background isolate.
      final result = await Isolate.run(
        () => _runGenerate(
          _GenerateRequest(_sessionAddress, prompt),
        ),
      );

      if (result.error != null) {
        yield '[Error: ${result.error}]';
      } else if (result.text != null && result.text!.isNotEmpty) {
        // Strip <think>...</think> reasoning blocks before yielding to UI.
        final cleaned = result.text!
            .replaceAll(RegExp(r'<think>[\s\S]*?</think>'), '')
            .trim();
        yield cleaned.isNotEmpty ? cleaned : 'What aspects of this topic would you like to explore further?';
      } else {
        yield 'What aspects of this topic would you like to explore further?';
      }

      await prefs.remove(_prefInferenceKey);
    } catch (e, stack) {
      try {
        await prefs?.remove(_prefInferenceKey);
      } catch (_) {}
      debugPrint('LLMService: Generation error: $e\n$stack');
      yield '[Error during generation. Please try again.]';
    } finally {
      _isGenerating = false;
    }
  }

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

  /// Clear the native conversation history and KV cache.
  /// The model stays loaded — the next generateResponse() starts a fresh chat.
  void resetConversation() {
    if (_sessionAddress == 0) return;
    _ensureBindings();
    _chatReset!(Pointer<Void>.fromAddress(_sessionAddress));
    debugPrint('LLMService: Conversation reset');
  }

  void abortLoad() {
    if (!_isInitializing) return;
    debugPrint('LLMService: Aborting in-progress model load');
    if (_sessionAddress != 0) {
      try {
        _ensureBindings();
        _chatDestroy!(Pointer<Void>.fromAddress(_sessionAddress));
      } catch (_) {}
      _sessionAddress = 0;
    }
    _isInitializing = false;
    _initializationFailed = true;
    _initializationError = 'Model loading was cancelled (timeout).';
    clearCrashFlag();
  }

  void resetInitializationFailure() {
    _initializationFailed = false;
    _initializationError = '';
    _isInitialized = false;
    debugPrint('LLMService: Reset. Call initialize() to retry.');
  }

  Future<void> dispose() async {
    _isGenerating = false;
    if (_sessionAddress != 0) {
      try {
        _ensureBindings();
        _chatDestroy!(Pointer<Void>.fromAddress(_sessionAddress));
      } catch (e) {
        debugPrint('LLMService: Error destroying session: $e');
      }
      _sessionAddress = 0;
    }
    _isInitialized = false;
    _initializationFailed = false;
    debugPrint('LLMService: Disposed');
  }
}
