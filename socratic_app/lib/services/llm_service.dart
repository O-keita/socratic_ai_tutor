import 'dart:io';
import 'dart:async';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:llama_flutter_android/llama_flutter_android.dart';
import '../models/message.dart' as model;
import 'tutor_engine.dart';

/// Service for local LLM inference using GGUF models with llama_flutter_android.
/// 
/// IMPORTANT: The model MUST be bundled as a release asset in android/app/src/main/assets/
/// Flutter's asset system does not support large (>100MB) files via pubspec.yaml.
/// 
/// Usage:
/// ```dart
/// final llm = SocraticLlmService();
/// final initialized = await llm.initialize().timeout(
///   Duration(minutes: 5),
///   onTimeout: () => throw TimeoutException('Model init timeout'),
/// );
/// 
/// if (initialized) {
///   llm.generateResponse("What is machine learning?").listen((token) {
///     print(token); // Print each token as it arrives
///   });
/// }
/// ```
class SocraticLlmService implements TutorEngine {
  static final SocraticLlmService _instance = SocraticLlmService._internal();
  factory SocraticLlmService() => _instance;
  SocraticLlmService._internal();

  // Model configuration
  static const String _modelFileName = 'socratic-q4_k_m.gguf';
  static const String _modelAssetPath = 'assets/models/$_modelFileName';
  
  bool _isInitialized = false;
  bool _isGenerating = false;
  bool _isInitializing = false;
  bool _initializationFailed = false;
  LlamaController? _controller;
  StreamSubscription? _generationSubscription;
  String? _modelPath;

  /// System prompt for Socratic tutoring behavior
  /// Optimized for q4_k_m quantization with concise responses
  final String _systemPrompt = """You are a Socratic AI tutor. Ask one focused guiding question.

Do NOT explain. Do NOT provide answers. Only ask questions.

Example:
User: "What is a neural network?"
You: "What is a simplified system in nature that processes information, and how do you think computers could imitate it?"

Ask exactly one question that guides thinking.""";

  /// Check if the model is initialized and ready
  bool get isReady => _isInitialized && _controller != null && !_isGenerating;
  
  /// Check if currently generating a response
  bool get isGenerating => _isGenerating;
  
  /// Check if initialization previously failed (prevents retry loops)
  bool get initializationFailed => _initializationFailed;

  /// Initialize the LLM service by copying model from assets and loading it.
  /// 
  /// CRITICAL: Must be called with timeout to prevent hangs:
  /// ```dart
  /// final ok = await initialize().timeout(Duration(minutes: 5));
  /// ```
  /// 
  /// Failure modes:
  /// - Returns false if model file not found in assets
  /// - Returns false if insufficient disk space
  /// - Returns false if LlamaController fails to load
  /// - Sets _initializationFailed to prevent infinite retries
  Future<bool> initialize() async {
    if (_isInitialized) return true;
    
    // Check architecture before anything else
    bool isX86 = false;
    try {
      isX86 = Platform.isAndroid && 
         (Platform.version.toLowerCase().contains('x86') || 
          Platform.operatingSystemVersion.toLowerCase().contains('x86') ||
          Platform.localHostname.toLowerCase().contains('x86'));
    } catch (_) {}
    
    if (isX86) {
      debugPrint('LLMService: ❌ x86_64 Architecture detected! Local inference is disabled.');
      _initializationFailed = true;
      return false;
    }

    if (_isInitializing) {
      debugPrint('LLMService: Initialization already in progress');
      return false;
    }
    
    if (_initializationFailed) {
      debugPrint('LLMService: Previous initialization failed. Manual reset required.');
      return false;
    }

    _isInitializing = true;

    try {
      // Step 1: Get application data directory
      final directory = await _getApplicationDataDirectory();
      if (directory == null) {
        throw Exception(
          'CRITICAL: Native plugins (path_provider) failed to initialize.\n\n'
          'REASON: This usually happens on x86_64 Emulators because the LLM native library (llama_flutter_android) '
          'only supports ARM64 (Real Devices). The native crash during startup blocked all other plugins.\n\n'
          'FIX: PLEASE USE A REAL ANDROID DEVICE or an ARM64 Emulator.'
        );
      }

      // Step 2: Check available disk space (Removed strict check due to unreliable FileStat on directories)
      // We will handle space issues during the actual file copy via catch/retry.
      
      // Step 3: Check local storage for model file (downloaded or copied)
      _modelPath = p.join(directory.path, _modelFileName);
      final modelFile = File(_modelPath!);
      
      if (await modelFile.exists()) {
        final size = await modelFile.length();
        debugPrint('LLMService: Model found in local storage (${(size / 1024 / 1024).toStringAsFixed(1)} MB)');
      } else {
        debugPrint('LLMService: Model not found in storage, checking assets fallback...');
        try {
          await _copyModelFromAssets(_modelPath!);
        } catch (e) {
          debugPrint('LLMService: Asset fallback failed: $e');
          throw Exception(
            'Model file not found and no asset fallback available.\n\n'
            'FIX: Please download the model from the setup screen.'
          );
        }
      }

      // Step 4: Initialize LlamaController
      await _initializeController(_modelPath!);

      _isInitialized = true;
      _isInitializing = false;
      debugPrint('LLMService: ✅ Initialization complete. Model: $_modelPath');
      return true;
      
    } catch (e, stack) {
      _isInitializing = false;
      _initializationFailed = true;
      debugPrint('LLMService: ❌ FATAL: Initialization failed: $e');
      debugPrint(stack.toString());
      return false;
    }
  }

  /// Get application data directory with retry and validation.
  /// 
  /// Uses exponential backoff to handle slow platform channel registration on startup.
  Future<Directory?> _getApplicationDataDirectory() async {
    const maxAttempts = 5;
    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        Directory directory;
        if (Platform.isAndroid) {
          // On Android, support directory is often more reliable for large assets
          directory = await getApplicationSupportDirectory().timeout(const Duration(seconds: 2));
        } else {
          directory = await getApplicationDocumentsDirectory().timeout(const Duration(seconds: 2));
        }
        
        // Verify directory is accessible and writable
        if (!await directory.exists()) {
          await directory.create(recursive: true);
        }
        
        // Quick permission check
        final testFile = File(p.join(directory.path, '.write_test'));
        await testFile.writeAsString('test');
        await testFile.delete();
        
        debugPrint('LLMService: ✅ Data directory verified: ${directory.path}');
        return directory;
        
      } catch (e) {
        debugPrint('LLMService: Directory access attempt $attempt/$maxAttempts failed: $e');
        if (attempt < maxAttempts) {
          // Exponential backoff
          await Future.delayed(Duration(seconds: attempt));
        }
      }
    }
    return null;
  }

  /// Copy GGUF model from Flutter assets to local filesystem.
  /// 
  /// CRITICAL ISSUE: Flutter's asset system (pubspec.yaml) does NOT support large files.
  /// The model MUST be in android/app/src/main/assets/
  /// This method loads from there using the assets system.
  /// 
  /// For files >100MB, this will fail unless bundled directly in APK.
  /// Use incremental copy for very large models.
  Future<void> _copyModelFromAssets(String destinationPath) async {
    final file = File(destinationPath);
    
    // Skip if already copied
    if (await file.exists()) {
      final size = await file.length();
      debugPrint('LLMService: Model already exists (${(size / 1024 / 1024).toStringAsFixed(1)} MB)');
      return;
    }

    debugPrint('LLMService: Copying model from assets...');
    debugPrint('LLMService: Source: $_modelAssetPath');
    debugPrint('LLMService: Destination: $destinationPath');
    
    final stopwatch = Stopwatch()..start();
    
    try {
      // Attempt to load from assets
      final ByteData data = await rootBundle.load(_modelAssetPath);
      debugPrint('LLMService: Loaded ${(data.lengthInBytes / 1024 / 1024).toStringAsFixed(1)} MB from assets');
      
      // Create parent directory
      await File(destinationPath).parent.create(recursive: true);
      
      // Write directly (small enough for this model)
      final Uint8List bytes = data.buffer.asUint8List();
      await file.writeAsBytes(bytes, flush: true);
      
      stopwatch.stop();
      debugPrint('LLMService: ✅ Model copied in ${stopwatch.elapsedMilliseconds}ms');
      
    } on PlatformException catch (e) {
      stopwatch.stop();
      throw Exception(
        'Asset loading failed. Model must be in android/app/src/main/assets/models/. '
        'Platform error: ${e.message}'
      );
    } catch (e) {
      stopwatch.stop();
      throw Exception('Failed to copy model: $e');
    }
  }

  /// Initialize the LlamaController with the model at the given path
  Future<void> _initializeController(String modelPath) async {
    debugPrint('LLMService: Initializing LlamaController...');
    
    // Verify file exists before loading
    if (!await File(modelPath).exists()) {
      throw Exception('Model file not found at $modelPath');
    }
    
    _controller = LlamaController();
    
    try {
      await _controller!.loadModel(
        modelPath: modelPath,
        threads: 4,           // Balanced for mid-range devices (Snapdragon 665-778)
        contextSize: 4096,    // INCREASED: Allows proper multi-turn dialogue
      );
      
      debugPrint('LLMService: ✅ LlamaController ready');
    } catch (e) {
      _controller = null;
      throw Exception('Failed to load model into LlamaController: $e');
    }
  }

  /// Generate a streaming response for the given prompt.
  /// Returns a Stream<String> that yields tokens in real-time.
  /// 
  /// CRITICAL: Subscribe immediately; tokens start flowing immediately.
  /// 
  /// Usage:
  /// ```dart
  /// final subscription = llm.generateResponse("What is AI?").listen(
  ///   (token) => setState(() => response += token),
  ///   onDone: () => print("Done!"),
  ///   onError: (e) => print("Error: $e"),
  /// );
  /// ```
  @override
  Stream<String> generateResponse(
    String userPrompt, {
    List<model.Message>? history,
    String? systemPrompt,
    int maxTokens = 150,
    double temperature = 0.6,
    double topP = 0.85,
    int topK = 30,
    double repeatPenalty = 1.2,
  }) async* {
    // Ensure initialized
    if (!_isInitialized || _controller == null) {
      if (_initializationFailed) {
        yield "Model initialization failed. App restart required.";
        return;
      }
      yield "Initializing model...";
      final success = await initialize();
      if (!success) {
        yield "Failed to initialize model.";
        return;
      }
    }

    if (_isGenerating) {
      yield "[Please wait for previous response...]";
      return;
    }

    _isGenerating = true;
    final stopwatch = Stopwatch()..start();

    try {
      // Build messages with system prompt
      List<ChatMessage> messages = [
        ChatMessage(role: 'system', content: systemPrompt ?? _systemPrompt),
      ];

      // Add conversation history
      if (history != null && history.isNotEmpty) {
        final recentHistory = history.length > 5 
            ? history.sublist(history.length - 5) 
            : history;
        for (var msg in recentHistory) {
          messages.add(ChatMessage(
            role: msg.isUser ? 'user' : 'assistant',
            content: msg.text,
          ));
        }
      }

      // Add current prompt
      messages.add(ChatMessage(role: 'user', content: userPrompt));

      debugPrint('LLMService: Generating stream...');

      // Use a StreamController that collects tokens BEFORE the async* subscribes
      late StreamSubscription<String> modelSubscription;
      final tokenController = StreamController<String>();
      
      // FIX: Subscribe to model FIRST, THEN set up the yield loop
      // This ensures no tokens are lost
      modelSubscription = _controller!.generateChat(
        messages: messages,
        template: 'chatml',
        maxTokens: maxTokens,
        temperature: temperature,
        topP: topP,
        topK: topK,
        repeatPenalty: repeatPenalty,
      ).listen(
        (token) {
          // FIX: Preserve end-of-sequence markers for proper termination
          // Only remove redundant markers
          final cleanToken = token
              .replaceAll('<|im_start|>', '')  // Remove role markers
              .replaceAll('<|end_header_id|>', '')
              .replaceAll('<|endoftext|>', '');  // Keep im_end for detection
          
          if (cleanToken.isNotEmpty && cleanToken != '<|im_end|>') {
            tokenController.add(cleanToken);
          } else if (cleanToken == '<|im_end|>') {
            // End marker detected; close the stream properly
            tokenController.close();
          }
        },
        onDone: () {
          stopwatch.stop();
          debugPrint('LLMService: ✅ Stream done in ${stopwatch.elapsedMilliseconds}ms');
          _isGenerating = false;
          if (!tokenController.isClosed) {
            tokenController.close();
          }
        },
        onError: (error) {
          stopwatch.stop();
          debugPrint('LLMService: ❌ Stream error: $error');
          _isGenerating = false;
          tokenController.addError(error);
          if (!tokenController.isClosed) {
            tokenController.close();
          }
        },
      );

      // NOW yield tokens from the controller (subscription is already active)
      await for (final token in tokenController.stream) {
        yield token;
      }
      
      await modelSubscription.cancel();
      
    } catch (e, stack) {
      _isGenerating = false;
      debugPrint('LLMService: ❌ Stream exception: $e');
      debugPrint(stack.toString());
      yield "[Error during generation]";
    }
  }

  /// Generate a complete Socratic response (non-streaming).
  /// Collects all tokens and returns the full response string.
  /// 
  /// Use this when you need the full response before updating UI.
  /// For UI, prefer `generateResponse()` stream for better UX.
  @override
  Future<String> generateSocraticResponse(
    String userQuestion, {
    List<model.Message>? history,
    int maxTokens = 150,
  }) async {
    if (!_isInitialized || _controller == null) {
      if (_initializationFailed) {
        return "Model initialization failed. App restart required.";
      }
      final success = await initialize();
      if (!success) {
        return "Failed to initialize model.";
      }
    }

    if (_isGenerating) {
      return "[Waiting for previous response...]";
    }

    _isGenerating = true;
    final stopwatch = Stopwatch()..start();

    try {
      // Build chat messages for the template
      List<ChatMessage> messages = [
        ChatMessage(role: 'system', content: _systemPrompt),
      ];

      // Add conversation history (last 3 messages to stay within context)
      if (history != null && history.isNotEmpty) {
        final recentHistory = history.length > 3 
            ? history.sublist(history.length - 3) 
            : history;
        for (var msg in recentHistory) {
          messages.add(ChatMessage(
            role: msg.isUser ? 'user' : 'assistant',
            content: msg.text,
          ));
        }
      }

      // Add current user question
      messages.add(ChatMessage(role: 'user', content: userQuestion));

      debugPrint('LLMService: Generating with ${messages.length} messages...');

      // Collect streaming response
      final StringBuffer responseBuffer = StringBuffer();
      final completer = Completer<String>();
      bool ended = false;

      _generationSubscription = _controller!.generateChat(
        messages: messages,
        template: 'chatml',
        maxTokens: maxTokens,
        temperature: 0.4,
        topP: 0.9,
        topK: 40,
        repeatPenalty: 1.1,
      ).listen(
        (token) {
          if (!ended) {
            responseBuffer.write(token);
          }
        },
        onDone: () {
          if (!ended && !completer.isCompleted) {
            ended = true;
            completer.complete(responseBuffer.toString());
          }
        },
        onError: (error) {
          if (!ended && !completer.isCompleted) {
            ended = true;
            completer.completeError(error);
          }
        },
      );

      final response = await completer.future;
      
      stopwatch.stop();
      debugPrint('LLMService: ✅ Generated in ${stopwatch.elapsedMilliseconds}ms');
      
      _isGenerating = false;
      
      // Clean up response (remove only markers, preserve content)
      String cleaned = response
          .replaceAll('<|im_start|>', '')
          .replaceAll('<|end_header_id|>', '')
          .replaceAll('<|im_end|>', '')
          .replaceAll('<|endoftext|>', '')
          .trim();

      return cleaned.isEmpty 
          ? "What aspects of this topic would you like to explore?" 
          : cleaned;
          
    } catch (e, stack) {
      _isGenerating = false;
      debugPrint('LLMService: ❌ Generation error: $e');
      debugPrint(stack.toString());
      return "I encountered an issue. Could you rephrase your question?";
    }
  }

  /// Stop the current generation immediately
  Future<void> stopGeneration() async {
    if (_controller != null && _isGenerating) {
      debugPrint('LLMService: Stopping generation...');
      try {
        await _controller!.stop();
      } catch (e) {
        debugPrint('LLMService: Error stopping controller: $e');
      }
      
      await _generationSubscription?.cancel();
      _generationSubscription = null;
      _isGenerating = false;
    }
  }

  /// Clean up resources and reset service
  /// Call this on app shutdown or when reinitializing
  Future<void> dispose() async {
    debugPrint('LLMService: Disposing...');
    await stopGeneration();
    try {
      await _controller?.dispose();
    } catch (e) {
      debugPrint('LLMService: Error disposing controller: $e');
    }
    _controller = null;
    _isInitialized = false;
    _initializationFailed = false;
    _modelPath = null;
  }

  /// Reset failure state to allow retry
  void resetInitializationFailure() {
    _initializationFailed = false;
    debugPrint('LLMService: Reset initialization failure flag');
  }
}
