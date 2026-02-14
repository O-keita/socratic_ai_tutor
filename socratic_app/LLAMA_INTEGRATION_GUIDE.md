# Integration Guide: Using Fixed LLMService in Your App

This guide shows the correct patterns for using the production-fixed `SocraticLlmService` and the **hybrid orchestration layer** used in the actual application.

---

## 🛰️ Hybrid Orchestration (Recommended)

While `SocraticLlmService` handles local inference, the app actually utilizes `HybridTutorService` to provide a robust experience across all devices.

**Key Benefits:**
- **Local-First**: Runs offline on ARM64 devices via `SocraticLlmService`.
- **Cloud-Fallback**: Seamlessly switches to `RemoteLlmService` (FastAPI) on x86 emulators or when connectivity is available.
- **Auto-Switching**: Monitors network status via `ConnectivityPlus`.

## Quick Start

### 1. Initialize on App Startup

**In your main.dart or app initialization:**
```dart
import 'lib/services/llm_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize LLM at startup with timeout protection
  final llm = SocraticLlmService();
  final initialized = await llm.initialize().timeout(
    Duration(minutes: 5),
    onTimeout: () {
      debugPrint('LLMService initialization timed out');
      return false;
    },
  );
  
  if (initialized) {
    debugPrint('LLM Service ready for inference');
  } else {
    debugPrint('LLM Service initialization failed - app will retry on first use');
  }
  
  runApp(MyApp());
}
```

---

## Pattern 1: Streaming Generation (Recommended for UI)

Use this for real-time token display. Tokens appear as they're generated.

```dart
import 'lib/services/llm_service.dart';
import 'lib/models/message.dart';

class TutorScreen extends StatefulWidget {
  @override
  State<TutorScreen> createState() => _TutorScreenState();
}

class _TutorScreenState extends State<TutorScreen> {
  final llm = SocraticLlmService();
  String responseText = '';
  bool isLoading = false;
  StreamSubscription<String>? _responseSubscription;

  @override
  void initState() {
    super.initState();
    // Optional: Verify initialized at screen open
    if (!llm.isReady) {
      llm.initialize().timeout(Duration(minutes: 3)).catchError((_) => false);
    }
  }

  void handleUserMessage(String userMessage) {
    if (llm.isGenerating) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please wait, tutor is thinking...')),
      );
      return;
    }

    setState(() {
      responseText = '';
      isLoading = true;
    });

    // Cancel any previous subscription
    _responseSubscription?.cancel();

    // Start streaming generation
    _responseSubscription = llm.generateResponse(
      userMessage,
      maxTokens: 100,
      temperature: 0.6,
    ).listen(
      (token) {
        // Update UI with each token as it arrives
        setState(() {
          responseText += token;
        });
      },
      onDone: () {
        setState(() {
          isLoading = false;
        });
        debugPrint('Response complete: $responseText');
      },
      onError: (error) {
        setState(() {
          isLoading = false;
          responseText = 'Error: $error';
        });
        debugPrint('Generation error: $error');
      },
    );
  }

  @override
  void dispose() {
    _responseSubscription?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Socratic Tutor')),
      body: Column(
        children: [
          Expanded(
            child: SingleChildScrollView(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Text(responseText),
              ),
            ),
          ),
          if (isLoading)
            Padding(
              padding: EdgeInsets.all(8),
              child: CircularProgressIndicator(),
            ),
          Padding(
            padding: EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    onSubmitted: handleUserMessage,
                    decoration: InputDecoration(
                      hintText: 'Ask a question...',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                SizedBox(width: 8),
                ElevatedButton(
                  onPressed: llm.isGenerating 
                    ? null
                    : () => handleUserMessage('Your message'),
                  child: Text('Send'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
```

---

## Pattern 2: Non-Streaming Generation (For Batch Processing)

Use this when you need the full response before updating UI.

```dart
class TutorService {
  final llm = SocraticLlmService();
  final List<Message> conversationHistory = [];

  Future<String> askQuestion(String userQuestion) async {
    // Ensure initialized
    if (!llm.isReady) {
      final ok = await llm.initialize().timeout(Duration(minutes: 5));
      if (!ok) {
        return 'Model initialization failed. Please try again.';
      }
    }

    // Check if already generating
    if (llm.isGenerating) {
      return 'Tutor is busy. Please wait.';
    }

    try {
      // Add user message to history
      conversationHistory.add(Message(
        text: userQuestion,
        isUser: true,
        timestamp: DateTime.now(),
      ));

      // Generate response with full history context
      final tutorResponse = await llm.generateSocraticResponse(
        userQuestion,
        history: conversationHistory,
      ).timeout(
        Duration(seconds: 30),
        onTimeout: () => 'Response took too long. Please try again.',
      );

      // Add tutor response to history
      conversationHistory.add(Message(
        text: tutorResponse,
        isUser: false,
        timestamp: DateTime.now(),
      ));

      return tutorResponse;

    } catch (e) {
      debugPrint('Error getting response: $e');
      return 'An error occurred. Please try again.';
    }
  }

  Future<void> reset() async {
    conversationHistory.clear();
    await llm.stopGeneration();
  }

  Future<void> dispose() async {
    await llm.dispose();
  }
}
```

---

## Pattern 3: Handling Multi-Turn Dialogue with Context

```dart
class DialogueManager {
  final llm = SocraticLlmService();
  final List<Message> messages = [];
  static const int maxHistoryMessages = 6; // 3 user-tutor pairs

  Future<void> initialize() async {
    final ok = await llm.initialize().timeout(Duration(minutes: 5));
    if (!ok) {
      throw Exception('Failed to initialize LLM service');
    }
  }

  Future<String> respondToStudent(String studentMessage) async {
    if (!llm.isReady) {
      throw Exception('LLM service not ready');
    }

    // Add student message
    messages.add(Message(
      text: studentMessage,
      isUser: true,
      timestamp: DateTime.now(),
    ));

    // Keep only recent history for context window
    final recentMessages = messages.length > maxHistoryMessages
        ? messages.sublist(messages.length - maxHistoryMessages)
        : messages;

    try {
      final tutorResponse = await llm.generateSocraticResponse(
        studentMessage,
        history: recentMessages.sublist(0, recentMessages.length - 1), // Exclude current message
      ).timeout(Duration(seconds: 30));

      // Add tutor response
      messages.add(Message(
        text: tutorResponse,
        isUser: false,
        timestamp: DateTime.now(),
      ));

      return tutorResponse;
    } catch (e) {
      debugPrint('Error in dialogue: $e');
      rethrow;
    }
  }

  void reset() {
    messages.clear();
  }

  Future<void> cleanup() async {
    await llm.dispose();
    messages.clear();
  }
}
```

---

## Pattern 4: Error Handling with User Feedback

```dart
class LLMWrapper {
  final llm = SocraticLlmService();

  Future<String?> safeGenerateResponse(
    String prompt, {
    VoidCallback? onTimeout,
    VoidCallback? onInitFailed,
  }) async {
    try {
      // Check if already initialized
      if (!llm.isReady) {
        debugPrint('LLM not ready, initializing...');
        final initialized = await llm.initialize().timeout(
          Duration(minutes: 5),
          onTimeout: () {
            onTimeout?.call();
            return false;
          },
        );

        if (!initialized) {
          onInitFailed?.call();
          if (llm.initializationFailed) {
            debugPrint('LLM initialization permanently failed');
            // User should restart app or try again later
          }
          return null;
        }
      }

      // Check if already generating
      if (llm.isGenerating) {
        debugPrint('LLM already generating a response');
        return null;
      }

      // Generate response with timeout
      final response = await llm.generateSocraticResponse(prompt).timeout(
        Duration(seconds: 30),
        onTimeout: () {
          onTimeout?.call();
          return null;
        },
      );

      return response;
    } on SocketException catch (e) {
      // File I/O error
      debugPrint('File I/O error: $e');
      return null;
    } on TimeoutException catch (e) {
      debugPrint('Timeout: $e');
      onTimeout?.call();
      return null;
    } catch (e) {
      debugPrint('Unexpected error: $e');
      return null;
    }
  }

  Future<void> cleanup() => llm.dispose();
}
```

---

## Pattern 5: Checking Service State Before Operations

```dart
class TutorWidget extends StatelessWidget {
  final llm = SocraticLlmService();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // Status indicator
          StatusBadge(
            isReady: llm.isReady,
            isGenerating: llm.isGenerating,
            initFailed: llm.initializationFailed,
          ),
          
          SizedBox(height: 16),

          // User prompt input
          TextField(
            enabled: llm.isReady && !llm.isGenerating,
            decoration: InputDecoration(
              hintText: llm.isReady 
                ? 'Ask a question...' 
                : llm.initializationFailed
                  ? 'Model failed to load. Restart app.'
                  : 'Loading model...',
              border: OutlineInputBorder(),
            ),
          ),

          SizedBox(height: 8),

          // Action buttons with state-based enabling
          if (!llm.isReady)
            ElevatedButton(
              onPressed: () async {
                final ok = await llm.initialize().timeout(
                  Duration(minutes: 5),
                );
                if (!ok && llm.initializationFailed) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Model initialization failed. Please restart the app.')),
                  );
                }
              },
              child: Text('Initialize Model'),
            )
          else if (llm.isGenerating)
            ElevatedButton(
              onPressed: llm.stopGeneration,
              child: Text('Stop Response'),
            )
          else
            ElevatedButton(
              onPressed: () {
                // Generate response
              },
              child: Text('Generate Response'),
            ),
        ],
      ),
    );
  }
}

class StatusBadge extends StatelessWidget {
  final bool isReady;
  final bool isGenerating;
  final bool initFailed;

  const StatusBadge({
    required this.isReady,
    required this.isGenerating,
    required this.initFailed,
  });

  @override
  Widget build(BuildContext context) {
    late Color color;
    late String text;

    if (initFailed) {
      color = Colors.red;
      text = 'Model Failed';
    } else if (isGenerating) {
      color = Colors.amber;
      text = 'Generating...';
    } else if (isReady) {
      color = Colors.green;
      text = 'Ready';
    } else {
      color = Colors.grey;
      text = 'Loading...';
    }

    return Chip(
      backgroundColor: color.withOpacity(0.3),
      label: Text(text, style: TextStyle(color: color)),
    );
  }
}
```

---

## Pattern 6: Recovery from Failures

```dart
class RobustLLMService {
  final llm = SocraticLlmService();
  int _retryCount = 0;
  static const int _maxRetries = 2;

  Future<String> generateWithRetry(String prompt) async {
    _retryCount = 0;

    while (_retryCount < _maxRetries) {
      try {
        // Ensure initialized
        if (!llm.isReady) {
          if (llm.initializationFailed) {
            // Reset and retry
            llm.resetInitializationFailure();
            await Future.delayed(Duration(seconds: 2));
          }

          final ok = await llm.initialize().timeout(
            Duration(minutes: 3),
          );
          if (!ok) {
            _retryCount++;
            if (_retryCount < _maxRetries) {
              debugPrint('Init failed, retry $_retryCount/$_maxRetries');
              await Future.delayed(Duration(seconds: 2));
              continue;
            } else {
              throw Exception('Initialization failed after $_maxRetries attempts');
            }
          }
        }

        // Generate response
        return await llm.generateSocraticResponse(prompt).timeout(
          Duration(seconds: 30),
        );

      } catch (e) {
        _retryCount++;
        if (_retryCount < _maxRetries) {
          debugPrint('Generation failed, retry $_retryCount/$_maxRetries: $e');
          await llm.stopGeneration();
          await Future.delayed(Duration(seconds: 1));
        } else {
          debugPrint('Generation failed after $_maxRetries attempts: $e');
          rethrow;
        }
      }
    }

    throw Exception('Failed to generate response after retries');
  }
}
```

---

## Common Gotchas

### ❌ Don't Do This

```dart
// ❌ WRONG: No timeout on initialize
await llm.initialize();  

// ❌ WRONG: Don't call generate in sync code
final resp = llm.generateSocraticResponse('hi');  // Returns Future immediately

// ❌ WRONG: Don't ignore initialization state
llm.generateResponse('hi').listen(...);  // May fail silently if not initialized

// ❌ WRONG: Don't pass huge history
await llm.generateSocraticResponse(
  'question',
  history: allMessagesEver,  // Will exceed context window
);

// ❌ WRONG: Don't dispose while generating
llm.stopGeneration();
await llm.dispose();  // No delay - may crash
```

### ✅ Do This Instead

```dart
// ✅ CORRECT: Always use timeout
await llm.initialize().timeout(Duration(minutes: 5));

// ✅ CORRECT: Await futures
final resp = await llm.generateSocraticResponse('hi');

// ✅ CORRECT: Check state first
if (llm.isReady && !llm.isGenerating) {
  llm.generateResponse('hi').listen(...);
}

// ✅ CORRECT: Limit history to last 3 messages
final recent = messages.length > 3 
  ? messages.sublist(messages.length - 3) 
  : messages;
await llm.generateSocraticResponse(question, history: recent);

// ✅ CORRECT: Stop then dispose
await llm.stopGeneration();
await Future.delayed(Duration(milliseconds: 100));
await llm.dispose();
```

---

## Testing Locally

Before deploying, test with this minimal example:

```dart
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  final llm = SocraticLlmService();
  
  print('Initializing...');
  final ok = await llm.initialize().timeout(Duration(minutes: 5));
  print('Initialized: $ok');
  
  if (ok) {
    print('\n--- Test 1: Streaming ---');
    final response1 = StringBuffer();
    await llm.generateResponse('What is AI?').listen(
      (token) {
        stdout.write(token);
        response1.write(token);
      },
      onDone: () => print('\nDone!'),
    ).asFuture();
    
    print('\n--- Test 2: Non-streaming ---');
    final response2 = await llm.generateSocraticResponse('What is ML?').timeout(
      Duration(seconds: 30),
    );
    print('Response: $response2\n');
    
    print('✅ All tests passed');
  } else {
    print('❌ Initialization failed');
  }
  
  exit(0);
}
```

---

## Monitoring and Debugging

### Enable Verbose Logging
All `SocraticLlmService` operations log with `debugPrint`. In debug mode, you'll see:
- Initialization steps
- Model loading progress
- Generation timing
- Errors with full context

### Check Logcat on Device
```bash
adb logcat | grep "LLMService"
```

### Verify at Runtime
```dart
print('Is ready: ${llm.isReady}');
print('Is generating: ${llm.isGenerating}');
print('Init failed: ${llm.initializationFailed}');
```

---

## Performance Tuning

### Adjust Generation Parameters
```dart
llm.generateResponse(
  'question',
  maxTokens: 50,      // Shorter responses = faster
  temperature: 0.5,   // Lower = more focused
  topK: 20,           // Lower = more predictable
  topP: 0.8,          // Lower = more consistent
)
```

### Reduce Context Window (if slow)
Edit the code: `contextSize: 2048` instead of 4096. Trade-off: less dialogue history.

### Limit Message History
In your code, keep only last 2-3 messages instead of 4.
