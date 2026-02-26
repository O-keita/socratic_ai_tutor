# Socratic AI Tutor — Senior Developer Review

> **Date:** 2026-02-20
> **Reviewed by:** Senior Developer Audit (Claude Code)
> **Branch:** `main`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Backend Deep Dive](#3-backend-deep-dive)
4. [Frontend Deep Dive](#4-frontend-deep-dive)
5. [Critical Problems](#5-critical-problems)
6. [Performance Issues](#6-performance-issues)
7. [Architecture & Code Quality Issues](#7-architecture--code-quality-issues)
8. [What's Actually Good](#8-whats-actually-good)
9. [Dependencies](#9-dependencies)
10. [Recommended Action Plan](#10-recommended-action-plan)

---

## 1. Project Overview

A **hybrid offline-first mobile learning platform** that teaches Data Science & Machine Learning through Socratic questioning. The system uses a Flutter frontend with a FastAPI backend, and intelligently routes LLM inference to either a local on-device GGUF model (Android ARM64) or a remote FastAPI server depending on connectivity and hardware.

| Property | Value |
|----------|-------|
| **Frontend** | Flutter 3.10+ (Dart) |
| **Backend** | FastAPI (Python) |
| **LLM** | Qwen3-0.6B GGUF (Q4_K_M, ~300MB) |
| **Local Inference** | `llama_flutter_android` plugin |
| **Remote Inference** | `llama-cpp-python` via FastAPI |
| **Local Storage** | SQLite + SharedPreferences + FlutterSecureStorage |
| **Deployment** | Docker Compose |
| **Dart Files** | 49 |
| **Python Files** | 16 |

---

## 2. Architecture

### 2.1 High-Level Data Flow

```
User Input
    │
    ▼
ChatScreen
    │
    ▼
TutorBridge           ← Session management, history trimming
    │
    ▼
HybridTutorService    ← Orchestrator: chooses engine based on platform + connectivity
    ├── Local:  SocraticLlmService  → llama_flutter_android (Android ARM64 only)
    └── Remote: RemoteLlmService   → FastAPI :8000/chat → InferenceEngine → Qwen3-0.6B GGUF
```

### 2.2 Hybrid Routing Logic

```
Is platform Desktop (Linux/Windows/macOS)?
    YES → Always Remote
    NO  ↓
Is mode forced (online/offline)?
    YES → Use forced engine
    NO  ↓
Is there internet connectivity?
    NO  → Local
    YES → Remote (with liveness check)
           └── Is architecture x86_64?
                   YES → Remote (incompatible with native plugin)
                   NO  → Local eligible
```

### 2.3 Response Pipeline

```
User message
    → System prompt prepended (Socratic guardrails)
    → Last N messages of history appended (5 frontend / 8 backend)
    → ChatML format: <|im_start|>...<|im_end|>
    → Model inference (temperature=0.3, max_tokens=128–256)
    → <think>...</think> blocks stripped
    → Final question returned to UI
```

---

## 3. Backend Deep Dive

### 3.1 Entry Point — `backend/main.py`

```
FastAPI App
├── CORS Middleware: allow_origins=["*"]   ← SECURITY ISSUE
├── GET  /           → Health check
└── POST /chat       → ChatRequest → InferenceEngine → ChatResponse
```

No authentication, no rate limiting, errors caught globally and returned as HTTP 500.

### 3.2 ML Layer

| File | Role |
|------|------|
| `backend/ml/model_loader.py` | Singleton — downloads from HuggingFace if absent, loads via `llama_cpp.Llama()` |
| `backend/ml/inference_engine.py` | Lazy-loads model on first request, builds ChatML prompt, strips `<think>` blocks |
| `backend/ml/socratic_prompts.py` | System prompt templates per difficulty level (beginner / intermediate / advanced) |
| `backend/evaluation/metrics.py` | `AdaptiveMetrics` — calculates Socratic index, recommends scaffolding level |

**Model configuration (hardcoded in code, ignoring `config.json`):**

```python
n_ctx      = 2048    # config.json says 4096 — mismatch
n_threads  = 4
temperature = 0.3
max_tokens  = 256
```

### 3.3 Backend Auth — `backend/core/auth.py`

- Users stored in `backend/data/users.json` as a plaintext JSON file
- Passwords hashed with **SHA256, no salt** — vulnerable to rainbow table attacks
- No token issuance, no session management

### 3.4 Empty / Incomplete Files

These files exist but contain no implementation:

```
backend/core/tutor_system.py        ← empty
backend/core/dialogue_controller.py ← empty
backend/core/session_manager.py     ← empty
backend/ml/fine_tune_lora.py        ← empty
```

These suggest an abandoned or in-progress refactor.

### 3.5 `config.json` — Loaded but Never Used

`backend/data/config.json` contains complete, correct configuration:

```json
{
  "model": {
    "n_ctx": 4096,
    "n_threads": 4,
    "temperature": 0.7,
    "max_tokens": 128
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "cors_origins": ["*"]
  }
}
```

None of these values are read by the application — all are hardcoded in Python files.

---

## 4. Frontend Deep Dive

### 4.1 Navigation Structure

```
main.dart
└── MaterialApp
    ├── SplashScreen        → checks auth state
    ├── AuthScreen          → Login / Register
    └── HomeScreen          → 4-tab BottomNavigationBar
        ├── Tab 0: Courses
        ├── Tab 1: AI Tutor (ChatTab → ChatScreen)
        ├── Tab 2: Profile
        └── Tab 3: Settings
```

### 4.2 Key Services

| Service | Role | Issues |
|---------|------|--------|
| `HybridTutorService` | Orchestrates local/remote engine selection | No timeout on liveness check |
| `SocraticLlmService` | Local inference via `llama_flutter_android` | Android-only, 569 lines |
| `RemoteLlmService` | Remote inference via HTTP POST | Hardcoded `10.0.2.2:8000`, fake streaming |
| `TutorBridge` | Session lifecycle + message routing | Tightly coupled to `HybridTutorService` |
| `AuthService` | Login / Register with local fallback | No password salt, hardcoded URL |
| `CourseService` | Load courses (assets → local → remote) | Reloads on every tab switch |
| `SessionService` | Persist sessions in SharedPreferences | No pagination, predictable ID |
| `DatabaseService` | SQLite user store | No encryption, no migration |

### 4.3 State Management

Uses the **Provider** pattern with three global providers:

```dart
MultiProvider(
  providers: [
    ChangeNotifierProvider(create: (_) => ThemeService()),
    ChangeNotifierProvider(create: (_) => ModelDownloadService()),
    ChangeNotifierProvider(create: (_) => AuthService()),
  ],
)
```

Session state lives in `TutorBridge`, not in a Provider — meaning it is not globally accessible and resets if the object is re-instantiated.

### 4.4 Local LLM Initialization Flow

```
SocraticLlmService.initialize()
    ├── Platform check (Desktop → false)
    ├── Architecture check (x86_64 → false)
    ├── Get ApplicationSupportDirectory
    ├── Check local storage for model file
    ├── Fallback: copy from assets bundle
    └── LlamaController.load(path, threads=4, contextSize=4096, chat_format="chatml")
```

Uses `_initializationFailed` flag to prevent retry loops on hard failures.

### 4.5 Remote Streaming — Fake Implementation

```dart
// remote_llm_service.dart — generateResponse()
// Splits full response into words and yields with 50ms delay
for (final word in words) {
  yield word + ' ';
  await Future.delayed(const Duration(milliseconds: 50));
}
```

This is simulated streaming, not real token streaming. It adds unnecessary latency and is misleading.

---

## 5. Critical Problems

### 5.1 Security Issues

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| 1 | SHA256 password hashing **without salt** — rainbow table vulnerable | **Critical** | `auth_service.dart`, `auth.py` |
| 2 | `/chat` endpoint has **no authentication** — anyone can call it | **High** | `main.py` |
| 3 | CORS `allow_origins=["*"]` — exposes API to CSRF | **High** | `main.py` |
| 4 | `users.json` stores password hashes in a **plaintext file** | **High** | `backend/data/users.json` |
| 5 | SQLite database is **unencrypted** on device | **Medium** | `database_service.dart` |
| 6 | HTTP (not HTTPS) used for all API communication | **Medium** | All services |

### 5.2 Hardcoded Values That Break in Production

```
10.0.2.2:8000   ← Android emulator magic IP, hardcoded in 3 files:
                   remote_llm_service.dart
                   auth_service.dart
                   course_service.dart

"Socratic-Qwen3-0.6-Merged-Quality_Data-752M-Q4_K_M (1).gguf"
                ← Model filename with spaces and parentheses (fragile)

n_ctx = 2048    ← Hardcoded in inference_engine.py
                   config.json correctly says 4096 — never read
```

### 5.3 Dead / Orphaned Code

- `AdaptiveMetrics` in `metrics.py` calculates scaffolding level dynamically but is **never called** by the inference engine
- 4 empty backend files suggest incomplete refactoring
- `config.json` exists with proper configuration but is never loaded

---

## 6. Performance Issues

### High Impact

**1. Course reload on every tab switch**

`getCourses()` reads from disk/assets every time the Courses tab becomes active. There is no state preservation across tab switches.

```dart
// home_screen.dart — every tab switch re-creates children
body: _screens[_currentIndex]  // new instance, state lost
```

**Fix:** Use `IndexedStack` or apply `AutomaticKeepAliveClientMixin` to tabs.

---

**2. Fake streaming adds unnecessary latency**

The remote engine splits words and adds a 50ms delay per word. A 50-word response takes 2.5 seconds of artificial delay on top of actual network + inference time.

**Fix:** Implement Server-Sent Events (SSE) on FastAPI with `StreamingResponse`.

---

**3. History window inconsistency**

| Location | History limit |
|----------|--------------|
| Backend (`inference_engine.py`) | Last 8 messages |
| Frontend (`tutor_bridge.dart`) | Last 5 messages |

The model receives different context depending on which engine is active, leading to inconsistent behavior.

**Fix:** Choose one value and enforce it in one place.

---

### Medium Impact

**4. No timeout on remote liveness check**

```dart
// hybrid_tutor_service.dart
final isRemoteAvailable = await _remoteEngine.initialize();
// No timeout — hangs indefinitely if server is slow
```

**Fix:** Wrap in `Future.timeout(Duration(seconds: 3))`.

---

**5. SharedPreferences for full session history**

All messages in all sessions are serialized to JSON and stored in SharedPreferences. Long conversations or many sessions will hit platform size limits and slow down reads.

**Fix:** Move session storage to SQLite with proper indexing.

---

**6. No database indexes**

User lookups in SQLite use email and username as search keys with no indexes defined.

```dart
// database_service.dart — no CREATE INDEX statements
'CREATE TABLE users(id TEXT PRIMARY KEY, username TEXT, email TEXT, ...)'
```

**Fix:** Add indexes on `email` and `username` columns.

---

### Low Impact

**7. CPU thread count never adapts to device**

`threads: 4` is hardcoded in both the backend Docker config and the Flutter LLM service. High-end devices could use more threads; low-end devices may struggle with 4.

---

## 7. Architecture & Code Quality Issues

### 7.1 Tight Coupling

`TutorBridge` directly instantiates `HybridTutorService`, which directly instantiates both engines. There is no interface/abstraction that would allow swapping engines in tests or in future implementations.

### 7.2 Mixed Responsibilities

- `AuthService` handles registration + login + HTTP + SQLite + secure storage
- `CourseService` handles loading + caching + persistence + remote fetching

These should be split into repositories (data access) and services (business logic).

### 7.3 No Global Error Handling

Errors are caught locally in each service and surfaced as strings or generic error messages. There is no global error provider or error boundary — errors in one service are invisible to other parts of the UI.

### 7.4 Inconsistent Logging

```python
# Backend mixes print() and no logging at all
print(f"Loading model from {model_path}")
```

```dart
// Frontend mixes print() and debugPrint()
print('LLM Service: Initializing...');
debugPrint('Error: $e');
```

No structured logging in either layer.

### 7.5 Magic Numbers

```dart
// Various token limits scattered across files
max_tokens: 128    // config.json
max_tokens: 150    // remote_llm_service.dart
max_tokens: 256    // inference_engine.py
max_tokens: 350    // llm_service.dart
```

No single source of truth for inference parameters.

### 7.6 Git History

All 5 recent commits have the identical message:

```
feat: comprehensive app upgrade, 4-tab navigation, and registration fixes
```

This makes `git log`, `git blame`, and `git bisect` useless for tracking changes.

---

## 8. What's Actually Good

| Strength | Detail |
|----------|--------|
| **Hybrid offline-first design** | Clean fallback chain: local → remote → graceful error |
| **Socratic guardrails** | System prompt is well-crafted and consistently applied across both engines |
| **`_initializationFailed` flag** | Prevents crash/retry loops when LLM fails to load |
| **Platform-aware routing** | Correctly detects desktop, x86_64 emulators, and ARM64 |
| **Asset fallback chain** | CourseService: assets → local storage → remote is resilient |
| **`<think>` block stripping** | Handled correctly in both frontend and backend |
| **Consistent theming** | `AppTheme` properly centralizes colors, typography, and gradients |
| **Light/dark mode** | Correctly implemented with `ThemeService` as a `ChangeNotifier` |
| **Adaptive metrics built** | `AdaptiveMetrics` is well-designed — just needs to be wired up |

---

## 9. Dependencies

### 9.1 Flutter (`pubspec.yaml`)

```yaml
provider: ^6.1.2                  # State management
shared_preferences: ^2.2.3        # Local key-value storage
path_provider: ^2.1.5             # Directory access
http: ^1.6.0                      # HTTP client
dio: ^5.7.0                       # Advanced HTTP (used in auth)
connectivity_plus: ^6.1.1         # Network detection
llama_flutter_android: ^0.1.1     # Local LLM — Android only, may be stale
flutter_markdown: ^0.6.22         # Markdown rendering
flutter_secure_storage: ^10.0.0   # Encrypted key-value storage
sqflite: ^2.4.2                   # SQLite
sqlite3_flutter_libs: ^0.5.24     # SQLite binaries
crypto: ^3.0.7                    # SHA256 (used for passwords — see security issues)
uuid: ^4.5.2                      # UUID generation
```

**Concerns:**
- `llama_flutter_android: ^0.1.1` is a very early version — check for updates or breakage
- `http` and `dio` both present — consolidate to one HTTP client
- No pinned versions — `^` constraints allow silent breaking changes

### 9.2 Python (`requirements.txt`)

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
llama-cpp-python>=0.2.50
python-multipart>=0.0.6
httpx>=0.26.0
```

**Concerns:**
- No version pinning — `>=` constraints are not reproducible
- `llama-cpp-python` requires a C++ compiler to build from source
- GPU support commented out — fine for development, needs docs for production
- No `pip freeze > requirements-lock.txt` equivalent

---

## 10. Recommended Action Plan

### Immediate — Fix Before Any Production Use

- [ ] **Replace SHA256 with bcrypt** — `passlib[bcrypt]` on Python, delegate hashing to backend (never hash on client)
- [ ] **Add JWT authentication** to `/chat` and all backend endpoints
- [ ] **Restrict CORS** to specific origins (your app's domain / IP)
- [ ] **Move backend URL to an environment variable** — one `.env` file, consumed by Flutter via `--dart-define` at build time
- [ ] **Load `config.json`** in the backend instead of hardcoded values everywhere

### Short Term — Within 2–4 Weeks

- [ ] **Wire up `AdaptiveMetrics`** — it's already built, just connect it to `InferenceEngine.generate_response()` and pass scaffolding level to the system prompt
- [ ] **Implement SSE streaming** on FastAPI using `StreamingResponse` + `asyncio` — replace the fake word-delay streaming
- [ ] **Fix tab state preservation** — use `IndexedStack` in `home_screen.dart` to prevent tab rebuild on switch
- [ ] **Add timeout to remote liveness check** — `Future.timeout(Duration(seconds: 3))` in `hybrid_tutor_service.dart`
- [ ] **Standardize history window** — pick one limit (suggest 8 messages) and enforce it only in `TutorBridge`
- [ ] **Write unit tests** for `TutorBridge`, `AuthService`, `SessionService`, and `InferenceEngine`

### Medium Term — Within 1–3 Months

- [ ] **Move session storage from SharedPreferences to SQLite** with pagination support
- [ ] **Add structured logging** — `loguru` for Python, `logger` package for Flutter
- [ ] **Add database schema migrations** — version the SQLite schema properly
- [ ] **Add Docker health check** and resource limits to `docker-compose.yml`
- [ ] **Implement rate limiting** on the FastAPI backend (`slowapi` or similar)
- [ ] **Consolidate HTTP clients** — pick either `http` or `dio` in Flutter, not both
- [ ] **Delete or implement empty files** — either flesh out `tutor_system.py`, `dialogue_controller.py`, etc., or remove them

### Long Term — 3+ Months

- [ ] **Replace `users.json` with a real database** (SQLite minimum, PostgreSQL for multi-instance)
- [ ] **Add iOS support** — requires alternative to `llama_flutter_android` (e.g., `llama.cpp` via FFI on iOS)
- [ ] **Implement OAuth** for third-party login
- [ ] **Add analytics** for learning insights (session length, question quality, scaffolding level trends)
- [ ] **Add CI/CD pipeline** with automated testing and linting on every commit

---

## File Reference Map

### Backend

| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI app entry point |
| `backend/ml/inference_engine.py` | Response generation |
| `backend/ml/model_loader.py` | Model download + loading |
| `backend/ml/socratic_prompts.py` | Prompt templates |
| `backend/core/auth.py` | User storage (JSON-based) |
| `backend/evaluation/metrics.py` | Adaptive scaffolding metrics (orphaned) |
| `backend/data/config.json` | Configuration (not read by app) |
| `backend/data/users.json` | User database (plaintext) |
| `docker-compose.yml` | Container orchestration |
| `requirements.txt` | Python dependencies |

### Frontend

| File | Purpose |
|------|---------|
| `socratic_app/lib/main.dart` | App entry point |
| `socratic_app/lib/screens/home_screen.dart` | 4-tab navigation hub |
| `socratic_app/lib/screens/chat_screen.dart` | Main tutoring UI |
| `socratic_app/lib/services/hybrid_tutor_service.dart` | Engine orchestrator |
| `socratic_app/lib/services/llm_service.dart` | Local LLM (569 lines) |
| `socratic_app/lib/services/remote_llm_service.dart` | Remote API client |
| `socratic_app/lib/services/tutor_bridge.dart` | Session + message bridge |
| `socratic_app/lib/services/auth_service.dart` | Auth (login/register) |
| `socratic_app/lib/services/database_service.dart` | SQLite user store |
| `socratic_app/lib/services/course_service.dart` | Course loading |
| `socratic_app/lib/services/session_service.dart` | Session persistence |
| `socratic_app/lib/theme/app_theme.dart` | Global theming |
| `socratic_app/pubspec.yaml` | Flutter dependencies |

---

*This review covers the full project as of the `main` branch snapshot on 2026-02-20.*
