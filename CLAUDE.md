# CLAUDE.md — Socratic AI Tutor

Developer reference for the Socratic AI Tutor capstone project. Read this before touching anything.

---

## Project Overview

Offline-first mobile app that teaches data science and machine learning using the Socratic method. The AI never gives direct answers — it responds with guiding questions to develop critical thinking.

**Stack:**
- Flutter (Dart) — mobile frontend
- FastAPI (Python) — backend inference server
- Qwen3-0.6B (GGUF Q4_K_M, ~460 MB) — the LLM powering the tutor
- `libchat` — custom C API wrapping llama.cpp, compiled from source via CMake into the APK
- `dart:ffi` — Dart FFI bindings to libchat.so (no third-party inference packages)
- `llama-cpp-python` — server-side inference

---

## Repo Layout

```
socratic_ai_tutor/
├── backend/                  # FastAPI server
│   ├── main.py               # All routes (auth, chat, content, admin API)
│   ├── core/
│   │   ├── auth.py           # User registration + JWT
│   │   └── admin_auth.py     # Admin session management
│   ├── ml/
│   │   ├── inference_engine.py
│   │   ├── model_loader.py
│   │   └── socratic_prompts.py
│   ├── evaluation/           # Adaptive metrics & session analysis
│   ├── data/
│   │   ├── config.json       # Model params, difficulty levels, admin key
│   │   └── users.json        # File-based user DB (no SQL server needed)
│   └── content/              # Runtime-uploaded courses (not in git)
├── socratic_app/             # Flutter app
│   ├── android/app/src/main/cpp/
│   │   ├── libchat.h         # C API header (4 functions)
│   │   ├── libchat.cc        # Implementation (~235 lines)
│   │   ├── CMakeLists.txt    # Builds llama.cpp from source + libchat
│   │   └── LIBCHAT.md        # Full documentation of the C API
│   ├── lib/
│   │   ├── main.dart
│   │   ├── screens/          # UI screens
│   │   ├── services/         # Business logic services
│   │   ├── models/           # Dart data classes
│   │   ├── widgets/          # Reusable UI components
│   │   ├── theme/app_theme.dart
│   │   └── utils/app_config.dart
│   └── assets/
│       ├── courses/          # Bundled course JSON + lesson markdown
│       ├── quizzes/
│       ├── glossary/
│       ├── images/logo.png
│       └── playground/index.html
├── local_llama_cpp_dart/     # llama.cpp source (not in git, ~208 MB)
├── webapp/                   # React web app (admin + student portal)
│   ├── src/
│   │   ├── App.tsx           # Routing
│   │   ├── context/          # Auth + Progress providers
│   │   ├── api/              # All HTTP calls
│   │   └── components/       # UI components
│   └── vite.config.ts        # Dev server, proxies /api → :8000
├── notebooks/training/       # Fine-tuning notebooks (Colab)
├── models/                   # GGUF model file (not in git, ~460 MB)
├── docker-compose.yml
└── requirements.txt
```

---

## Running the Project

### Backend

```bash
# First time
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Put the GGUF model in:
#   models/socratic-q4_k_m.gguf

# Run dev server
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or via Docker
docker-compose up --build
```

Admin panel: `http://localhost:8000/admin`
API docs: `http://localhost:8000/docs`

### Flutter App

```bash
cd socratic_app
flutter pub get
flutter run                   # connects to backend at AppConfig.backendUrl
```

To point the app at your local machine from an emulator:
- Edit `socratic_app/lib/utils/app_config.dart`
- Android emulator uses `10.0.2.2` to reach `localhost`
- Physical device needs your machine's LAN IP

---

## Architecture: Hybrid Edge-Cloud Routing

The most important concept in this codebase. **`HybridTutorService`** (`lib/services/hybrid_tutor_service.dart`) routes every AI request to either the on-device model or the remote backend:

```
User forced mode?               → Use that mode (online/offline)
Auto mode + network available?  → Remote (3-second liveness ping)
Auto mode + no network?         → Local GGUF (if downloaded)
Previous inference crashed?     → Block local, route to remote (SIGILL protection)
Previous model load crashed?    → Block local, route to remote (OOM protection)
```

On-device inference uses **libchat** (our custom C API wrapping llama.cpp via `dart:ffi`). Works on **ARM64 Android** only.

---

## On-Device Inference: libchat C API

Instead of third-party Flutter packages (which have broken builds, missing submodules, or wrong CPU targets), we compile llama.cpp directly from source into the APK via CMake.

**Architecture:**
```
Dart (llm_service.dart) → dart:ffi → libchat.so → llama.cpp (compiled from source)
```

**API (4 functions):**
```c
chat_session * chat_create(const char * model_path, int n_ctx, int n_threads);
char * chat_generate(chat_session * session, const char * user_message);
void chat_string_free(char * str);
void chat_destroy(chat_session * session);
```

**Key details:**
- Chat template auto-detected from GGUF metadata (works with ChatML, Qwen, Llama, etc.)
- Conversation history managed natively in `chat_session` struct
- CPU-only, ARM64 baseline (no dotprod/i8mm/SVE — runs on all ARM64 devices)
- No OpenMP, OpenCL, or Vulkan dependencies
- Inference runs in `Isolate.run()` to keep Flutter UI responsive
- FFI bindings re-resolved inside the isolate (Dart isolates don't share state)

**Files:**
- `socratic_app/android/app/src/main/cpp/libchat.h` — C API header
- `socratic_app/android/app/src/main/cpp/libchat.cc` — Implementation
- `socratic_app/android/app/src/main/cpp/CMakeLists.txt` — Build config
- `socratic_app/android/app/src/main/cpp/LIBCHAT.md` — Full documentation
- `socratic_app/lib/services/llm_service.dart` — Dart FFI bindings

**llama.cpp source:** Located at `local_llama_cpp_dart/src/llama.cpp/` (referenced by CMakeLists.txt via relative path). Not checked into git (~208 MB).

See `LIBCHAT.md` for the full standalone documentation including setup guide and Dart library plan.

---

## Key Services

| Service | Purpose |
|---------|---------|
| `HybridTutorService` | Routes LLM requests local vs remote |
| `SocraticLlmService` (llm_service.dart) | On-device GGUF inference via libchat FFI |
| `RemoteLlmService` | HTTP calls to FastAPI `/chat` |
| `AuthService` | Login/register, JWT, current user state |
| `CourseService` | Load bundled courses; fetch remote courses |
| `ModelDownloadService` | Download the GGUF model from HuggingFace |
| `ThemeService` | Light/dark mode (`ChangeNotifier`) |
| `ProgressService` | Track completed lessons (SharedPreferences) |

All services are singletons (factory constructors). `AuthService`, `ThemeService`, `ModelDownloadService` are registered as `ChangeNotifierProvider` in `main.dart`.

---

## Screens

| Screen | Route / Entry |
|--------|--------------|
| `SplashScreen` | App start — checks auth → home or login |
| `AuthScreen` | Login + register tabs |
| `HomeScreen` | 4-tab shell (Courses / AI Tutor / Profile / Settings) |
| `CoursesScreen` | Course list (standalone tab via `IndexedStack`) |
| `ChatScreen` | Free-form Socratic conversation |
| `CourseDetailScreen` | Module/chapter/lesson tree |
| `LessonScreen` | Markdown lesson content |
| `LessonChatScreen` | Chat scoped to a lesson topic |
| `QuizScreen` | Multiple-choice quizzes |
| `GlossaryScreen` | Searchable term definitions |
| `PlaygroundScreen` | Python playground (Pyodide in WebView) |
| `ProfileScreen` | User stats and progress |
| `SettingsScreen` | Theme, server URL, model management |
| `ModelSetupScreen` | First-time GGUF download flow |

`HomeScreen` uses `IndexedStack` so all tabs stay alive when switching — no state loss on tab change.

---

## Content System

### Bundled Content (offline, ships in APK)

Assets are declared in `pubspec.yaml` and compiled into the APK:
```
assets/courses/{courseId}/course.json      # Course metadata + structure
assets/courses/{courseId}/lessons/*.md     # Lesson markdown files
```

`CourseService.getCourses()` loads ONLY bundled content — fast, no network.

### Remote Content (admin-uploaded, cached locally)

The backend serves courses from `backend/content/` via:
- `GET /content/manifest` — list of all server-side courses
- `GET /content/{courseId}/course.json`
- `GET /content/{courseId}/lessons/{filename}`

The "Fetch new courses" button in the app calls `CourseService.fetchRemoteCourses()`, which merges remote courses into the list and caches them in `getApplicationSupportDirectory()` for offline use.

Admin uploads courses through the web panel at `/admin` using the API key from `backend/data/config.json → admin.api_key`.

### Adding a New Bundled Course

1. Create `assets/courses/{courseId}/course.json` (match the schema in existing courses)
2. Add lesson markdown files to `assets/courses/{courseId}/lessons/`
3. Add the course ID to `assets/courses/courses.json`
4. Register the new paths in `pubspec.yaml` under `flutter: assets:`
5. Run `flutter pub get`

---

## Backend API

### Public endpoints (no auth)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/register` | Create account |
| POST | `/login` | Returns JWT |
| POST | `/chat` | Socratic AI response |
| GET | `/content/manifest` | Course list |
| GET | `/content/{id}/course.json` | Course data |
| GET | `/content/{id}/lessons/{file}` | Lesson markdown |

### Admin endpoints (header: `X-Admin-Key: <key>`)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/admin/api/courses` | List server courses |
| POST | `/admin/api/courses/{id}` | Create/update course |
| DELETE | `/admin/api/courses/{id}` | Delete course |
| POST | `/admin/api/courses/{id}/lessons/{file}` | Save lesson |
| DELETE | `/admin/api/courses/{id}/lessons/{file}` | Delete lesson |

### Chat request format
```json
POST /chat
{
  "message": "What is gradient descent?",
  "session_id": "uuid",
  "user_id": "username",
  "difficulty": "intermediate"
}
```

---

## LLM & Prompting

The model uses **ChatML format** (auto-detected from GGUF metadata by llama.cpp):
```
<|im_start|>system
You are a Socratic AI tutor...
<|im_end|>
<|im_start|>user
{message}
<|im_end|>
<|im_start|>assistant
<think>reasoning here</think>
Response here
```

**System prompt** (matches the training notebook `Qwen3_0_6B.ipynb`):
```
You are a Socratic AI tutor specializing in data science and machine learning.

RULES:
1. ALWAYS begin your response with a thinking block containing your reasoning.
2. For conceptual questions: respond with ONE guiding question. NEVER give direct answers.
   If the student is stuck, give a small hint before your question.
3. For code questions: guide the student to write the code themselves through Socratic questioning.
4. For casual messages (greetings, thanks, chitchat): respond warmly and naturally.

Always start with a thinking block. This is mandatory.
```

**`<think>` block handling:**
- The Qwen3 model generates `<think>...</think>` blocks as internal reasoning
- `SocraticLlmService` strips them before yielding to the UI
- `RemoteLlmService` strips them from backend responses
- `ChatBubble._cleanMessage()` strips any that slip through as a final safety net

Prompt templates live in `backend/ml/socratic_prompts.py`. Difficulty levels (beginner/intermediate/advanced) adjust how much scaffolding the prompt instructs the model to provide.

Model config in `backend/data/config.json`:
```json
"model": {
  "path": "models/socratic-q4_k_m.gguf",
  "n_ctx": 4096,
  "temperature": 0.7,
  "max_tokens": 128
}
```

---

## Python Playground

`PlaygroundScreen` wraps a `WebViewWidget` loading `assets/playground/index.html`.

The HTML page uses:
- **CodeMirror 5** (from cdnjs CDN) — syntax-highlighted editor
- **Pyodide 0.27** (from jsDelivr CDN) — full Python compiled to WebAssembly

The HTML is loaded with `controller.loadHtmlString(html, baseUrl: 'https://localhost')`. The `baseUrl` is critical — without it, Android WebView's `file://` origin blocks Pyodide's CDN `fetch()` calls.

Pyodide loads ~8 MB on first open. Package installs use `micropip` (pre-loaded at boot). Both CDN resources require internet on first use; once the WebView caches them, subsequent loads are fast.

---

## Theme

Everything is in `lib/theme/app_theme.dart`.

Key color tokens:
```dart
AppTheme.accentOrange      // Primary brand color (#F97316)
AppTheme.primaryLight      // Background accent
AppTheme.surfaceDark       // Dark card surface
AppTheme.textPrimary       // Main text (dark mode)
AppTheme.textSecondary     // Secondary text
AppTheme.success           // #10B981 (green)
AppTheme.warning           // #F59E0B (amber)
```

Light/dark mode is managed by `ThemeService` (a `ChangeNotifier`). Persist the preference with `shared_preferences`. Read in widgets with `context.watch<ThemeService>().isDarkMode`.

---

## Known Issues & Workarounds

### Crash-loop protection
On-device inference can crash the app (OOM during model load, SIGILL on incompatible CPUs). The app uses SharedPreferences flags to detect and block repeat crashes:
- `SocraticLlmService._prefLoadingKey`: set before model load, cleared after. If OOM kills the process, flag persists → blocks local init on next launch.
- `SocraticLlmService._prefInferenceKey`: same pattern for inference crashes (SIGILL).
- User can clear flags via Settings → Reset Model.

### x86_64 emulator
libchat is compiled for ARM64 only. On x86_64 emulators the native code won't load. Use a physical ARM64 device or set "Online only" in Settings.

### RenderFlex overflows
Several screens have `Expanded`/`Flexible` wrappers to prevent overflow on small screens. Don't remove them:
- `home_screen.dart` `_buildToolCard` — `Text` in Row
- `chat_screen.dart` `_buildAppBar` — title Column inside AppBar Row
- `profile_screen.dart` `_buildProgressItem` — label Text in Row

---

## Common Commands

```bash
# Run backend locally
cd backend && uvicorn main:app --reload

# Run Flutter app (debug)
cd socratic_app && flutter run

# Build release APK (arm64 only, ~60 MB without model)
cd socratic_app && flutter build apk --release

# Install on connected device
flutter install --release

# Rebuild launcher icons from assets/images/logo.png
cd socratic_app && dart run flutter_launcher_icons

# Run backend in Docker
docker-compose up --build

# Web app dev server
cd webapp && npm run dev
```

---

## Authentication Flow

1. `SplashScreen` checks `AuthService.isLoggedIn` (stored JWT in `FlutterSecureStorage`)
2. Valid token → `HomeScreen`; no token → `AuthScreen`
3. `AuthScreen` calls `POST /register` or `POST /login`
4. On success: saves JWT, checks `ModelDownloadService.isModelDownloaded()`
5. Model exists → `HomeScreen`; missing → `ModelSetupScreen` (download flow)

JWT expiry is not yet handled — refresh logic would go in `AuthService`.

---

## Deployment

### Backend (Docker)
```bash
docker-compose up -d
```
The compose file mounts `./models/` so the GGUF file doesn't need to be inside the container.

### Android APK
```bash
cd socratic_app && flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk (~60 MB)
```

Before building:
1. Set the production backend URL in `lib/utils/app_config.dart`
2. Ensure `local_llama_cpp_dart/src/llama.cpp/` exists (CMake needs it)

### Web App
```bash
cd webapp && npm run build    # → dist/ (static files, serve behind backend)
```
