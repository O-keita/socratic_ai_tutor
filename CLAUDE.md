# CLAUDE.md — Socratic AI Tutor

Developer reference for the Socratic AI Tutor capstone project. Read this before touching anything.

---

## Project Overview

Offline-first mobile app that teaches data science and machine learning using the Socratic method. The AI never gives direct answers — it responds with guiding questions to develop critical thinking.

**Stack:**
- Flutter (Dart) — mobile frontend
- FastAPI (Python) — backend inference server
- Qwen3-0.6B (GGUF Q4_K_M, ~300 MB) — the LLM powering the tutor
- `llama_flutter_android` — on-device inference (ARM64 Android only)
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
│   ├── lib/
│   │   ├── main.dart
│   │   ├── screens/          # 18 screens
│   │   ├── services/         # 15 services (see Services section)
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
├── models/                   # GGUF model file (not in git, ~300 MB)
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
Desktop (Linux/Windows/macOS)?  → Always remote
Forced mode set?                → Use that mode
Network available?              → Remote (3-second liveness ping)
No network / offline mode?      → Local GGUF (if downloaded)
x86_64 arch (emulator)?        → Always remote (ARM64-only plugin)
```

`llama_flutter_android` only works on **ARM64 Android**. On x86_64 emulators it crashes the native plugin channel, which also breaks `path_provider`. See the Known Issues section.

---

## Key Services

| Service | Purpose |
|---------|---------|
| `HybridTutorService` | Routes LLM requests local vs remote |
| `SocraticLlmService` (llm_service.dart) | On-device GGUF inference |
| `RemoteLlmService` | HTTP calls to FastAPI `/chat` |
| `AuthService` | Login/register, JWT, current user state |
| `CourseService` | Load bundled courses; fetch remote courses |
| `ModelDownloadService` | Download the GGUF model to device storage |
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

The model uses **ChatML format**:
```
<|im_start|>system
You are a Socratic tutor...
<|im_end|>
<|im_start|>user
{message}
<|im_end|>
<|im_start|>assistant
```

The system prompt enforces Socratic guardrails — the model must never directly answer questions, only guide with questions. `<think>...</think>` blocks in the output are stripped before sending to the client.

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

### x86_64 emulator hang on startup
`llama_flutter_android` is ARM64-only. On x86_64 emulators, the native crash blocks **all** Flutter plugin channels, including `path_provider`. This causes a hang during initialization.

Workarounds already in place:
- `SocraticLlmService._getApplicationDataDirectory()`: max 2 retries, 500 ms flat delay
- `HybridTutorService.initialize()`: 5-second timeout on local engine init
- `ModelDownloadService.isModelDownloaded()`: try/catch + 2-second timeout, returns `false` on any failure

Use a physical ARM64 device or set the engine to "Online only" in Settings when running on x86_64.

### Login stuck on login screen
Caused by `isModelDownloaded()` throwing before the fix above. If you see this after a regression, check that `ModelDownloadService.isModelDownloaded()` still has its try/catch.

### RenderFlex overflows
Several screens have `Expanded`/`Flexible` wrappers added to prevent overflow on small screens. Don't remove them when refactoring layout code:
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

# Rebuild launcher icons from assets/images/logo.png
cd socratic_app && dart run flutter_launcher_icons

# Build release APK
cd socratic_app && flutter build apk --release --split-per-abi

# Run backend in Docker
docker-compose up --build

# Check Flutter issues
flutter analyze
flutter doctor
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
flutter build apk --release --split-per-abi
# Outputs:
#   build/app/outputs/flutter-apk/app-arm64-v8a-release.apk  ← use this one
#   build/app/outputs/flutter-apk/app-armeabi-v7a-release.apk
```

Before building, set the production backend URL in `lib/utils/app_config.dart`.
