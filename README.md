# Socratic AI Tutor: Hybrid Offline-First Personalized Learning

[![Flutter](https://img.shields.io/badge/Frontend-Flutter-blue.svg)](https://flutter.dev)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The **Socratic AI Tutor** is a self-contained, **hybrid offline-first** mobile application and backend system designed to revolutionize self-learning through the **Socratic Method**. Instead of providing direct answers, this system acts as a guide, asking focused questions to help students discover concepts through their own reasoning.

---

## 🚀 Key Features

*   **🧠 Socratic Guardrails**: Strictly enforced pedagogical logic—the AI never gives direct answers and focuses on scaffolding knowledge.
*   **🔄 Hybrid Intelligence**: Intelligently switches between high-speed local inference (100% offline) and high-accuracy remote inference depending on connectivity and priority.
*   **📶 100% Offline Inference**: No internet required. Powered by quantized GGUF models running locally on ARM64 processors (Android/iOS).
*   **⚡ Hardware Accelerated**: Uses efficient native libraries (`llama_flutter_android`) for low-latency, on-device Socratic reasoning.
*   **📚 Curriculum-Based**: Integrated course library (Programming, Data Science, Critical Thinking) with metadata and lesson modules.
*   **💾 Local Persistence**: Sessions and progress are saved locally, allowing students to pick up where they left off.
*   **🎨 High-Contrast UI**: Refined "Modern Orange and Dark Blue" theme optimized for readability and accessibility in both Light and Dark modes.

---

## 🛠️ Tech Stack

### Frontend (Flutter)
- **State Management**: `Provider`
- **Native LLM Engine**: `llama_flutter_android` (C++ backend)
- **Hybrid Orchestration**: `HybridTutorService` for seamless cloud/edge switching.
- **Local Storage**: `SharedPreferences` & `PathProvider`
- **Networking**: `Dio` & `ConnectivityPlus` (for optional fallback to remote API)

### ML Layer
- **Base Model**: Qwen3-0.6B (Optimized for mobile inference)
- **Dataset**: ~37,000 Socratic dialogue turns training the model on pedagogical reasoning.
- **Quantization**: GGUF (Q4_K_M) for 100% offline mobile utilization.
- **Tuning**: LoRA-based fine-tuning with integrated "Thought Chains" for teaching logic.

### Backend (Python/FastAPI)
- **Framework**: FastAPI
- **Inference**: `llama-cpp-python`
- **Metrics**: Adaptive performance tracking and log analysis.

---

## �️ Hybrid Architecture

The **Socratic AI Tutor** implements a specialized hybrid routing system:
1. **Edge-First**: The system primarily attempts local inference on-device using the bundled GGUF model for privacy and zero-latency.
2. **Cloud-Fallback**: If the device hardware is insufficient (e.g., x86 emulators or low-RAM devices) or if high-precision reasoning is required, the system seamlessly transitions to the FastAPI-based remote inference engine.
3. **Connectivity Aware**: Real-time monitoring via `ConnectivityPlus` ensures the best available engine is selected automatically.

---

## �📂 Project Structure

```text
socratic_ai_tutor/
├── backend/            # FastAPI server & evaluation logic
│   ├── core/           # Tutor system orchestrators
│   ├── ml/             # Inference engine & prompt templates
│   ├── data/           # Configs & instructional content
│   └── main.py         # Backend entry point
├── socratic_app/       # Flutter mobile application
│   ├── lib/            # Dart source code (UI/Services/Models)
│   ├── assets/         # Course content & bundled models
│   └── android/        # Native Android configurations
├── notebooks/          # Training and quantization research
│   ├── training/       # LoRA fine-tuning notebooks
│   └── quantization/   # GGUF conversion scripts
└── models/             # Local GGUF model files
```

---

## ⚙️ Getting Started

### 1. Prerequisites
- Python 3.10+
- Flutter SDK (latest stable)
- Android Studio / Xcode

### 2. Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
python main.py
```

### 3. Frontend Setup
```bash
cd frontend
flutter pub get
flutter run
```

### 4. Model Setup
Download the quantized model (e.g., `socratic-q4_k_m.gguf`) and place it in:
- `backend/models/` (for server-side inference)
- `frontend/assets/models/` (for mobile-native inference)

---

## 📖 Socratic Guardrails
The application is strictly programmed to follow these pedagogical rules:
1. **No Direct Answers**: The AI identifies when it is being asked for a solution and pivots to a guiding question.
2. **Scaffolding**: Complex problems are broken down into smaller, manageable inquiries.
3. **Logic Verification**: The AI analyzes student reasoning to identify knowledge gaps and adapts dynamically.
4. **Visible Reasoning**: The model uses `<think>` tags to internalize pedagogical strategy before generating its response.

---

## 📈 ML Development
The project includes specialized notebooks for model development:
- **`training/Qwen3_0_6B.ipynb`**: Process for fine-tuning the base model on Socratic dialogue examples using LoRA.
- **`quantization/gguf_quantization.ipynb`**: Techniques used to compress 1.5B+ parameter models down to <1.5GB for mobile performance.

## 📚 Content Library
The tutor's intelligence is supplemented by a structured curriculum across multiple domains:
- **Mathematics**: Calculus, Algebra, and Geometry.
- **Programming**: Algorithms, Data Structures, and Software Design.
- **Science**: Physics and Biology.
- **Critical Thinking**: Logic, Ethics, and Epistemology.

---

## 🛠️ System Prompt Example
The AI's behavior is governed by a strict system prompt that ensures pedagogical integrity:
```text
You are a Socratic AI tutor specializing in data science and machine learning.

Your core teaching philosophy:
1. NEVER provide direct answers or explanations
2. ALWAYS respond with thoughtful guiding questions
3. Help learners discover answers through their own reasoning
4. Use questions that prompt reflection, analysis, and critical thinking

Response format:
- Begin with <think>...</think> to show your pedagogical reasoning
- Follow with a single, focused question that guides the learner
```

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.
