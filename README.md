# Socratic AI Tutor: Hybrid Offline-First Personalized Learning

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/O-keita/socratic_ai_tutor.git)
[![Flutter](https://img.shields.io/badge/Frontend-Flutter-blue.svg)](https://flutter.dev)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Project Description
The **Socratic AI Tutor** is a self-contained, **hybrid offline-first** mobile application and backend system designed to revolutionize self-learning through the **Socratic Method**, specifically tailored for technology education in low-resource environments. Instead of providing direct answers, this system acts as a guide, asking focused questions to help students discover concepts through their own reasoning.

By combining on-device LLM inference (GGUF) with a cloud-fallback API, it provides a seamless learning experience regardless of internet availability.

---

## 🔗 Project Resources
- **GitHub Repository**: [https://github.com/O-keita/socratic_ai_tutor.git](https://github.com/O-keita/socratic_ai_tutor.git)
- **Research Background**: Refer to our internal documentation on "Bridging the Digital Reasoning Divide in Africa".

---

## 🎨 Designs & Architecture

### 📱 App Interfaces
| Onboarding | Chat Interface | Course Library |
| :---: | :---: | :---: |
| ![Onboarding Placeholder](https://via.placeholder.com/200x400?text=Onboarding) | ![Chat Placeholder](https://via.placeholder.com/200x400?text=Socratic+Chat) | ![Library Placeholder](https://via.placeholder.com/200x400?text=Courses) |

### 🛠️ System Architecture (Flow Diagram)
```mermaid
graph TD
    A[User Prompt] --> B{Connectivity?}
    B -- Offline --> C[Local llama_flutter Engine]
    B -- Online --> D[Remote FastAPI Engine]
    C --> E[Socratic Guardrails]
    D --> E
    E --> F[Guiding Question]
```

---

## 🚀 Key Features

*   **🧠 Socratic Guardrails**: Strictly enforced pedagogical logic—the AI never gives direct answers and focuses on scaffolding knowledge.
*   **🔄 Hybrid Intelligence**: Intelligently switches between high-speed local inference (100% offline) and high-accuracy remote inference depending on connectivity and priority.
*   **📶 100% Offline Inference**: No internet required. Powered by quantized GGUF models running locally on ARM64 processors (Android/iOS).
*   **⚡ Hardware Accelerated**: Uses efficient native libraries (`llama_flutter_android`) for low-latency, on-device Socratic reasoning.
*   **📚 Curriculum-Based**: Integrated course library (Programming, Data Science, Critical Thinking) with metadata and lesson modules.
*   **💾 Local Persistence**: Sessions and progress are saved locally, allowing students to pick up where they left off.
*   **🎨 High-Contrast UI**: Refined "Modern Orange and Dark Blue" theme optimized for readability and accessibility in both Light and Dark modes.

### ☁️ Remote Server Deployment (Fallback Engine)

The system can optionally connect to a remote inference engine hosted on a **Virtual Private Server (VPS)** with 4GB+ RAM.

#### 1. Server Setup
```bash
# Clone the repository
git clone https://github.com/O-keita/socratic_ai_tutor.git
cd socratic_ai_tutor

# Ensure Docker & Docker Compose are installed
docker compose build
```

#### 2. Model Placement
Place the `socratic-q4_k_m.gguf` (~300MB) model in the root `models/` directory of the server. 

#### 3. Launch
```bash
docker compose up -d
```
The server will be available at your server's IP on port 8000.

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
- **Quantization**: GGUF (Q4_K_M) for ~300MB footprint, optimized for offline mobile utilization on commodity hardware.
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

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- **Flutter**: 3.19.0+ 
- **Python**: 3.10 or 3.11 (3.12 support depends on `llama-cpp-python` wheels)
- **Mobile Hardware**: Android device with ARM64 architecture (Local inference will not work on x86 emulators).
- **RAM**: 4GB+ RAM recommended for both backend and mobile device.

### 2. Backend Environment (FastAPI)
```bash
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies with build tools (required for llama-cpp)
pip install --upgrade pip
pip install -r ../requirements.txt

# Run the server
python main.py
```

### 3. Mobile App Environment (Flutter)
```bash
# Navigate to app folder
cd socratic_app

# Install dependencies
flutter pub get

# Connect an ARM64 Android device
flutter run
```

### 4. GGUF Model Setup
Currently, the system is designed for small, quantized models (~300MB).
1. Download or export the `.gguf` model (e.g., `socratic-q4_k_m.gguf`).
2. Place a copy in `backend/models/` for server-side fallback.
3. For mobile, the app can be configured to load from `assets` or download on first run.

## 🚢 Deployment Plan

### ☁️ Remote Backend
- **Host**: Any standard Linux VPS (4GB RAM minimum).
- **Containerization**: Use the provided `Dockerfile` and `docker-compose.yml`.
- **SSL/Production**: Recommended to use a reverse proxy (like Nginx) for HTTPS.
- **Orchestration**:
  ```bash
  docker compose up --build -d
  ```

### 📱 Mobile Application (Android/iOS)
1. **Model Delivery**: The ~300MB model file is hosted on a file server or model hub (e.g. Hugging Face).
2. **Post-Install Download**: Due to the file size (~300MB), the app is designed to download the model to local storage on first launch or during setup.
3. **Distribution**: 
   - **Android**: Distribution via APK or App Bundle.
   - **iOS**: Distribution via TestFlight or App Store.

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
- **`quantization/gguf_quantization.ipynb`**: Techniques used to compress LLMs down to ~300MB for mobile performance.

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
