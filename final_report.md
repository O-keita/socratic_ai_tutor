# Offline Edge-AI Socratic Tutor for Technology Education in Africa

## Dedication

I dedicate this work to my family for their unwavering support and encouragement throughout my academic journey. To my parents, who always believed in the power of education to transform lives, and to my siblings, for their constant motivation. This work is also dedicated to the learners across Africa who strive for knowledge despite technological barriers; may this project contribute in some small way to your success.

## Acknowledgements

I would like to express my sincere gratitude to my supervisor for their invaluable guidance, patience, and insightful feedback throughout the duration of this capstone project. Their expertise in artificial intelligence and commitment to educational equity significantly shaped the direction of this research.

I am also grateful to the faculty and staff of the department for providing a conducive environment for learning and research. Special thanks to my colleagues and friends for the stimulating discussions and technical assistance that helped resolve numerous challenges during the implementation phase.

Finally, I acknowledge the open-source community, particularly the developers of Pylance, Flutter, FastAPI, and llama.cpp, whose tools made this offline-first AI tutoring system a reality.

## Abstract

This project implemented an offline-capable Socratic AI tutoring system designed to support critical thinking in data science and machine learning education for learners in low-resource African contexts. The work addressed the problem that most existing AI tutors are cloud-based, answer-centric, and inaccessible where connectivity, hardware, and cost constraints are severe. To bridge this digital reasoning divide, we designed and deployed a hybrid edge–cloud architecture combining a compressed and LoRA-fine-tuned Qwen3-0.6B model running locally on Android devices with a FastAPI backend for remote inference. The mobile app delivers guided questioning rather than direct answers, integrating course content, quizzes, glossary, and a Python playground in a fully offline experience once the model is downloaded. Evaluation on real devices (including a Huawei P Smart) showed that the tutor achieved 5–8 second response latency in both online and offline modes, maintained Socratic behaviour through question-only responses and scaffolding, and was robust across heterogeneous hardware. The findings indicate that small, quantized LLMs with Socratic guardrails can feasibly support inquiry-oriented learning on commodity devices in African settings. The report concludes with recommendations for scaling the system, expanding content coverage, and conducting larger-scale educational impact studies.

---

## CHAPTER ONE: INTRODUCTION

### 1.1 Introduction and Background

Critical thinking is a foundational skill in technology education, where learners must analyse problems, evaluate competing solutions, and justify design decisions. However, many African education systems have historically been characterized by teacher-centred instruction, exam-driven curricula, and rote memorisation, limiting opportunities for inquiry, reflection, and independent reasoning. At the same time, advances in artificial intelligence have produced powerful intelligent tutoring systems and conversational agents that can provide personalised learning support at scale, but these systems are typically designed for well-resourced environments with stable internet and modern hardware.

This project responded to that gap by designing and implementing a Socratic AI tutor that runs on low-cost devices and can operate offline after an initial model download. Instead of giving direct answers, the tutor asks probing questions, encourages learners to explain intermediate steps, and scaffolds reflection. The implementation targeted data science and machine learning content, providing a practical demonstration of how compressed and LoRA-fine-tuned LLMs can support inquiry-based learning on edge devices in African contexts.

### 1.2 Problem Statement

Although AI-driven tutoring systems and large language models (LLMs) have shown strong potential for personalised learning, they remain largely inaccessible in many African settings. Most existing tools assume continuous high-speed connectivity, high-performance hardware, and access to commercial cloud services. Furthermore, many AI tutors are answer-centric and can be used to shortcut reasoning instead of developing critical thinking. Learners in rural and underserved communities thus face a double disadvantage: limited access to AI tools in general, and limited access to AI systems that promote Socratic, critical-thinking-oriented learning in particular.

This project addressed the problem of designing and deploying an AI tutoring system that:
- Runs on low-cost, locally available devices.
- Functions offline or with minimal connectivity.
- Enforces Socratic questioning strategies instead of direct solutions.

### 1.3 Project Main Objective

The main objective of the project was to design, implement, and evaluate an offline-capable Socratic AI tutoring system that leverages compressed and LoRA-fine-tuned LLMs to foster critical thinking in data science and machine learning education in Africa.

#### 1.3.1 Specific Objectives

1. To review existing intelligent tutoring systems, conversational agents, and LLM-based tools to identify how they support or fail to support Socratic, critical-thinking-oriented learning in African contexts.
2. To specify functional and non-functional requirements for an offline, edge-based Socratic tutor tailored to low-resource technology education environments.
3. To design a modular architecture and dialogue strategy for a Socratic tutor running on commodity hardware using compressed and LoRA-fine-tuned LLMs.
4. To implement a working prototype consisting of a Flutter-based mobile app and a FastAPI backend, supporting hybrid online/offline Socratic tutoring.
5. To evaluate the prototype in terms of technical performance (latency, resource usage) and Socratic behaviour (guiding questions, scaffolding) on representative low-spec devices.

### 1.4 Research Questions

The study was guided by the following research questions:

1. How do existing AI-driven tutoring and conversational systems support or fail to support Socratic, critical-thinking-oriented learning, particularly in African contexts?
2. What requirements and design considerations are necessary for an offline, edge-AI Socratic tutor for technology education in low-resource settings?
3. How can compressed and LoRA-fine-tuned LLMs be configured and deployed to deliver effective Socratic questioning on commodity hardware?
4. To what extent does the implemented prototype perform technically under low-resource conditions, and how well does it preserve Socratic questioning behaviour in real interactions?

### 1.5 Project Scope

The project focused on tertiary-level and advanced secondary learners studying introductory data science and machine learning topics. The scope was limited to English-language interaction and deployment on Android smartphones with at least 4 GB of RAM and commodity laptops running the backend. The prototype supported conversational Socratic tutoring, a course library, adaptive quizzes, a glossary, and an offline Python playground. The evaluation concentrated on technical feasibility and Socratic interaction quality rather than long-term learning gains across large populations.

### 1.6 Significance and Justification

The project was significant for several reasons:

- **Pedagogical**: It operationalised Socratic principles inside an AI tutor that consistently asks guiding questions and delays direct answers, supporting critical thinking, reflection, and problem-solving in technology education.
- **Technological**: It demonstrated that small, compressed, LoRA-fine-tuned LLMs quantised to GGUF can run locally on low-cost Android devices using a custom native inference layer, enabling offline AI tutoring.
- **Contextual**: It explicitly targeted low-resource African contexts by designing for bandwidth limitations, heterogeneous device capabilities, and offline-first operation.
- **Practical**: The working prototype, APK, and backend provide a concrete artefact that educators, developers, and policymakers can adapt and extend for similar offline AI learning scenarios.

### 1.7 Research Budget

The project incurred modest costs aligned with a student capstone budget:

- **Hardware**: Use of a personal laptop (8 GB RAM) and an Android smartphone (Huawei P Smart, 4 GB RAM) for development and testing.
- **Cloud Services**: Use of Google Colab (300 Compute Units, ~30 USD) for LoRA fine-tuning on L4 GPUs and Hugging Face for model hosting.
- **Internet and Utilities**: Data bundles and electricity for downloads, experiments, and deployment.
- **Miscellaneous**: Printing and binding of the final report.

Most costs were absorbed through existing personal equipment and student-level cloud credits, reinforcing the feasibility of replicating the solution in low-budget environments.

### 1.8 Research Timeline

The work followed an iterative, prototype-based schedule over approximately ten weeks:

- **Weeks 1–2**: Rapid review of existing ITS, chatbots, and Socratic LLM tutors; problem refinement and context analysis.
- **Weeks 2–3**: Requirements elicitation and documentation of functional and non-functional requirements for the offline Socratic tutor.
- **Weeks 3–5**: System analysis and design, including architecture, dialogue strategy, data model, and UML diagrams.
- **Weeks 5–7**: Implementation of the FastAPI backend, Socratic inference engine, hybrid Flutter frontend, and on-device inference via libchat.
- **Weeks 7–9**: Testing on emulator and physical devices, performance measurement, and Socratic behaviour verification.
- **Weeks 9–10**: Documentation, report writing, and preparation of artefacts for submission.

---

## CHAPTER TWO: LITERATURE REVIEW

### 2.1 Introduction

This chapter reviewed prior work on intelligent tutoring systems, conversational agents, and LLM-based educational tools with a focus on Socratic questioning, critical thinking support, and deployment in low-resource contexts. The aim was to identify methodologies and limitations that informed the design of the offline Socratic AI tutor developed in this project.

### 2.2 Historical Background of the Research Topic

The evolution of AI in education progressed from rule-based intelligent tutoring systems to adaptive web-based platforms, conversational chatbots, and, more recently, LLM-driven tutors. Early systems focused on stepwise problem-solving and immediate feedback, while later platforms incorporated richer multimedia and analytics. The emergence of large language models enabled more natural dialogue and flexible support but at the cost of heavy dependence on cloud infrastructure. In African contexts, inconsistent connectivity and limited hardware slowed adoption of these tools, reinforcing an existing digital divide in STEM education and access to reasoning-support technologies.

### 2.3 Overview of Existing Systems

Existing intelligent tutoring systems personalise learning by modelling student knowledge and adapting content and feedback. SMS-based systems such as M-Shule showed that even low-technology ITS can improve numeracy and literacy when carefully designed for local constraints. Educational chatbots deployed via messaging platforms provided 24/7 support and communities of practice for teachers and learners. LLM-based tools, including Socratic chatbots and conversational agents, delivered context-aware explanations and adaptive questioning. However, most of these systems remained cloud-hosted, answer-centric, and poorly aligned with the infrastructural realities of rural and underserved African communities.

### 2.4 Review of Related Work

The literature review examined four main strands of related work:

1. **Intelligent Tutoring Systems (ITS)**: Studies showed that adaptive ITS can improve performance but may reduce opportunities for open-ended inquiry when they are heavily solution-focused.
2. **Conversational AI Systems**: Educational chatbots supported real-time explanation and motivation but risked cognitive offloading and shallow engagement when not guided by strong pedagogy.
3. **LLM-Based Tools**: Socratic LLM tutors combining structured questioning frameworks with generative models improved reflection and critical thinking compared to generic chatbots, though they typically ran in the cloud.
4. **AI in Low-Resource Contexts**: Research on AI deployments in rural, refugee, and crisis-affected settings highlighted infrastructure gaps and emphasised offline, low-footprint AI tools as a requirement for equitable access.

### 2.3.1 Summary of Reviewed Literature

Across these strands, the literature demonstrated that AI can personalise learning and increase access to support, but existing systems often prioritised efficiency and correctness over reflection and Socratic inquiry. LLM-based tutors with explicit Socratic prompts achieved gains in critical thinking but were constrained by cloud dependence and compute requirements. Work on offline-first AI and small models suggested that quantisation and parameter-efficient fine-tuning could make AI tutors feasible on edge devices.

### 2.5 Strengths and Weaknesses of Existing Systems

**Strengths**
- Personalised feedback and adaptive pacing improved learner performance in controlled studies.
- LLM-based Socratic tutors showed measurable gains in reflection, argumentation, and critical thinking.
- Emerging offline-first platforms demonstrated that AI can be adapted to humanitarian and low-connectivity environments.

**Weaknesses**
- Many systems were answer-centric, enabling learners to bypass productive struggle and reflective thinking.
- Cloud dependence and hardware requirements made most tools inaccessible in many African schools and communities.
- LLMs remained prone to hallucinations and lacked built-in support for metacognition without strong guardrails.
- Support for African languages, local curricula, and socio-cultural contexts was limited.

### 2.6 General Comments

The literature therefore motivated the development of an AI tutor that combines Socratic pedagogy with offline capability and resource efficiency. The project’s contribution was not to create new algorithms but to integrate existing techniques—LoRA fine-tuning, quantisation, and local inference—into a coherent system aligned with African educational realities and critical thinking goals.

---

## CHAPTER THREE: SYSTEM ANALYSIS AND DESIGN

### 3.1 Introduction

This chapter presents the analysis and design of the offline Socratic AI tutoring system. It describes the research and development approach, functional and non-functional requirements, system models, architecture, and development tools used to implement the prototype.

### 3.2 Research Design and Development Model

The project followed a Design Science Research (DSR) approach combined with an iterative prototyping model. Within DSR, the Socratic AI tutor constituted the artefact created to address the identified problem of limited access to reasoning-supportive AI tools in low-resource African contexts.

The development cycle involved:

- **Problem Identification**: Limited access to AI tools that encourage critical, inquiry-based learning in low-resource environments.
- **Design and Development**: Creation of an offline, compressed Socratic AI tutoring system using a small LLM and hybrid edge–cloud routing.
- **Demonstration**: Deployment and testing of the prototype on a low-spec Android device and emulator.
- **Evaluation**: Measurement of latency, resource usage, and Socratic behaviour.

From a software engineering perspective, the implementation followed an incremental SDLC where core backend services, inference engine, and frontend screens were built in successive iterations, with continuous integration and manual testing.

#### 3.1.1 Dataset and Dataset Description

The instructional content focused on introductory data science and machine learning topics. Sources included open educational resources and instructor-authored material covering:

- Machine learning fundamentals (regression, classification, clustering, neural networks).
- Data science workflows (data cleaning, feature engineering, exploratory data analysis).
- Mathematical foundations (probability, statistics, linear algebra basics).

For Socratic behaviour, the project used synthetic and instructor-authored Socratic dialogue examples rather than student data. Approximately 307 base conversations (234 explicitly Socratic, 73 additional covering code, greetings, and general topics) were curated and augmented to around 991 samples via conversation windowing. These dialogues encoded patterns of guided questioning, scaffolding, and hinting behaviour used during LoRA fine-tuning of the Qwen3-0.6B base model.

### 3.3 Functional and Non-Functional Requirements

#### 3.3.1 Functional Requirements

The proposed AI tutoring system must satisfy the following functional requirements:

1.  **Socratic Dialogue Generation**: The system shall generate responses exclusively in the form of Socratic questions that guide learners through step-by-step reasoning rather than providing direct answers or solutions.
2.  **Offline or Low-Connectivity Operation**: The system shall operate fully offline or with minimal internet connectivity after initial installation, enabling use in resource-constrained environments.
3.  **Local Inference on Low-Resource Devices**: The system shall perform all model inference locally on low-cost devices (e.g., smartphones or laptops with limited memory and compute capacity) using quantized GGUF models.
4.  **Curriculum-Aligned Questioning**: The system shall support instructional content aligned with technology education topics (e.g., programming, data science, machine learning fundamentals).
5.  **User Prompt Handling**: The system shall accept natural language questions from learners and respond with contextually relevant Socratic prompts.
6.  **Session Continuity**: The system shall maintain conversational context within a session to ensure coherent and progressive questioning.
7.  **Adaptive Scaffolding and Hints**: The system shall dynamically adjust the difficulty of guiding questions based on the learner's responses to prevent frustration and maintain "productive struggle."
8.  **Automated Socratic Metrics**: The system shall log and calculate performance metrics, including a "Socratic Index" (measuring the proportion of questions to statements) and response latency.

#### 3.3.2 Non-Functional Requirements

Key non-functional requirements included:

- **Resource Efficiency**: Operate within 2–4 GB RAM on target devices using quantisation and parameter-efficient fine-tuning.
- **Latency**: Provide responses in less than 10 seconds under typical conditions.
- **Reliability**: Function without external cloud services once installed and configured.
- **Usability**: Offer an intuitive mobile UI with clear navigation and offline feedback on model status.
- **Scalability**: Support multiple devices via the same backend and enable model reuse.
- **Maintainability**: Use modular code organisation with clear separation between core logic, inference engine, and UI components.
- **Security**: Protect user accounts and sessions through JWT-based authentication and cautious logging.

### 3.2.1 Proposed Model Diagram

The conceptual model comprised the following main classes and subsystems:

- **SocraticTutorSystem**: Orchestrated interactions between the dialogue controller, inference engine, and user interface.
- **SocraticDialogueController**: Managed Socratic questioning strategies, including probing questions, hints, and scaffolding levels.
- **InferenceEngine**: Wrapped llama-cpp-python and constructed chat prompts for the quantised Qwen3 model, applying system prompts and difficulty hints.
- **CompressedLLM**: Represented the LoRA-fine-tuned and quantised GGUF model deployed both on the server and on-device.
- **InstructionalContent**: Stored course topics, learning objectives, quizzes, and glossary entries.
- **LearningSession**: Maintained conversation history, user progress, and performance metrics.
- **UserInterface**: Flutter screens implementing chat, courses, quizzes, glossary, Python playground, and settings.

These components interacted through clearly defined interfaces, enabling future substitution of models or UI components without redesigning the entire system.

### 3.4 System Architecture

The implemented architecture followed a hybrid edge–cloud pattern:

- **Backend Layer (FastAPI)**: Exposed REST endpoints for registration, login, and chat, using an `InferenceEngine` that loaded the quantised Socratic model via llama-cpp-python. It logged chat metrics such as Socratic index, response time, and scaffolding level in JSON files for analysis.
- **ML Layer**: Encapsulated model loading, configuration, and prompt construction. The `InferenceEngine` built prompts that combined a fixed Socratic system message, difficulty hints from `AdaptiveMetrics`, conversation history, and the latest user message.
- **Frontend Layer (Flutter)**: Implemented a mobile app with multiple screens and a `HybridTutorService` that decided whether to call the backend or use local on-device inference via a custom native library.
- **On-Device Inference Layer**: A thin C wrapper, `libchat`, compiled llama.cpp directly into the Android APK and exposed four functions for creating a chat session, generating responses, freeing strings, and destroying sessions. Dart accessed this via `dart:ffi`.

If the device was capable and the local model was downloaded, the app defaulted to local inference to guarantee offline operation and minimise network latency. In `auto` mode, it checked connectivity, prior crashes, and device capabilities to select either local or remote inference dynamically.

### 3.5 Flowchart, Use Case Diagram, Sequence Diagram

- **Flowchart**: Described the high-level flow from app launch, user authentication, model availability checks, and routing of chat requests to either local or remote inference, followed by Socratic response rendering and session logging.
- **Use Case Diagram**: Modelled the main actor (Student) and use cases: browse course library, select topic, chat with Socratic tutor, request hints, take quizzes, and save session history. The Local AI was represented as a secondary actor providing Socratic questioning.
- **Sequence Diagram**: Illustrated a typical learning session where the UI requested lesson content, the course data repository returned structured material, the student sent a message, and the Local LLM generated a Socratic response that was rendered in the chat interface. Annotations highlighted that all processing occurred offline when using on-device inference.

### 3.6 Development Tools

The main tools and technologies included:

- **Python 3.10+** for backend and ML integration.
- **FastAPI** for REST API endpoints and documentation.
- **llama-cpp-python** for efficient quantised model inference on the server.
- **PyTorch and Hugging Face Transformers** for fine-tuning the base Qwen model.
- **Flutter** for cross-platform mobile UI.
- **CMake and NDK** for compiling llama.cpp into the Android APK via `libchat`.
- **Docker and docker-compose** for containerised backend deployment.
- **VS Code and Android Studio** for development and debugging.
- **Git and GitHub** for version control and release management.

---

## CHAPTER FOUR: SYSTEM IMPLEMENTATION AND TESTING

### 4.1 Implementation and Coding

#### 4.1.1 Introduction

This chapter describes how the offline Socratic AI tutor was implemented, including the backend services, inference engine, mobile frontend, and on-device inference layer. It details the technical integration of native C++ libraries within the Flutter framework and the logic used to manage limited system resources.

#### 4.1.2 Implementation Tools and Technologies

- **Backend (Cloud Layer)**: Implemented in Python using FastAPI. The main application in [backend/main.py](backend/main.py) defined routes for registration, login, chat (public and authenticated), user progress, and health checks. JWT-based authentication secured user sessions, with rate limiting provided by SlowAPI.
- **Inference Engine**: Implemented in [backend/ml/inference_engine.py](backend/ml/inference_engine.py). The engine loaded the quantized Socratic model lazily via `model_loader`, constructed system and user prompts using `SocraticPromptBuilder`, and called the `llama-cpp-python` `create_chat_completion` method. It stripped internal `<think>...</think>` reasoning blocks and calculated Socratic metrics via `AdaptiveMetrics`.
- **Frontend (Application Layer)**: Implemented in the Flutter project under [socratic_app/](socratic_app/). Screens for chat, courses, quizzes, and glossary were organized into separate modules, with state management handled via `Provider`. Networking used `Dio`, while `SharedPreferences` stored user settings and session metadata.
- **On-Device Inference (Native Layer)**: The `libchat` C library in [socratic_app/android/app/src/main/cpp](socratic_app/android/app/src/main/cpp) wrapped `llama.cpp` and exposed a minimal API. Flutter invoked `chat_create`, `chat_generate`, and `chat_destroy` via **Dart FFI**. To prevent UI jank, all inference operations were offloaded to a **background Dart Isolate**.
- **OOM Safety Guard**: A dedicated registry-based monitor was implemented to prevent "crash loops" on low-spec devices. If the native engine failed to initialize due to Out-of-Memory (OOM) errors, the system set a persistent flag and automatically defaulted the user to Cloud Inference mode until manually reset.

### 4.2 Graphical View of the Project

#### 4.2.1 System Architecture Diagram

[INSERT DIAGRAM HERE: Hybrid AI System Architecture (The one we just made)]
*Figure 4.1: Hybrid Edge–Cloud Architecture showing local FFI bridge and remote fallback.*

#### 4.2.2 Workflow for Activating and Utilizing the Offline Socratic Mode

[INSERT CANVA GRID HERE: Image showing Settings, Starting Chat, and Socratic Interaction]
*Figure 4.2: Workflow for Activating and Utilizing the Offline Socratic Mode.*

Figure 4.2 illustrates the sequential flow for switching the application to its edge-AI state. (Left) The **Intelligence Engine** settings allow the user to toggle 'Force Offline' mode, which triggers the local GGUF model; a persistent status indicator in the top-right confirms the 'Offline' state. This interface directly controls the `HybridTutorService` logic. (Center) The **Socratic Chat Interface** maintains a clean material design, featuring a dedicated 'Hint' button and an adaptive input field. (Right) A **Socratic Interaction** in progress, where the student ('Buba', shown in orange) asks a question about neural networks and the AI tutor responds with an analogy of experts in different fields—demonstrating the system's ability to provide high-quality, inquiry-based guidance without an active internet connection.

#### 4.2.3 Supplementary Core Features

[INSERT CANVA GRID HERE: Image showing Login, Model Download, and Home Page]
*Figure 4.3: Onboarding and Core Supporting Features.*

Figure 4.3 illustrates the initial setup and secondary functional modules of the tutoring system. (Left) The **Authentication Interface** ("Bantaba AI") provides a secure entry point via JWT-based login, ensuring user sessions are synchronized across devices. (Center) The **Local Intelligence Setup** screen manages the one-time, 460MB download of the quantized Qwen3-0.6B engine, featuring a real-time progress bar to ensure the local inference layer is correctly initialized. (Right) The **Home Dashboard** serves as the central hub, providing quick access to practice quizzes, key term glossaries, and the offline Python playground, while also displaying personalized learning progress for the user ('Omar').

#### 4.2.4 Evaluation and Practical Utility Tools

[INSERT CANVA GRID HERE: Image showing Quiz, Python Playground, and Glossary]
*Figure 4.4: Practical Learning Tools: Quiz System, Python Playground, and Offline Glossary.*

Figure 4.4 showcases the assessment and utility features designed to reinforce the Socratic learning experience. (Left) The **ML Practice Quiz** interface presents curriculum-aligned questions with adaptive feedback; a notable "Ask Bantaba AI for Help" feature at the bottom allows learners to trigger a Socratic session if they are stuck on a conceptual question like regularization. (Center) The **Python Playground** enables immediate, offline execution of Python 3.12 code directly in the browser (via Pyodide), allowing students to test concepts discussed with the tutor. (Right) The **Key Terms Glossary** provides a searchable reference for technical vocabulary, categorized by topics like Deep Learning and Machine Learning, ensuring definitions for terms like 'Gradient Descent' are always available without internet access.

### 4.3 Testing

#### 4.3.1 Introduction

Testing focused on ensuring that the system behaved correctly across different devices and modes, met performance constraints, and preserved Socratic behaviour. Both functional and non-functional aspects were assessed using emulator and physical device tests.

#### 4.3.2 Objectives of Testing

The main testing objectives were to:

- Verify that all critical user flows (authentication, chat, courses, quizzes, glossary, playground) functioned correctly.
- Confirm that online and offline inference modes produced valid Socratic responses.
- Measure response latency and resource usage under realistic conditions.
- Assess Socratic compliance (responses ending with questions, presence of scaffolding hints).

#### 4.3.3 Unit Testing Outputs

Backend components such as authentication helpers, configuration loaders, and the inference engine were exercised via targeted function calls during development. While a full automated unit test suite was beyond the project’s time constraints, manual tests verified:

- Correct JWT token generation and validation.
- Robust handling of missing or malformed chat histories.
- Correct stripping of `<think>` blocks and computation of Socratic metrics.

#### 4.3.4 Validation Testing Outputs

Validation tests simulated typical user interactions:

- Registering a new account and logging in with valid and invalid credentials.
- Sending conceptual questions and confirming that responses were guiding questions rather than solutions.
- Asking for code help and verifying that the tutor responded with stepwise guidance.
- Moving between course content, quizzes, and chat while maintaining conversational context.

#### 4.3.5 Integration Testing Outputs

End-to-end integration tests connected the Flutter app to both the local FastAPI backend and the deployed cloud instance. These tests validated:

- Correct handling of network failures and automatic fallback to offline mode when appropriate.
- Synchronisation between app settings and backend endpoints.
- Consistent behaviour across emulator and physical device.

#### 4.3.6 Functional and System Testing Results

Functional testing confirmed that all core features worked as intended on both the emulator (x86_64) and a Huawei P Smart (ARM64, 4 GB RAM). System-level tests recorded the following approximate performance metrics:

- **Online (remote) inference**: 4.4–8.5 seconds per response, depending on network latency and server load.
- **Offline (local) inference**: 5–7 seconds per response on ARM64 devices, independent of network conditions.

The tutor consistently produced Socratic responses that ended with questions and integrated scaffolded hints when learners appeared stuck.

#### 4.3.7 Acceptance Testing Report

Acceptance criteria required that the system:

- Support offline Socratic chat on at least one low-spec Android device.
- Maintain average response times below 10 seconds.
- Provide complete flows for authentication, course browsing, quizzes, glossary, and playground.

Testing confirmed that these criteria were met. The prototype was therefore accepted as a valid demonstration of offline Socratic tutoring in a low-resource context.

---

## CHAPTER FIVE: RESULTS AND DISCUSSION

### 5.1 Overview of Results

The implemented system successfully delivered a hybrid offline-first Socratic tutoring experience in data science and machine learning. The key results related to technical performance, Socratic behaviour, and user experience.

### 5.2 Technical Performance Results

On the tested devices, the system achieved the following approximate metrics:

- **Latency**: 4.4–8.5 seconds in online mode and 5–7 seconds in offline mode.
- **Stability**: No critical crashes occurred on the tested ARM64 device during extended sessions; crash-loop protection logic ensured that devices incapable of local inference defaulted to cloud mode.
- **Resource Usage**: The quantised GGUF model (~460 MB) fit within the 4 GB RAM budget of the Huawei P Smart, confirming feasibility on lower-end smartphones.

These results showed that compressed, quantised LLMs can run acceptably on commodity devices while respecting the response-time constraints of interactive tutoring.

### 5.3 Socratic Behaviour and Interaction Quality

The Socratic guardrails encoded in the system prompt and fine-tuning data were reflected in observed interactions:

- Responses consistently ended with guiding questions rather than answers.
- The tutor broke complex problems into smaller sub-questions, providing hints when learners showed confusion.
- For code-related prompts, the tutor encouraged learners to write or revise code themselves instead of supplying full solutions.

These behaviours aligned with the project’s pedagogical objective of fostering reflection and critical thinking rather than answer retrieval.

### 5.4 User Experience Observations

From a product perspective, several UX aspects contributed to the system’s effectiveness:

- The dedicated model download screen and progress feedback reduced user frustration during initial setup.
- Clear indicators of online/offline mode helped users understand how the tutor was operating.
- The integration of chat with courses, quizzes, glossary, and playground allowed learners to move fluidly between content consumption, practice, and Socratic dialogue.
- The modern UI theme and dark mode support improved readability and comfort during extended sessions.

### 5.5 Discussion

The findings indicated that an offline-capable Socratic tutor built on a small, quantised LLM can meet pragmatic constraints in low-resource African contexts while preserving key elements of Socratic pedagogy. The hybrid edge–cloud design allowed the same app to run in both offline and connected settings, reducing deployment friction.

However, the project also surfaced challenges. Tuning the balance between guidance and frustration required careful prompt engineering and future work on adaptive scaffolding thresholds. Moreover, while latency was acceptable for a capstone prototype, further optimisation (e.g. more aggressive quantisation or prompt compression) would improve responsiveness on lower-end devices.

---

## CHAPTER SIX: CONCLUSIONS AND RECOMMENDATIONS

### 6.1 Conclusions

The project set out to design, implement, and evaluate an offline-capable Socratic AI tutor for data science and machine learning education in African contexts. It concluded that:

1. A small, LoRA-fine-tuned and quantised LLM (Qwen3-0.6B) can support Socratic questioning on commodity Android devices with 4 GB RAM.
2. A hybrid edge–cloud architecture enables flexible deployment, allowing devices to use local inference when capable and cloud inference otherwise.
3. Socratic guardrails implemented via system prompts, specialised training data, and runtime checks effectively constrained the model to generate guiding questions rather than direct answers.
4. The prototype met its performance targets, achieving sub-10-second response times and stable operation during testing on both emulator and physical hardware.

Overall, the project demonstrated the feasibility of delivering inquiry-driven AI tutoring in low-resource settings, contributing to efforts to narrow the digital reasoning divide.

### 6.2 Limitations of the Study

- The evaluation used a limited number of devices and informal user interactions rather than a large-scale controlled study.
- The content focused on data science and machine learning; generalisation to other subjects and languages was not tested.
- Ethical considerations such as bias analysis, long-term learner dependence, and data protection were recognised but not fully explored.
- The system relied on an initial online model download, which may still pose barriers in the lowest-connectivity settings.

### 6.3 Recommendations

#### 6.3.1 Product and Technical Recommendations

To improve and extend the product, the following steps are recommended:

1. **Content Expansion**: Add more courses, problem sets, and Socratic dialogues across additional technology topics and, where possible, in African languages.
2. **Automated Evaluation**: Implement lightweight in-app surveys and logging to systematically capture learner feedback and usage patterns.
3. **Further Optimisation**: Experiment with smaller models, more aggressive quantisation, or distillation to reduce latency and memory footprint on very low-spec devices.
4. **Robust Testing Suite**: Develop automated unit and integration tests for both backend and frontend to improve reliability.
5. **Offline Installer Bundles**: Provide pre-bundled APK + model packages on SD cards or local servers for contexts with extremely limited connectivity.

#### 6.3.2 Future Research

For future academic work, it is recommended to:

- Conduct controlled studies comparing the Socratic tutor with non-Socratic AI tools and human tutoring, measuring impacts on critical thinking and learning outcomes.
- Explore hybrid human–AI tutoring models where educators use the system as a co-pilot rather than a replacement.
- Investigate multi-lingual and culturally adapted Socratic dialogue strategies for diverse African learning environments.

---

## References

*(References follow the list and citation style used in the research document, ensuring that all in-text citations are properly attributed and formatted according to the required academic style.)*
