from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

# Configure logging
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "registration.log")
    ]
)
logger = logging.getLogger("registration")

from ml.inference_engine import inference_engine
from ml.model_loader import model_loader
from evaluation.metrics import AdaptiveMetrics
from core.auth import UserRegister, UserLogin, UserResponse, users_db, get_password_hash, save_users

@app.middleware("http")
async def log_requests(request, call_next):
    import time
    start_time = time.time()
    path = request.url.path
    method = request.method
    logger.info(f">>> Request: {method} {path}")
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"<<< Response: {method} {path} - Status: {response.status_code} - Duration: {duration:.4f}s")
    return response

# Load config
def load_config():
    config_path = Path(__file__).parent / 'data' / 'config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return {"server": {"host": "0.0.0.0", "port": 8000, "cors_origins": ["*"]}}

config = load_config()


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    print("🚀 Starting Socratic AI Tutor Backend...")
    model_loaded = model_loader.load_model()
    if model_loaded:
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Model not loaded - using fallback responses")
    yield
    # Shutdown: Clean up
    print("👋 Shutting down...")
    model_loader.unload_model()


app = FastAPI(
    title="Socratic AI Tutor",
    description="An AI-powered Socratic tutoring system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('server', {}).get('cors_origins', ['*']),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    difficulty: Optional[str] = "intermediate"
    topic: Optional[str] = None
    max_tokens: Optional[int] = 150
    history: Optional[List[Dict[str, Any]]] = None


class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None


class SessionStartRequest(BaseModel):
    topic: Optional[str] = None
    difficulty: Optional[str] = "intermediate"


class HintRequest(BaseModel):
    session_id: Optional[str] = None
    context: Optional[str] = None


class HintResponse(BaseModel):
    hint: str


# Store for session data (in production, use Redis or database)
sessions: Dict[str, Dict] = {}


# Routes
@app.get("/")
async def root():
    return {
        "message": "Socratic AI Tutor Backend is running",
        "model_loaded": model_loader.is_loaded,
        "version": "1.0.0"
    }


# Auth Routes
@app.post("/register", response_model=UserResponse)
async def register(user: UserRegister):
    email_lower = user.email.lower()
    username_lower = user.username.lower()
    logger.info(f"--- Registration Attempt ---")
    logger.info(f"Username: {user.username} (normalized: {username_lower})")
    logger.info(f"Email: {user.email} (normalized: {email_lower})")
    logger.info(f"Current users in DB keys: {list(users_db.keys())}")
    
    if email_lower in users_db:
        logger.warning(f"Conflict: Email {email_lower} already exists")
        existing_user = users_db[email_lower]
        password_matches = existing_user["password_hash"] == get_password_hash(user.password)
        logger.info(f"Password matches existing record: {password_matches}")
        
        if password_matches:
            logger.info(f"Auto-returning existing user info for matching credentials")
            return UserResponse(**existing_user)
            
        logger.error(f"Registration REJECTED: Email {email_lower} taken and password mismatch")
        raise HTTPException(status_code=400, detail="Email already registered with a different password")
    
    # Check if username is already taken
    for existing_email, existing_data in users_db.items():
        if existing_data["username"].lower() == username_lower:
            logger.warning(f"Conflict: Username {username_lower} already taken by {existing_email}")
            raise HTTPException(status_code=400, detail="Username already taken")

    import uuid
    user_id = str(uuid.uuid4())
    user_data = {
        "id": user_id,
        "username": user.username,
        "email": email_lower,
        "password_hash": get_password_hash(user.password),
        "created_at": datetime.now()
    }
    users_db[email_lower] = user_data
    save_users()
    logger.info(f"SUCCESS: Created user {user.username} with ID {user_id}")
    return UserResponse(**user_data)


@app.post("/login")
async def login(user: UserLogin):
    identifier_lower = user.email.lower()
    logger.info(f"--- Login Attempt ---")
    logger.info(f"Identifier provided: {identifier_lower}")
    logger.info(f"Targeting logic: searching in {len(users_db)} keys: {list(users_db.keys())}")
    
    target_user = None
    # Search by email (direct key)
    if identifier_lower in users_db:
        target_user = users_db[identifier_lower]
        logger.info("User found by email key match")
    else:
        # Search by username
        for u in users_db.values():
            if u["username"].lower() == identifier_lower:
                target_user = u
                logger.info(f"User found by username match: {u['username']}")
                break
            
    if not target_user:
        logger.warning(f"Login FAILED: User '{identifier_lower}' not found")
        raise HTTPException(status_code=400, detail="User account not found")
        
    if target_user["password_hash"] != get_password_hash(user.password):
        logger.warning(f"Login FAILED: Password mismatch for {identifier_lower}")
        raise HTTPException(status_code=400, detail="Incorrect password")
    
    logger.info(f"SUCCESS: {target_user['username']} logged in")
    return {
        "id": target_user["id"],
        "username": target_user["username"],
        "token": f"mock-token-{target_user['id']}"
    }


@app.get("/download/model")
async def download_model():
    """Serve the GGUF model file for mobile download."""
    model_path = Path(__file__).parent / config.get('model', {}).get('path', 'models/socratic-q4_k_m.gguf')
    if not model_path.exists():
        # For prototype purposes, let's create a placeholder if it doesn't exist
        # to prevent 404s during development of the download UI
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            with open(model_path, 'wb') as f:
                f.write(b"Placeholder for GGUF model content - Qwen3-0.6B")
    
    return FileResponse(
        path=model_path,
        filename="socratic-q4_k_m.gguf",
        media_type="application/octet-stream"
    )


@app.get("/content/manifest")
async def get_content_manifest():
    """Serve the central content manifest for course discovery."""
    manifest_path = Path(__file__).parent / 'data' / 'content_manifest.json'
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")
    
    with open(manifest_path, 'r') as f:
        return json.load(f)


@app.get("/content/{course_id}/course.json")
async def get_course_data(course_id: str):
    """Serve full course JSON data."""
    # Try backend specific content first
    backend_path = Path(__file__).parent / 'data' / 'courses' / course_id / 'course.json'
    if backend_path.exists():
        with open(backend_path, 'r') as f:
            return json.load(f)

    # Fallback to frontend assets
    frontend_path = Path(__file__).parent.parent / 'frontend' / 'assets' / 'courses' / course_id / 'course.json'
    if frontend_path.exists():
        with open(frontend_path, 'r') as f:
            return json.load(f)
    
    raise HTTPException(status_code=404, detail=f"Course {course_id} not found")


@app.get("/content/{course_id}/lessons/{lesson_file}")
async def get_lesson_content(course_id: str, lesson_file: str):
    """Serve lesson markdown content."""
    # Check backend first
    backend_path = Path(__file__).parent / 'data' / 'courses' / course_id / 'lessons' / lesson_file
    if backend_path.exists():
        return FileResponse(backend_path)

    # Fallback to frontend assets
    frontend_path = Path(__file__).parent.parent / 'frontend' / 'assets' / 'courses' / course_id / 'lessons' / lesson_file
    if frontend_path.exists():
        return FileResponse(frontend_path)
    
    raise HTTPException(status_code=404, detail="Lesson not found")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and receive a Socratic response."""
    try:
        # If history is provided, we use it (stateless)
        # Otherwise use the engine's internal history
        formatted_history = []
        if request.history:
            # Prepare history in the format expected by prompt builder
            # Converting from Flutter Message JSON to simple role/content pairs
            for msg in request.history:
                formatted_history.append({
                    "role": "user" if msg.get("isUser") else "assistant",
                    "content": msg.get("text", "")
                })
            inference_engine.conversation_history = formatted_history

        # ADAPTIVE DIFFICULTY: If history exists, recommend level
        difficulty = request.difficulty or "intermediate"
        if formatted_history and len(formatted_history) >= 2:
            adaptive_level = AdaptiveMetrics.recommend_scaffolding_level(formatted_history)
            if difficulty == "intermediate": # Only auto-adjust if at default
                difficulty = adaptive_level

        response = inference_engine.generate_response(
            user_message=request.message,
            difficulty=difficulty,
            topic=request.topic
        )
        
        return ChatResponse(
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/start")
async def start_session(request: SessionStartRequest):
    """Start a new learning session."""
    import uuid
    session_id = str(uuid.uuid4())
    
    sessions[session_id] = {
        "topic": request.topic,
        "difficulty": request.difficulty,
        "started_at": str(__import__('datetime').datetime.now()),
        "messages": []
    }
    
    # Clear previous conversation history for fresh start
    inference_engine.clear_history()
    
    return {
        "session_id": session_id,
        "topic": request.topic,
        "difficulty": request.difficulty,
        "message": "Session started successfully"
    }


@app.post("/session/end")
async def end_session(session_id: str):
    """End a learning session and get summary."""
    if session_id in sessions:
        session = sessions.pop(session_id)
        history = inference_engine.get_history()
        
        return {
            "session_id": session_id,
            "topic": session.get("topic"),
            "message_count": len(history),
            "summary": "Session ended successfully"
        }
    
    return {"message": "Session not found or already ended"}


@app.post("/hint", response_model=HintResponse)
async def get_hint(request: HintRequest):
    """Get a hint for the current discussion."""
    hint = inference_engine.generate_hint(context=request.context)
    return HintResponse(hint=hint)


@app.get("/sessions")
async def get_sessions():
    """Get list of active sessions."""
    return {"sessions": list(sessions.keys())}


@app.get("/metrics")
async def get_metrics():
    """Get usage metrics (placeholder for analytics)."""
    return {
        "total_sessions": len(sessions),
        "model_loaded": model_loader.is_loaded,
        "model_path": model_loader.model_path
    }


@app.post("/difficulty")
async def set_difficulty(difficulty: str):
    """Update the difficulty level."""
    valid_levels = ["beginner", "intermediate", "advanced"]
    if difficulty not in valid_levels:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid difficulty. Must be one of: {valid_levels}"
        )
    return {"difficulty": difficulty, "message": "Difficulty updated"}


if __name__ == "__main__":
    server_config = config.get('server', {})
    uvicorn.run(
        app,
        host=server_config.get('host', '0.0.0.0'),
        port=server_config.get('port', 8000)
    )
