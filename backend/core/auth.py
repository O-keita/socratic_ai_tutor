from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
import hashlib
import json
from pathlib import Path

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: str # email or username
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: EmailStr
    created_at: datetime

# Mock Database with Persistence
DB_FILE = Path(__file__).parent.parent / 'data' / 'users.json'
users_db = {}

def load_users():
    if DB_FILE.exists():
        try:
            with open(DB_FILE, 'r') as f:
                data = json.load(f)
                # Use .update() to preserve the reference to the dictionary
                users_db.clear()
                users_db.update(data)
                print(f"Loaded {len(users_db)} users from {DB_FILE}")
        except Exception as e:
            print(f"Error loading users: {e}")

def save_users():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Convert objects to serializable format
        serializable_db = {}
        for k, v in users_db.items():
            user_copy = v.copy()
            if isinstance(user_copy.get('created_at'), datetime):
                user_copy['created_at'] = user_copy['created_at'].isoformat()
            serializable_db[k] = user_copy
            
        with open(DB_FILE, 'w') as f:
            json.dump(serializable_db, f, indent=4)
    except Exception as e:
        print(f"Error saving users: {e}")

# Initial load
load_users()

def get_password_hash(password: str):
    return hashlib.sha256(password.encode()).hexdigest()
