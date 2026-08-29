"""
Unit tests for core/auth.py

Covers:
  - PBKDF2 password hashing and verification
  - Legacy SHA-256 hash verification (backward compat)
  - JWT creation, verification, expiry, and tampering
  - get_current_user lookup
  - register_user (success + duplicate checks)
  - authenticate_user (email, username, wrong password, unknown user)
"""

import hashlib
import time
import uuid
from datetime import timedelta
from unittest.mock import patch, MagicMock

import pytest
from jose import jwt as jose_jwt

# Patch save_users so tests never touch the filesystem
import unittest.mock as mock


@pytest.fixture(autouse=True)
def isolate_users_db():
    """Give each test an empty, isolated users_db and no-op save_users."""
    import core.auth as auth
    original = dict(auth.users_db)
    auth.users_db.clear()
    with patch.object(auth, "save_users"):
        yield auth
    auth.users_db.clear()
    auth.users_db.update(original)


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

class TestPasswordHashing:
    def test_hash_produces_pbkdf2_format(self, isolate_users_db):
        auth = isolate_users_db
        h = auth.get_password_hash("secret")
        assert h.startswith("pbkdf2:sha256:")

    def test_verify_correct_password(self, isolate_users_db):
        auth = isolate_users_db
        h = auth.get_password_hash("mypassword")
        assert auth.verify_password("mypassword", h) is True

    def test_verify_wrong_password(self, isolate_users_db):
        auth = isolate_users_db
        h = auth.get_password_hash("mypassword")
        assert auth.verify_password("wrongpassword", h) is False

    def test_two_hashes_of_same_password_differ(self, isolate_users_db):
        """Each call uses a new random salt."""
        auth = isolate_users_db
        h1 = auth.get_password_hash("same")
        h2 = auth.get_password_hash("same")
        assert h1 != h2

    def test_legacy_sha256_verify(self, isolate_users_db):
        """Bare hex SHA-256 hashes from the old format still verify correctly."""
        auth = isolate_users_db
        plain = "oldpassword"
        legacy_hash = hashlib.sha256(plain.encode()).hexdigest()
        assert auth.verify_password(plain, legacy_hash) is True

    def test_legacy_sha256_wrong_password(self, isolate_users_db):
        auth = isolate_users_db
        legacy_hash = hashlib.sha256("correct".encode()).hexdigest()
        assert auth.verify_password("wrong", legacy_hash) is False


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

class TestJWT:
    def test_create_token_is_string(self, isolate_users_db):
        auth = isolate_users_db
        token = auth.create_access_token({"sub": "user-123"})
        assert isinstance(token, str)
        assert token.count(".") == 2  # header.payload.signature

    def test_verify_valid_token(self, isolate_users_db):
        auth = isolate_users_db
        token = auth.create_access_token({"sub": "user-abc"})
        payload = auth.verify_token(token)
        assert payload is not None
        assert payload["sub"] == "user-abc"

    def test_verify_expired_token_returns_none(self, isolate_users_db):
        auth = isolate_users_db
        token = auth.create_access_token(
            {"sub": "user-xyz"}, expires_delta=timedelta(seconds=-1)
        )
        assert auth.verify_token(token) is None

    def test_verify_tampered_payload_returns_none(self, isolate_users_db):
        auth = isolate_users_db
        token = auth.create_access_token({"sub": "user-123"})
        # Corrupt the payload section (middle part)
        parts = token.split(".")
        parts[1] = parts[1][:-3] + "AAA"
        bad_token = ".".join(parts)
        assert auth.verify_token(bad_token) is None

    def test_verify_wrong_key_returns_none(self, isolate_users_db):
        auth = isolate_users_db
        # Sign with a different key
        fake_token = jose_jwt.encode(
            {"sub": "hacker", "exp": 9999999999},
            "completely-different-secret",
            algorithm="HS256",
        )
        assert auth.verify_token(fake_token) is None

    def test_verify_garbage_string_returns_none(self, isolate_users_db):
        auth = isolate_users_db
        assert auth.verify_token("this.is.garbage") is None

    def test_token_contains_exp_claim(self, isolate_users_db):
        auth = isolate_users_db
        token = auth.create_access_token({"sub": "u1"})
        payload = auth.verify_token(token)
        assert "exp" in payload

    def test_custom_expiry_honoured(self, isolate_users_db):
        auth = isolate_users_db
        token = auth.create_access_token(
            {"sub": "u1"}, expires_delta=timedelta(hours=2)
        )
        payload = auth.verify_token(token)
        # exp should be ~2h from now — at least 1h away
        assert payload["exp"] > time.time() + 3600


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------

class TestGetCurrentUser:
    def _seed_user(self, auth):
        uid = str(uuid.uuid4())
        auth.users_db["test@example.com"] = {
            "id": uid,
            "username": "testuser",
            "email": "test@example.com",
            "password_hash": auth.get_password_hash("pw"),
            "created_at": "2024-01-01T00:00:00",
        }
        return uid

    def test_returns_user_for_valid_token(self, isolate_users_db):
        auth = isolate_users_db
        uid = self._seed_user(auth)
        token = auth.create_access_token({"sub": uid})
        user = auth.get_current_user(token)
        assert user is not None
        assert user["id"] == uid

    def test_returns_none_for_expired_token(self, isolate_users_db):
        auth = isolate_users_db
        uid = self._seed_user(auth)
        token = auth.create_access_token(
            {"sub": uid}, expires_delta=timedelta(seconds=-1)
        )
        assert auth.get_current_user(token) is None

    def test_returns_none_when_user_not_in_db(self, isolate_users_db):
        auth = isolate_users_db
        token = auth.create_access_token({"sub": "nonexistent-id"})
        assert auth.get_current_user(token) is None

    def test_returns_none_for_token_without_sub(self, isolate_users_db):
        auth = isolate_users_db
        token = auth.create_access_token({"data": "no-sub-field"})
        assert auth.get_current_user(token) is None


# ---------------------------------------------------------------------------
# register_user
# ---------------------------------------------------------------------------

class TestRegisterUser:
    def test_register_creates_user(self, isolate_users_db):
        auth = isolate_users_db
        user = auth.register_user("alice", "alice@example.com", "pass123")
        assert user["username"] == "alice"
        assert user["email"] == "alice@example.com"
        assert "id" in user
        assert "alice@example.com" in auth.users_db

    def test_register_hashes_password(self, isolate_users_db):
        auth = isolate_users_db
        auth.register_user("alice", "alice@example.com", "pass123")
        stored = auth.users_db["alice@example.com"]["password_hash"]
        assert stored.startswith("pbkdf2:")
        assert stored != "pass123"

    def test_register_duplicate_email_raises(self, isolate_users_db):
        auth = isolate_users_db
        auth.register_user("alice", "alice@example.com", "pw")
        with pytest.raises(ValueError, match="email already exists"):
            auth.register_user("alice2", "alice@example.com", "pw2")

    def test_register_duplicate_username_raises(self, isolate_users_db):
        auth = isolate_users_db
        auth.register_user("alice", "alice@example.com", "pw")
        with pytest.raises(ValueError, match="username is already taken"):
            auth.register_user("Alice", "other@example.com", "pw2")

    def test_register_email_stored_lowercase(self, isolate_users_db):
        auth = isolate_users_db
        auth.register_user("bob", "BOB@EXAMPLE.COM", "pw")
        assert "bob@example.com" in auth.users_db


# ---------------------------------------------------------------------------
# authenticate_user
# ---------------------------------------------------------------------------

class TestAuthenticateUser:
    def _register(self, auth):
        return auth.register_user("charlie", "charlie@example.com", "correct")

    def test_authenticate_by_email(self, isolate_users_db):
        auth = isolate_users_db
        self._register(auth)
        user = auth.authenticate_user("charlie@example.com", "correct")
        assert user is not None
        assert user["username"] == "charlie"

    def test_authenticate_by_username(self, isolate_users_db):
        auth = isolate_users_db
        self._register(auth)
        user = auth.authenticate_user("charlie", "correct")
        assert user is not None
        assert user["email"] == "charlie@example.com"

    def test_authenticate_wrong_password_returns_none(self, isolate_users_db):
        auth = isolate_users_db
        self._register(auth)
        assert auth.authenticate_user("charlie@example.com", "wrong") is None

    def test_authenticate_unknown_user_returns_none(self, isolate_users_db):
        auth = isolate_users_db
        assert auth.authenticate_user("nobody@example.com", "pw") is None

    def test_authenticate_case_insensitive_identifier(self, isolate_users_db):
        auth = isolate_users_db
        self._register(auth)
        user = auth.authenticate_user("CHARLIE@EXAMPLE.COM", "correct")
        assert user is not None

    def test_authenticate_upgrades_legacy_hash(self, isolate_users_db):
        """A legacy SHA-256 hash should be silently upgraded to PBKDF2 on login."""
        auth = isolate_users_db
        plain = "legacypass"
        legacy_hash = hashlib.sha256(plain.encode()).hexdigest()
        auth.users_db["legacy@example.com"] = {
            "id": str(uuid.uuid4()),
            "username": "legacyuser",
            "email": "legacy@example.com",
            "password_hash": legacy_hash,
            "created_at": "2024-01-01T00:00:00",
        }
        user = auth.authenticate_user("legacy@example.com", plain)
        assert user is not None
        assert user["password_hash"].startswith("pbkdf2:")
