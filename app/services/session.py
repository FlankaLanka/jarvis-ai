"""
Jarvis Voice Assistant - Session Management Service

In-memory session storage for MVP. Can be upgraded to Redis/DynamoDB later.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional

from loguru import logger

from app.config import settings


class SessionManager:
    """Manages user sessions with in-memory storage."""
    
    def __init__(self):
        self._sessions: Dict[str, dict] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new session.
        
        Args:
            user_id: Optional user ID. If not provided, uses default_user_id from settings.
        
        Returns:
            The new session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Use default user if not provided (for now, single user mode)
        if user_id is None:
            user_id = settings.default_user_id
        
        self._sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "created_at": now,
            "expires_at": now + timedelta(minutes=settings.session_expiry_minutes),
            "last_activity": now,
        }
        
        logger.debug(f"Created session: {session_id} for user: {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """
        Get a session by ID.
        
        Args:
            session_id: The session ID to retrieve
            
        Returns:
            Session data or None if not found/expired
        """
        session = self._sessions.get(session_id)
        
        if session and session["expires_at"] > datetime.utcnow():
            return session
        elif session:
            # Session expired, clean it up
            self.delete_session(session_id)
        
        return None
    
    def get_or_create_session(self, session_id: Optional[str], user_id: Optional[str] = None) -> dict:
        """
        Get an existing session or create a new one.
        
        Args:
            session_id: Optional session ID to retrieve
            user_id: Optional user ID for new sessions
            
        Returns:
            Session data
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                self._update_activity(session_id)
                return session
        
        new_session_id = self.create_session(user_id=user_id)
        return self._sessions[new_session_id]
    
    def extend_session(self, session_id: str) -> bool:
        """
        Extend a session's expiration time.
        
        Args:
            session_id: The session ID to extend
            
        Returns:
            True if successful, False if session not found
        """
        session = self.get_session(session_id)
        
        if session:
            session["expires_at"] = datetime.utcnow() + timedelta(
                minutes=settings.session_expiry_minutes
            )
            self._update_activity(session_id)
            logger.debug(f"Extended session: {session_id}")
            return True
        
        return False
    
    def delete_session(self, session_id: str):
        """
        Delete a session.
        
        Args:
            session_id: The session ID to delete
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Deleted session: {session_id}")
    
    def _update_activity(self, session_id: str):
        """Update the last activity timestamp for a session."""
        if session_id in self._sessions:
            self._sessions[session_id]["last_activity"] = datetime.utcnow()
    
    def cleanup_expired(self):
        """Remove all expired sessions."""
        now = datetime.utcnow()
        expired = [
            sid for sid, session in self._sessions.items()
            if session["expires_at"] <= now
        ]
        
        for session_id in expired:
            self.delete_session(session_id)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def start_cleanup_task(self):
        """Start the background cleanup task."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(60)  # Run every minute
                self.cleanup_expired()
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Session cleanup task started")
    
    def stop_cleanup_task(self):
        """Stop the background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
            logger.info("Session cleanup task stopped")
    
    @property
    def active_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self._sessions)


# Global session manager instance
session_manager = SessionManager()

