"""
Jarvis Voice Assistant - Session Management API Endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.session import session_manager
from app.services.memory import memory_service


router = APIRouter()


class SessionCreateResponse(BaseModel):
    """Response model for session creation."""
    session_id: str
    expires_at: str


class SessionInfoResponse(BaseModel):
    """Response model for session info."""
    session_id: str
    created_at: str
    expires_at: str
    message_count: int


@router.post("/create", response_model=SessionCreateResponse)
async def create_session():
    """
    Create a new session.
    
    Returns:
        New session ID and expiration time
    """
    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    
    return SessionCreateResponse(
        session_id=session_id,
        expires_at=session["expires_at"].isoformat()
    )


@router.get("/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str):
    """
    Get information about a session.
    
    Args:
        session_id: The session ID to query
    
    Returns:
        Session information including message count
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = memory_service.get_history(session_id)
    
    return SessionInfoResponse(
        session_id=session_id,
        created_at=session["created_at"].isoformat(),
        expires_at=session["expires_at"].isoformat(),
        message_count=len(history)
    )


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and its associated data.
    
    Args:
        session_id: The session ID to delete
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_manager.delete_session(session_id)
    memory_service.clear_history(session_id)
    
    return {"status": "deleted", "session_id": session_id}


@router.post("/{session_id}/extend")
async def extend_session(session_id: str):
    """
    Extend a session's expiration time.
    
    Args:
        session_id: The session ID to extend
    """
    success = session_manager.extend_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = session_manager.get_session(session_id)
    
    return {
        "status": "extended",
        "session_id": session_id,
        "expires_at": session["expires_at"].isoformat()
    }

