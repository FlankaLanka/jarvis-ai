"""
Jarvis Voice Assistant - Interrupt Handler Service

Manages interruption of ongoing voice processing when users speak over responses.
"""

from typing import Dict, Set
from datetime import datetime

from loguru import logger


class InterruptHandler:
    """Handles interruption of voice processing tasks."""
    
    def __init__(self):
        self._interrupted_sessions: Set[str] = set()
        self._interrupt_timestamps: Dict[str, datetime] = {}
    
    def interrupt(self, session_id: str):
        """
        Mark a session as interrupted.
        
        Args:
            session_id: The session ID to interrupt
        """
        self._interrupted_sessions.add(session_id)
        self._interrupt_timestamps[session_id] = datetime.utcnow()
        logger.info(f"Session {session_id} interrupted")
    
    def is_interrupted(self, session_id: str) -> bool:
        """
        Check if a session is currently interrupted.
        
        Args:
            session_id: The session ID to check
            
        Returns:
            True if interrupted, False otherwise
        """
        return session_id in self._interrupted_sessions
    
    def clear_interrupt(self, session_id: str):
        """
        Clear the interrupt flag for a session.
        
        Args:
            session_id: The session ID to clear
        """
        self._interrupted_sessions.discard(session_id)
        self._interrupt_timestamps.pop(session_id, None)
        logger.debug(f"Cleared interrupt for session {session_id}")
    
    def cancel_session(self, session_id: str):
        """
        Cancel all processing for a session (used on disconnect).
        
        Args:
            session_id: The session ID to cancel
        """
        self.interrupt(session_id)
        logger.info(f"Cancelled all processing for session {session_id}")
    
    def get_interrupt_time(self, session_id: str) -> datetime:
        """
        Get the timestamp of the last interrupt for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            Interrupt timestamp or None
        """
        return self._interrupt_timestamps.get(session_id)


# Global interrupt handler instance
interrupt_handler = InterruptHandler()

