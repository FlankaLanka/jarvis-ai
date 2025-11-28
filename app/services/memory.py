"""
Jarvis Voice Assistant - Conversation Memory Service

In-memory conversation history storage for MVP. Can be upgraded to Redis/DynamoDB later.
"""

from typing import Dict, List, Optional
from datetime import datetime

from loguru import logger


class MemoryService:
    """Manages conversation history with in-memory storage."""
    
    def __init__(self, max_history_length: int = 20):
        """
        Initialize the memory service.
        
        Args:
            max_history_length: Maximum number of exchanges to keep per session
        """
        self._histories: Dict[str, List[dict]] = {}
        self._max_history = max_history_length
    
    def add_exchange(self, session_id: str, user_message: str, assistant_message: str):
        """
        Add a conversation exchange to history.
        
        Args:
            session_id: The session ID
            user_message: The user's message
            assistant_message: The assistant's response
        """
        if session_id not in self._histories:
            self._histories[session_id] = []
        
        exchange = {
            "timestamp": datetime.utcnow().isoformat(),
            "user": user_message,
            "assistant": assistant_message,
        }
        
        self._histories[session_id].append(exchange)
        
        # Trim to max length
        if len(self._histories[session_id]) > self._max_history:
            self._histories[session_id] = self._histories[session_id][-self._max_history:]
        
        logger.debug(f"Added exchange to session {session_id}, history length: {len(self._histories[session_id])}")
    
    def get_history(self, session_id: str) -> List[dict]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            List of conversation exchanges
        """
        return self._histories.get(session_id, [])
    
    def get_formatted_history(self, session_id: str) -> List[dict]:
        """
        Get history formatted for LLM context.
        
        Args:
            session_id: The session ID
            
        Returns:
            List of messages in LLM format
        """
        history = self.get_history(session_id)
        messages = []
        
        for exchange in history:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        return messages
    
    def clear_history(self, session_id: str):
        """
        Clear conversation history for a session.
        
        Args:
            session_id: The session ID
        """
        if session_id in self._histories:
            del self._histories[session_id]
            logger.debug(f"Cleared history for session {session_id}")
    
    def get_last_exchange(self, session_id: str) -> Optional[dict]:
        """
        Get the most recent exchange for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            The last exchange or None
        """
        history = self.get_history(session_id)
        return history[-1] if history else None
    
    def get_summary(self, session_id: str, max_exchanges: int = 5) -> str:
        """
        Get a summary of recent conversation for context.
        
        Args:
            session_id: The session ID
            max_exchanges: Maximum exchanges to include
            
        Returns:
            Summary string
        """
        history = self.get_history(session_id)
        recent = history[-max_exchanges:] if len(history) > max_exchanges else history
        
        if not recent:
            return "No previous conversation."
        
        summary_parts = []
        for exchange in recent:
            summary_parts.append(f"User: {exchange['user'][:100]}...")
            summary_parts.append(f"Assistant: {exchange['assistant'][:100]}...")
        
        return "\n".join(summary_parts)


# Global memory service instance
memory_service = MemoryService()

