"""
Jarvis Voice Assistant - Conversation Memory Service

In-memory conversation history storage with vector database integration for 
persistent memory and semantic search capabilities.
"""

from typing import Dict, List, Optional
from datetime import datetime

from loguru import logger

from app.config import settings


class MemoryService:
    """
    Manages conversation history with in-memory storage for active sessions
    and vector database integration for persistent semantic search.
    """
    
    def __init__(self, max_history_length: int = 20):
        """
        Initialize the memory service.
        
        Args:
            max_history_length: Maximum number of exchanges to keep per session
        """
        self._histories: Dict[str, List[dict]] = {}
        self._max_history = max_history_length
        self._vectordb = None  # Lazy import
        self._indexing = None  # Lazy import
    
    @property
    def vectordb(self):
        """Lazy import of vectordb service to avoid circular dependencies."""
        if self._vectordb is None:
            from app.services.vectordb import vectordb_service
            self._vectordb = vectordb_service
        return self._vectordb
    
    @property
    def indexing(self):
        """Lazy import of indexing service to avoid circular dependencies."""
        if self._indexing is None:
            from app.services.indexing import indexing_service
            self._indexing = indexing_service
        return self._indexing
    
    def add_exchange(self, session_id: str, user_message: str, assistant_message: str, tool_calls: list = None, tool_results: list = None):
        """
        Add a conversation exchange to history.
        
        Args:
            session_id: The session ID
            user_message: The user's message
            assistant_message: The assistant's response
            tool_calls: Optional list of tool calls made (name, arguments)
            tool_results: Optional list of tool results (name, success, message)
        """
        if session_id not in self._histories:
            self._histories[session_id] = []
        
        exchange = {
            "timestamp": datetime.utcnow().isoformat(),
            "user": user_message,
            "assistant": assistant_message,
        }
        
        # Include tool information if provided - this is CRITICAL for contextual awareness
        if tool_calls:
            exchange["tool_calls"] = tool_calls
        if tool_results:
            exchange["tool_results"] = tool_results
        
        self._histories[session_id].append(exchange)
        
        # Trim to max length
        if len(self._histories[session_id]) > self._max_history:
            self._histories[session_id] = self._histories[session_id][-self._max_history:]
        
        logger.debug(f"Added exchange to session {session_id}, history length: {len(self._histories[session_id])}, tool_calls: {len(tool_calls) if tool_calls else 0}")
    
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
    
    async def save_to_vectordb(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        force: bool = False
    ) -> bool:
        """
        Save conversation summary to vector database for persistent memory.
        
        This method indexes the current conversation history if it meets the
        threshold for indexing, or if force=True.
        
        Args:
            session_id: The session ID
            user_id: Optional user ID for filtering
            force: Force indexing regardless of threshold
            
        Returns:
            True if successfully indexed
        """
        history = self.get_history(session_id)
        
        if not history:
            logger.debug(f"No history to save for session {session_id}")
            return False
        
        # Check threshold
        if not force and len(history) < settings.conversation_summary_threshold:
            logger.debug(
                f"History length {len(history)} below threshold "
                f"{settings.conversation_summary_threshold}, skipping indexing"
            )
            return False
        
        try:
            success = await self.indexing.index_conversation_summary(
                session_id=session_id,
                history=history,
                user_id=user_id
            )
            
            if success:
                logger.info(f"Saved conversation summary to vector DB for session {session_id} (user_id: {user_id}, exchanges: {len(history)})")
            else:
                logger.warning(f"Failed to save conversation summary for session {session_id} (user_id: {user_id})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving to vector DB: {e}", exc_info=True)
            return False
    
    async def search_relevant_memories(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        user_id: Optional[str] = None
    ) -> List[dict]:
        """
        Search vector database for relevant past conversations.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            user_id: Filter by user ID
            
        Returns:
            List of relevant memory results with content and metadata
        """
        try:
            # Build metadata filter if user_id provided
            metadata_filter = None
            if user_id:
                metadata_filter = {"user_id": user_id}
            
            results = await self.vectordb.search(
                query=query,
                collection="conversation_summaries",
                top_k=top_k,
                threshold=threshold,
                metadata_filter=metadata_filter
            )
            
            # Format results for use in LLM context
            memories = []
            for result in results:
                memories.append({
                    "content": result.content,
                    "score": result.score,
                    "timestamp": result.metadata.get("timestamp"),
                    "topics": result.metadata.get("topics", []),
                    "session_id": result.metadata.get("session_id")
                })
            
            logger.info(f"Found {len(memories)} relevant memories for query '{query}' (user_id: {user_id})")
            if memories:
                for i, mem in enumerate(memories):
                    logger.debug(f"  Memory {i+1}: score={mem.get('score', 0):.3f}, content={mem.get('content', '')[:100]}...")
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}", exc_info=True)
            return []
    
    async def search_indexed_content(
        self,
        query: str,
        content_type: Optional[str] = None,
        repo: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[dict]:
        """
        Search vector database for relevant indexed content (GitHub files, docs, etc.).
        
        Args:
            query: Search query
            content_type: Filter by content type (e.g., 'github_file', 'documentation')
            repo: Filter by repository
            top_k: Number of results to return
            
        Returns:
            List of relevant content results
        """
        try:
            # Build metadata filter
            metadata_filter = {}
            if content_type:
                metadata_filter["content_type"] = content_type
            if repo:
                metadata_filter["repo"] = repo
            
            results = await self.vectordb.search(
                query=query,
                collection="indexed_content",
                top_k=top_k,
                metadata_filter=metadata_filter if metadata_filter else None
            )
            
            # Format results
            content_results = []
            for result in results:
                content_results.append({
                    "content": result.content,
                    "score": result.score,
                    "path": result.metadata.get("path"),
                    "repo": result.metadata.get("repo"),
                    "content_type": result.metadata.get("content_type"),
                    "language": result.metadata.get("language")
                })
            
            logger.debug(f"Found {len(content_results)} relevant content results")
            return content_results
            
        except Exception as e:
            logger.error(f"Error searching indexed content: {e}")
            return []
    
    async def get_combined_context(
        self,
        query: str,
        session_id: str,
        include_memories: bool = True,
        include_indexed_content: bool = True,
        user_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Get combined context from current history, past memories, and indexed content.
        
        This method searches both conversation summaries and indexed content,
        combining the results into a formatted context string for the LLM.
        
        Args:
            query: The user's query
            session_id: Current session ID
            include_memories: Whether to search past conversation summaries
            include_indexed_content: Whether to search indexed content
            user_id: Optional user ID for filtering memories
            
        Returns:
            Formatted context string or None if no context found
        """
        context_parts = []
        
        # Search past memories
        if include_memories:
            logger.debug(f"Searching memories for query: '{query}' with user_id: {user_id}")
            memories = await self.search_relevant_memories(
                query=query,
                user_id=user_id
            )
            
            logger.debug(f"Found {len(memories)} memories for query")
            
            if memories:
                context_parts.append("=== Relevant Past Conversations ===")
                for mem in memories:
                    topics = ", ".join(mem.get("topics", []))
                    context_parts.append(
                        f"[Score: {mem['score']:.2f}] {mem['content']}"
                        + (f" (Topics: {topics})" if topics else "")
                    )
        
        # Search indexed content
        if include_indexed_content:
            content = await self.search_indexed_content(query=query)
            
            if content:
                context_parts.append("\n=== Relevant Indexed Content ===")
                for item in content:
                    source = item.get("path") or item.get("content_type", "unknown")
                    repo = item.get("repo", "")
                    # Truncate content for context
                    truncated = item["content"][:500] + "..." if len(item["content"]) > 500 else item["content"]
                    context_parts.append(
                        f"[{source}] ({repo})\n{truncated}"
                    )
        
        result = "\n\n".join(context_parts) if context_parts else None
        if result:
            logger.debug(f"Returning combined context ({len(result)} chars) for query: '{query}' (user_id: {user_id})")
        else:
            logger.debug(f"No context found for query: '{query}' (user_id: {user_id})")
        return result if result else None
    
    def get_session_stats(self, session_id: str) -> dict:
        """
        Get statistics about a session's memory.
        
        Args:
            session_id: The session ID
            
        Returns:
            Dictionary with memory statistics
        """
        history = self.get_history(session_id)
        
        return {
            "exchange_count": len(history),
            "max_history": self._max_history,
            "first_exchange": history[0]["timestamp"] if history else None,
            "last_exchange": history[-1]["timestamp"] if history else None,
            "can_index": len(history) >= settings.conversation_summary_threshold
        }


# Global memory service instance
memory_service = MemoryService()
