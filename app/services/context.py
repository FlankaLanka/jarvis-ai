"""
Jarvis Voice Assistant - Context Stitching Service

Manages conversation context for follow-up questions and multi-turn conversations.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from loguru import logger


@dataclass
class ContextWindow:
    """Represents a context window for the LLM."""
    messages: List[Dict[str, str]]
    token_count: int
    max_tokens: int


class ContextService:
    """
    Service for managing conversation context and follow-up handling.
    
    Responsibilities:
    1. Build context windows from conversation history
    2. Detect follow-up questions
    3. Inject relevant context for ambiguous queries
    4. Summarize long conversations to fit context limits
    """
    
    def __init__(self, max_context_tokens: int = 4000):
        """
        Initialize the context service.
        
        Args:
            max_context_tokens: Maximum tokens to include in context
        """
        self._max_tokens = max_context_tokens
    
    def is_follow_up(self, query: str, history: List[dict]) -> bool:
        """
        Detect if a query is a follow-up to previous conversation.
        
        Args:
            query: The current query
            history: Conversation history
            
        Returns:
            True if this appears to be a follow-up question
        """
        if not history:
            return False
        
        query_lower = query.lower().strip()
        
        # Check for pronouns that reference previous context
        follow_up_indicators = [
            "it", "that", "this", "they", "them", "those",
            "what about", "how about", "and", "also",
            "more", "else", "another", "same", "again",
            "why", "how", "when", "where",  # Question words alone
        ]
        
        # Check for very short queries (likely follow-ups)
        word_count = len(query_lower.split())
        if word_count <= 3:
            # Short query - likely a follow-up
            return True
        
        # Check for follow-up indicators at the start
        for indicator in follow_up_indicators:
            if query_lower.startswith(indicator):
                return True
        
        return False
    
    def build_context(
        self,
        query: str,
        history: List[dict],
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build a context object for the LLM.
        
        Args:
            query: The current query
            history: Conversation history
            additional_context: Extra context (e.g., GitHub data)
            
        Returns:
            Context dictionary with messages and metadata
        """
        is_follow_up = self.is_follow_up(query, history)
        
        # Determine how much history to include
        if is_follow_up:
            # Include more history for follow-up questions
            relevant_history = history[-5:] if len(history) > 5 else history
        else:
            # Less history for new topics
            relevant_history = history[-2:] if len(history) > 2 else history
        
        context = {
            "is_follow_up": is_follow_up,
            "history": relevant_history,
            "additional_context": additional_context,
            "query": query,
        }
        
        logger.debug(f"Built context: follow_up={is_follow_up}, history_length={len(relevant_history)}")
        return context
    
    def resolve_references(
        self,
        query: str,
        history: List[dict]
    ) -> str:
        """
        Resolve ambiguous references in a query using history.
        
        Args:
            query: The query with potential references
            history: Conversation history
            
        Returns:
            Query with resolved references
        """
        if not history:
            return query
        
        query_lower = query.lower()
        last_exchange = history[-1]
        
        # Simple reference resolution
        # Can be made more sophisticated with NLP
        
        # Handle "it" references
        if " it " in query_lower or query_lower.startswith("it "):
            # Try to find what "it" refers to from previous exchange
            # This is a simplified heuristic
            last_response = last_exchange.get("assistant", "")
            # For now, just note that it's a reference
            logger.debug("Detected 'it' reference in query")
        
        # Handle "that" references
        if " that " in query_lower or query_lower.startswith("that "):
            logger.debug("Detected 'that' reference in query")
        
        return query
    
    def summarize_history(
        self,
        history: List[dict],
        max_length: int = 500
    ) -> str:
        """
        Summarize conversation history for context.
        
        Args:
            history: Conversation history
            max_length: Maximum characters for summary
            
        Returns:
            Summary string
        """
        if not history:
            return ""
        
        summary_parts = []
        current_length = 0
        
        # Start from most recent
        for exchange in reversed(history):
            user_msg = exchange.get("user", "")[:100]
            assistant_msg = exchange.get("assistant", "")[:100]
            
            part = f"User asked about: {user_msg}..."
            
            if current_length + len(part) > max_length:
                break
            
            summary_parts.insert(0, part)
            current_length += len(part)
        
        return " ".join(summary_parts)
    
    def get_relevant_topics(self, history: List[dict]) -> List[str]:
        """
        Extract relevant topics from conversation history.
        
        Args:
            history: Conversation history
            
        Returns:
            List of topic keywords
        """
        topics = set()
        
        # Simple keyword extraction
        # Can be improved with NLP
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be",
            "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must",
            "i", "you", "he", "she", "it", "we", "they",
            "what", "how", "why", "when", "where", "which",
            "this", "that", "these", "those", "can", "to",
            "of", "in", "for", "on", "with", "at", "by"
        }
        
        for exchange in history:
            for text in [exchange.get("user", ""), exchange.get("assistant", "")]:
                words = text.lower().split()
                for word in words:
                    word = word.strip(".,!?\"'")
                    if len(word) > 3 and word not in stop_words:
                        topics.add(word)
        
        return list(topics)[:10]  # Limit to top 10 topics


# Global context service instance
context_service = ContextService()

