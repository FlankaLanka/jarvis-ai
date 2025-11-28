"""
Jarvis Voice Assistant - LLM Service

Uses OpenAI GPT-4 API for intelligent responses with streaming support.
"""

from typing import List, Dict, Any, AsyncGenerator, Optional

from openai import AsyncOpenAI
from loguru import logger

from app.config import settings


# System prompt for the voice assistant
SYSTEM_PROMPT = """You are Jarvis, a real-time voice assistant designed for frontline workers. 
Your responses must be:

1. CONCISE: Keep responses short and actionable. Voice interactions require brevity.
2. ACCURATE: Only provide verified information. If uncertain, say so clearly.
3. HELPFUL: Focus on solving the user's immediate problem.
4. NATURAL: Speak conversationally, as if talking to a colleague.

Key behaviors:
- Answer questions directly without unnecessary preamble
- If you need clarification, ask one specific question
- For technical queries, provide step-by-step guidance
- When accessing GitHub or API data, cite your sources
- If you cannot help with something, explain why briefly

Remember: Users are often in time-sensitive situations. Every word should add value."""


class LLMService:
    """LLM service using OpenAI GPT-4."""
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client
    
    async def generate_response(
        self,
        query: str,
        conversation_history: List[dict] = None,
        session_id: str = None,
        context: Optional[str] = None,
    ) -> str:
        """
        Generate a response to a user query.
        
        Args:
            query: The user's question or request
            conversation_history: Previous conversation exchanges
            session_id: Session ID for logging
            context: Additional context (e.g., GitHub data)
            
        Returns:
            The assistant's response
        """
        messages = self._build_messages(query, conversation_history, context)
        
        try:
            logger.debug(f"Generating response for session {session_id}, query: {query[:50]}...")
            
            response = await self.client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                max_tokens=500,  # Keep responses concise for voice
                temperature=0.7,
            )
            
            result = response.choices[0].message.content
            logger.debug(f"Generated response: {result[:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise
    
    async def generate_response_stream(
        self,
        query: str,
        conversation_history: List[dict] = None,
        session_id: str = None,
        context: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response to a user query.
        
        Args:
            query: The user's question or request
            conversation_history: Previous conversation exchanges
            session_id: Session ID for logging
            context: Additional context
            
        Yields:
            Response chunks as they're generated
        """
        messages = self._build_messages(query, conversation_history, context)
        
        try:
            logger.debug(f"Streaming response for session {session_id}")
            
            stream = await self.client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise
    
    def _build_messages(
        self,
        query: str,
        conversation_history: List[dict] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build the messages array for the API call.
        
        Args:
            query: The user's query
            conversation_history: Previous exchanges
            context: Additional context
            
        Returns:
            List of message dictionaries
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add context if provided
        if context:
            messages.append({
                "role": "system",
                "content": f"Additional context:\n{context}"
            })
        
        # Add conversation history
        if conversation_history:
            for exchange in conversation_history:
                messages.append({"role": "user", "content": exchange["user"]})
                messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    async def check_capabilities(self, query: str) -> Dict[str, Any]:
        """
        Check what the assistant can do to answer a query.
        Used for self-awareness responses.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary describing capabilities
        """
        capabilities = {
            "can_answer": True,
            "needs_github": False,
            "needs_api": False,
            "confidence": "high",
            "limitations": [],
        }
        
        query_lower = query.lower()
        
        # Check for GitHub-related queries
        if any(word in query_lower for word in ["github", "repo", "code", "commit", "pr", "pull request"]):
            capabilities["needs_github"] = True
            if not settings.github_token:
                capabilities["limitations"].append("GitHub integration not configured")
                capabilities["confidence"] = "low"
        
        # Check for real-time data queries
        if any(word in query_lower for word in ["current", "now", "today", "latest"]):
            capabilities["needs_api"] = True
            capabilities["limitations"].append("Real-time data may be up to 3 minutes old")
        
        return capabilities


# Global LLM service instance
llm_service = LLMService()

