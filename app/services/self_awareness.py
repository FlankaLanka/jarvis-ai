"""
Jarvis Voice Assistant - Self-Awareness Service

Provides system capability detection and explanations.
"""

from typing import Dict, List, Any

from loguru import logger

from app.config import settings


class SelfAwarenessService:
    """
    Service for system self-awareness responses.
    
    Provides:
    1. Capability detection
    2. Limitation explanations
    3. Help responses
    """
    
    def __init__(self):
        self._capabilities = {
            "voice_interaction": {
                "name": "Voice Interaction",
                "description": "I can listen to your voice and respond with speech.",
                "available": True,
            },
            "conversation_memory": {
                "name": "Conversation Memory",
                "description": "I remember our conversation context within a session.",
                "available": True,
            },
            "github_integration": {
                "name": "GitHub Integration",
                "description": "I can search code, read files, and view commit history from GitHub.",
                "available": bool(settings.github_token),
            },
            "api_data": {
                "name": "API Data Access",
                "description": "I can fetch and analyze data from configured APIs.",
                "available": True,
            },
            "interruption": {
                "name": "Interruptibility",
                "description": "You can interrupt me at any time while I'm speaking.",
                "available": True,
            },
        }
        
        self._limitations = [
            "I cannot browse the web or access real-time internet data.",
            "My knowledge has a training cutoff date.",
            "I cannot execute code or make changes to systems.",
            "I can only access GitHub repositories I'm configured for.",
            "Audio quality depends on your microphone and connection.",
        ]
        
        self._help_triggers = [
            "what can you do",
            "help",
            "capabilities",
            "what are you",
            "who are you",
            "how do you work",
        ]
    
    def is_self_awareness_query(self, query: str) -> bool:
        """
        Check if a query is asking about system capabilities.
        
        Args:
            query: The user's query
            
        Returns:
            True if this is a self-awareness query
        """
        query_lower = query.lower()
        return any(trigger in query_lower for trigger in self._help_triggers)
    
    def get_capabilities_response(self) -> str:
        """
        Generate a response explaining capabilities.
        
        Returns:
            Human-readable capabilities description
        """
        available = []
        unavailable = []
        
        for key, cap in self._capabilities.items():
            if cap["available"]:
                available.append(f"- {cap['name']}: {cap['description']}")
            else:
                unavailable.append(f"- {cap['name']}: Currently unavailable")
        
        response_parts = [
            "I'm Jarvis, your voice assistant. Here's what I can do:",
            "",
            *available,
        ]
        
        if unavailable:
            response_parts.extend([
                "",
                "Currently unavailable:",
                *unavailable,
            ])
        
        return "\n".join(response_parts)
    
    def get_limitations_response(self) -> str:
        """
        Generate a response explaining limitations.
        
        Returns:
            Human-readable limitations description
        """
        return "Here are my current limitations:\n" + "\n".join(
            f"- {lim}" for lim in self._limitations
        )
    
    def get_help_response(self, query: str) -> str:
        """
        Generate a contextual help response.
        
        Args:
            query: The user's query
            
        Returns:
            Help response text
        """
        query_lower = query.lower()
        
        if "limitation" in query_lower or "can't" in query_lower:
            return self.get_limitations_response()
        
        if "github" in query_lower:
            if self._capabilities["github_integration"]["available"]:
                return (
                    "I can help you with GitHub! Try asking me to:\n"
                    "- Search for code in a repository\n"
                    "- Read a specific file\n"
                    "- Show recent commits\n"
                    "- Get repository information"
                )
            else:
                return "GitHub integration isn't configured yet. Please set up a GitHub token."
        
        # Default to capabilities
        return self.get_capabilities_response()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            Status dictionary
        """
        return {
            "capabilities": {
                k: v["available"] for k, v in self._capabilities.items()
            },
            "configured": {
                "openai": bool(settings.openai_api_key),
                "github": bool(settings.github_token),
            },
        }
    
    def check_capability(self, capability: str) -> bool:
        """
        Check if a specific capability is available.
        
        Args:
            capability: The capability name
            
        Returns:
            True if available
        """
        cap = self._capabilities.get(capability)
        return cap["available"] if cap else False


# Global self-awareness service instance
self_awareness_service = SelfAwarenessService()

