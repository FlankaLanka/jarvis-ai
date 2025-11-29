"""
Jarvis Voice Assistant - LLM Service

Uses OpenAI GPT-4 API for intelligent responses with streaming support.
Integrates with vector database for context retrieval.
"""

from typing import List, Dict, Any, AsyncGenerator, Optional, Union
import json

from openai import AsyncOpenAI
from loguru import logger

from app.config import settings
from app.services.tools import GITHUB_TOOLS, ToolCall


# System prompt for the voice assistant
SYSTEM_PROMPT = """You are Jarvis, a real-time voice assistant designed for frontline workers in construction and project management.

=== YOUR CORE IDENTITY ===
- Voice-first assistant: All interactions are via voice, so keep responses concise and natural
- Construction project assistant: Specialized in managing construction projects, tasks, and progress
- Real-time data access: You can read and write to GitHub repositories for project data
- Memory-enabled: You remember past conversations through vector database

=== YOUR CAPABILITIES ===

1. INFORMATION RETRIEVAL:
   - Read construction project data from GitHub repository (tasks.json, progress-report.md, project-overview.md, team-responsibilities.md)
   - Query vector database for past conversation summaries and indexed content
   - Access real-time project status, task assignments, and progress
   - Search indexed documentation and code from GitHub

2. GITHUB WRITE OPERATIONS (AUTOMATIC):
   - UPDATE TASKS (update_task): Modify any field of an existing task
     * Updatable fields: status, progress, assigned_to, priority, due_date, title, description, location, estimated_hours, actual_hours, dependencies
     * Status options: pending, in_progress, completed, blocked
     * Priority options: low, medium, high, critical
     * Example: "Update task 1 to completed" or "Mark task-002 as 75% complete"
     * Example: "Assign task 3 to John Martinez" or "Change task 1 priority to critical"
     * Example: "Set task 5 due date to 2024-04-15"
     * Task IDs format: TASK-001, TASK-002, etc. (user can say "task 1", you convert to TASK-001)
   - CREATE TASKS (create_task): Add new tasks to the project
     * Required: title, description
     * Optional: assigned_to, priority, due_date, location, estimated_hours, dependencies
     * Example: "Create a task to inspect the foundation" or "Add a new task for electrical inspection"
   - DELETE TASKS (delete_task): Remove tasks from the project
     * Example: "Delete task 5" or "Remove task TASK-003"
   - ADD PROGRESS NOTES (add_progress_note): Record notes to progress-report.md
     * Example: "Add a note that we finished floor 5" or "Record that foundation is complete"
   - All writes automatically commit and push to GitHub (no manual push needed)

3. MEMORY & CONTEXT:
   - Access past conversation summaries via vector database
   - Use indexed GitHub content for semantic search
   - Remember project context across sessions
   - Combine vector DB context with fresh GitHub data for accuracy

4. VOICE INTERACTION:
   - Real-time speech-to-text transcription
   - Streaming text-to-speech responses
   - Natural conversation flow
   - Status feedback via audio cues

=== YOUR LIMITATIONS ===

1. WHAT YOU CANNOT DO:
   - Cannot execute arbitrary code or system commands
   - Cannot access files outside the configured GitHub repository
   - Cannot modify repository structure (create/delete directories)
   - Cannot perform git operations directly (GitHub API handles commits/pushes automatically)
   - Cannot access external APIs beyond GitHub and OpenAI
   - Cannot modify system settings or configurations
   - Cannot access user's local file system
   - Cannot perform destructive operations without explicit user request

2. DATA CONSTRAINTS:
   - Only works with the configured GitHub repository (default: FlankaLanka/sample-jarvis-read-write-repo)
   - Task operations (update/delete) only work for tasks that exist in tasks.json
   - Progress notes are appended to progress-report.md (cannot delete old notes)
   - Vector database requires Firebase configuration to persist data
   - Task IDs are auto-generated (TASK-XXX format) and cannot be customized

3. ERROR HANDLING:
   - If GitHub write fails, clearly state what went wrong
   - If task ID not found, inform user and suggest checking available tasks
   - If vector DB unavailable, still provide responses but note memory limitations
   - Always be transparent about failures

=== HOW TO USE YOUR CAPABILITIES ===

1. WHEN USER ASKS ABOUT PROJECT:
   - First check vector database for relevant past conversations
   - Then read fresh data from GitHub if needed (especially for "latest" or "current" queries)
   - Combine both sources for comprehensive answers
   - Cite your sources: "According to the progress report..." or "Based on our previous discussion..."

2. WHEN USER REQUESTS UPDATES/CHANGES:
   - For UPDATES: Confirm what will be changed: "I'll update task TASK-001 to completed status"
     * Extract task ID and the fields to update from user's request
     * Execute the update_task tool
     * Confirm success: "Task TASK-001 has been updated"
   - For NEW TASKS: Confirm what will be created: "I'll create a new task for foundation inspection"
     * Extract title, description, and optional fields from user's request
     * Execute the create_task tool
     * Confirm success with new task ID: "Created task TASK-009"
   - For DELETIONS: Confirm before deleting: "I'll delete task TASK-003"
     * Execute the delete_task tool
     * Confirm success: "Task TASK-003 has been deleted"
   - Note: All changes are automatically committed and pushed to GitHub

3. WHEN USER ASKS ABOUT TASKS:
   - Read tasks.json from GitHub
   - Provide specific information: task ID, status, progress, assigned person, due date
   - If task not found, list available tasks or ask for clarification

4. WHEN USER WANTS TO ADD NOTES:
   - Extract the note content from user's request
   - Add timestamp automatically
   - Append to progress-report.md
   - Confirm: "I've added your note to the progress report"

=== RESPONSE GUIDELINES ===

1. CONCISE: Voice interactions require brevity. Get to the point quickly.
2. ACCURATE: Only provide verified information. If uncertain, say "I'm not certain, but..." or "Let me check..."
3. HELPFUL: Focus on solving the immediate problem.
4. NATURAL: Speak conversationally, like talking to a colleague.
5. TRANSPARENT: If you're using vector DB context, mention it: "Based on our previous conversation..."
6. CONFIRMING: After write operations, always confirm what was done.

=== ERROR MESSAGES ===

When something fails, be specific:
- "I couldn't find task TASK-XXX. Available tasks are TASK-001, TASK-002..."
- "The GitHub write failed. The repository might be unavailable."
- "I don't have access to that information. It's not in the project repository."

=== REMEMBER ===
- Users are often in time-sensitive situations
- Every word should add value
- Voice-first means shorter, clearer responses
- When in doubt, ask one specific clarifying question
- Always confirm write operations before and after execution"""


class LLMService:
    """LLM service using OpenAI GPT-4 with vector database integration."""
    
    def __init__(self):
        self._client = None
        self._memory_service = None  # Lazy import
    
    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client
    
    @property
    def memory(self):
        """Lazy import of memory service to avoid circular dependencies."""
        if self._memory_service is None:
            from app.services.memory import memory_service
            self._memory_service = memory_service
        return self._memory_service
    
    async def get_vectordb_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_memories: bool = True,
        include_indexed_content: bool = True
    ) -> Optional[str]:
        """
        Retrieve relevant context from vector database.
        
        Args:
            query: The user's query
            session_id: Current session ID
            user_id: Optional user ID for filtering
            include_memories: Search past conversation summaries
            include_indexed_content: Search indexed content
            
        Returns:
            Formatted context string or None if no relevant context
        """
        try:
            context = await self.memory.get_combined_context(
                query=query,
                session_id=session_id or "",
                include_memories=include_memories,
                include_indexed_content=include_indexed_content,
                user_id=user_id
            )
            
            if context:
                logger.debug(f"Retrieved vector DB context ({len(context)} chars)")
            
            return context if context else None
            
        except Exception as e:
            logger.error(f"Error getting vector DB context: {e}")
            return None
    
    async def generate_response(
        self,
        query: str,
        conversation_history: List[dict] = None,
        session_id: str = None,
        context: Optional[str] = None,
        model: Optional[str] = None,
        use_vectordb: bool = False,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Generate a response to a user query.
        
        Args:
            query: The user's question or request
            conversation_history: Previous conversation exchanges
            session_id: Session ID for logging
            context: Additional context (e.g., GitHub data)
            model: Optional model override
            use_vectordb: Whether to retrieve context from vector DB
            user_id: Optional user ID for filtering vector DB results
            
        Returns:
            The assistant's response
        """
        # Optionally get vector DB context
        combined_context = context or ""
        
        if use_vectordb:
            vectordb_context = await self.get_vectordb_context(
                query=query,
                session_id=session_id,
                user_id=user_id
            )
            
            if vectordb_context:
                if combined_context:
                    combined_context = f"{combined_context}\n\n{vectordb_context}"
                else:
                    combined_context = vectordb_context
        
        messages = self._build_messages(query, conversation_history, combined_context or None)
        
        try:
            # Use provided model or fall back to default from settings
            model_to_use = model or settings.llm_model
            logger.debug(f"Generating response for session {session_id}, model: {model_to_use}, query: {query[:50]}...")
            
            response = await self.client.chat.completions.create(
                model=model_to_use,
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
        model: Optional[str] = None,
        use_vectordb: bool = False,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response to a user query.
        
        Args:
            query: The user's question or request
            conversation_history: Previous conversation exchanges
            session_id: Session ID for logging
            context: Additional context
            model: Optional model override
            use_vectordb: Whether to retrieve context from vector DB
            user_id: Optional user ID for filtering vector DB results
            
        Yields:
            Response chunks as they're generated
        """
        # Optionally get vector DB context
        combined_context = context or ""
        
        if use_vectordb:
            vectordb_context = await self.get_vectordb_context(
                query=query,
                session_id=session_id,
                user_id=user_id
            )
            
            if vectordb_context:
                if combined_context:
                    combined_context = f"{combined_context}\n\n{vectordb_context}"
                else:
                    combined_context = vectordb_context
        
        messages = self._build_messages(query, conversation_history, combined_context or None)
        model_to_use = model or settings.llm_model
        
        try:
            logger.debug(f"Streaming response for session {session_id}, model: {model_to_use}")
            
            stream = await self.client.chat.completions.create(
                model=model_to_use,
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
    
    async def generate_response_stream_with_tools(
        self,
        query: str,
        conversation_history: List[dict] = None,
        session_id: str = None,
        context: Optional[str] = None,
        model: Optional[str] = None,
        use_tools: bool = True,
    ) -> AsyncGenerator[Union[str, ToolCall], None]:
        """
        Generate a streaming response with tool calling support.
        
        This method allows the LLM to decide when to call tools (like updating tasks
        or adding progress notes) based on user intent.
        
        Args:
            query: The user's question or request
            conversation_history: Previous conversation exchanges
            session_id: Session ID for logging
            context: Additional context (e.g., from GitHub API, vector DB)
            model: Optional model override
            use_tools: Whether to enable tool calling (default True)
            
        Yields:
            Either text chunks (str) or tool calls (ToolCall) as they're generated
        """
        messages = self._build_messages(query, conversation_history, context)
        model_to_use = model or settings.llm_model
        
        try:
            logger.debug(f"Streaming response with tools for session {session_id}, model: {model_to_use}")
            
            # Create streaming request with tools
            stream_kwargs = {
                "model": model_to_use,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.7,
                "stream": True,
            }
            
            if use_tools:
                stream_kwargs["tools"] = GITHUB_TOOLS
                stream_kwargs["tool_choice"] = "auto"
            
            stream = await self.client.chat.completions.create(**stream_kwargs)
            
            # Track tool calls being built up across chunks
            current_tool_calls: Dict[int, dict] = {}  # index -> {id, name, arguments}
            
            async for chunk in stream:
                delta = chunk.choices[0].delta
                
                # Handle text content
                if delta.content:
                    yield delta.content
                
                # Handle tool calls
                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        idx = tool_call_delta.index
                        
                        # Initialize tool call tracking if new
                        if idx not in current_tool_calls:
                            current_tool_calls[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": ""
                            }
                        
                        # Accumulate tool call data
                        if tool_call_delta.id:
                            current_tool_calls[idx]["id"] = tool_call_delta.id
                        if tool_call_delta.function:
                            if tool_call_delta.function.name:
                                current_tool_calls[idx]["name"] = tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                current_tool_calls[idx]["arguments"] += tool_call_delta.function.arguments
                
                # Check if stream is finished (finish_reason)
                if chunk.choices[0].finish_reason == "tool_calls":
                    # Stream ended with tool calls - yield them
                    for idx in sorted(current_tool_calls.keys()):
                        tc = current_tool_calls[idx]
                        if tc["id"] and tc["name"]:
                            try:
                                arguments = json.loads(tc["arguments"]) if tc["arguments"] else {}
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse tool arguments: {tc['arguments']}")
                                arguments = {}
                            
                            tool_call = ToolCall(
                                id=tc["id"],
                                name=tc["name"],
                                arguments=arguments
                            )
                            logger.info(f"Tool call detected: {tool_call.name}({tool_call.arguments})")
                            yield tool_call
                    
        except Exception as e:
            logger.error(f"LLM streaming with tools error: {e}")
            raise
    
    async def generate_response_with_full_context(
        self,
        query: str,
        session_id: str,
        conversation_history: List[dict] = None,
        additional_context: Optional[str] = None,
        model: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response with automatic vector DB context retrieval.
        
        This is the recommended method for voice interactions as it:
        1. Automatically retrieves relevant context from vector DB
        2. Combines with conversation history
        3. Streams the response for low latency
        
        Args:
            query: The user's question or request
            session_id: Session ID
            conversation_history: Previous conversation exchanges
            additional_context: Extra context (e.g., from GitHub API)
            model: Optional model override
            user_id: Optional user ID for filtering
            
        Yields:
            Response chunks as they're generated
        """
        # Get vector DB context
        vectordb_context = await self.get_vectordb_context(
            query=query,
            session_id=session_id,
            user_id=user_id
        )
        
        # Combine contexts
        combined_context = ""
        if vectordb_context:
            combined_context = vectordb_context
        if additional_context:
            if combined_context:
                combined_context = f"{combined_context}\n\n{additional_context}"
            else:
                combined_context = additional_context
        
        # Stream the response
        async for chunk in self.generate_response_stream(
            query=query,
            conversation_history=conversation_history,
            session_id=session_id,
            context=combined_context or None,
            model=model
        ):
            yield chunk
    
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
            context: Additional context (including vector DB results)
            
        Returns:
            List of message dictionaries
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add context if provided (includes vector DB results)
        if context:
            messages.append({
                "role": "system",
                "content": f"Relevant context from memory and indexed content:\n{context}\n\nNote: This context comes from vector database searches and GitHub repository reads. Use it to provide accurate, up-to-date information."
            })
        
        # Add conversation history with tool call awareness
        if conversation_history:
            for exchange in conversation_history:
                messages.append({"role": "user", "content": exchange["user"]})
                
                # Build assistant message content including tool information
                assistant_content = exchange["assistant"]
                
                # If there were tool calls, append their results to give context
                if exchange.get("tool_calls") and exchange.get("tool_results"):
                    tool_summary = "\n\n[Actions I took in this exchange:]"
                    for i, (call, result) in enumerate(zip(exchange["tool_calls"], exchange["tool_results"])):
                        status = "✓ Success" if result.get("success") else "✗ Failed"
                        tool_summary += f"\n- Called {call['name']}({call.get('arguments', {})})"
                        tool_summary += f"\n  Result: {status} - {result.get('message', 'No message')}"
                        if result.get("data"):
                            # Include key data like task_id or updates
                            if result["data"].get("task_id"):
                                tool_summary += f" (Task: {result['data']['task_id']})"
                            if result["data"].get("updates"):
                                tool_summary += f" Updates: {result['data']['updates']}"
                    assistant_content += tool_summary
                
                messages.append({"role": "assistant", "content": assistant_content})
        
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
            "needs_memory": False,
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
        
        # Check for memory-related queries
        if any(word in query_lower for word in ["remember", "we discussed", "last time", "before", "earlier", "previously"]):
            capabilities["needs_memory"] = True
        
        return capabilities


# Global LLM service instance
llm_service = LLMService()

