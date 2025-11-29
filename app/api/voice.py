"""
Jarvis Voice Assistant - Voice API Endpoints

Handles voice interactions including audio streaming, STT, TTS, and LLM processing.
Includes vector database integration for persistent memory.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncio

from loguru import logger

from app.services.stt import stt_service
from app.services.tts import tts_service
from app.services.llm import llm_service
from app.services.memory import memory_service
from app.services.interrupt import interrupt_handler
from app.services.session import session_manager
from app.services.vectordb import vectordb_service
from app.services.indexing import indexing_service
from app.services.tools import ToolCall, execute_tool, ToolResult
from app.integrations.github import github_service
from app.config import settings
import json


router = APIRouter()


# Construction-related query patterns for smart GitHub reading
CONSTRUCTION_QUERY_PATTERNS = {
    "project": ["project", "construction", "site", "building", "development"],
    "tasks": ["task", "assignment", "responsible", "due date", "deadline", "work"],
    "progress": ["progress", "status", "update", "completion", "percent", "latest", "current"],
    "team": ["team", "lead", "coordinator", "manager", "crew", "worker", "responsible"],
}

# File mapping for construction queries
CONSTRUCTION_FILES = {
    "project": "project-overview.md",
    "tasks": "tasks.json",
    "progress": "progress-report.md",
    "team": "team-responsibilities.md",
}


async def get_github_context(query: str, vectordb_context: Optional[str] = None) -> Optional[str]:
    """
    Retrieve relevant GitHub file content based on query analysis.
    
    This function:
    1. Analyzes the query to determine which construction files are needed
    2. Reads the relevant files directly from GitHub for fresh data
    3. Formats the content for LLM consumption
    
    Args:
        query: The user's query
        vectordb_context: Existing vector DB context (to check if we already have data)
        
    Returns:
        Formatted GitHub context string or None
    """
    query_lower = query.lower()
    files_to_read = []
    
    # Determine which files to read based on query patterns
    for category, patterns in CONSTRUCTION_QUERY_PATTERNS.items():
        if any(pattern in query_lower for pattern in patterns):
            files_to_read.append(CONSTRUCTION_FILES[category])
    
    # If query mentions specific things like "latest" or "current", prioritize fresh data
    if any(word in query_lower for word in ["latest", "current", "now", "today", "recent"]):
        # Add progress report for latest updates
        if "progress-report.md" not in files_to_read:
            files_to_read.append("progress-report.md")
    
    # If no specific patterns matched but query seems construction-related
    if not files_to_read and any(word in query_lower for word in ["construction", "project", "site"]):
        files_to_read = ["project-overview.md", "progress-report.md"]
    
    if not files_to_read:
        return None
    
    # Read files from GitHub
    repo = settings.github_default_repo or "FlankaLanka/sample-jarvis-read-write-repo"
    context_parts = []
    
    for file_path in files_to_read:
        try:
            file_content = github_service.get_file_content(repo, file_path)
            if file_content:
                # Format based on file type
                if file_path.endswith(".json"):
                    # Parse JSON for better readability
                    try:
                        data = json.loads(file_content.content)
                        if file_path == "tasks.json":
                            # Format tasks nicely
                            formatted = format_tasks_for_context(data)
                            context_parts.append(f"=== Construction Tasks ===\n{formatted}")
                        else:
                            context_parts.append(f"=== {file_path} ===\n{json.dumps(data, indent=2)}")
                    except json.JSONDecodeError:
                        context_parts.append(f"=== {file_path} ===\n{file_content.content}")
                else:
                    context_parts.append(f"=== {file_path} ===\n{file_content.content}")
                
                logger.debug(f"Read GitHub file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not read GitHub file {file_path}: {e}")
    
    if context_parts:
        return "\n\n".join(context_parts)
    
    return None


def format_tasks_for_context(tasks_data: dict) -> str:
    """Format tasks.json data for LLM context."""
    tasks = tasks_data.get("tasks", [])
    last_updated = tasks_data.get("last_updated", "Unknown")
    
    lines = [f"Last Updated: {last_updated}", ""]
    
    for task in tasks:
        status_emoji = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅"
        }.get(task.get("status", ""), "❓")
        
        lines.append(f"{status_emoji} {task.get('id', 'N/A')}: {task.get('title', 'Untitled')}")
        lines.append(f"   Status: {task.get('status', 'unknown')} | Progress: {task.get('progress', 0)}%")
        lines.append(f"   Assigned to: {task.get('assigned_to', 'Unassigned')}")
        lines.append(f"   Due: {task.get('due_date', 'No due date')}")
        lines.append(f"   Location: {task.get('location', 'N/A')}")
        lines.append("")
    
    return "\n".join(lines)


class TextQueryRequest(BaseModel):
    """Request model for text-based queries (fallback for testing)."""
    text: str
    session_id: str


class TextQueryResponse(BaseModel):
    """Response model for text-based queries."""
    response: str
    session_id: str


class TranscriptionResponse(BaseModel):
    """Response model for audio transcription."""
    text: str
    confidence: float


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """
    Transcribe audio to text using OpenAI Whisper.
    
    Args:
        audio: Audio file to transcribe
        session_id: Optional session ID for context
    
    Returns:
        Transcribed text and confidence score
    """
    try:
        audio_bytes = await audio.read()
        result = await stt_service.transcribe(audio_bytes)
        return TranscriptionResponse(
            text=result["text"],
            confidence=result.get("confidence", 1.0)
        )
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/synthesize")
async def synthesize_speech(text: str, voice: Optional[str] = None):
    """
    Convert text to speech using OpenAI TTS.
    
    Args:
        text: Text to convert to speech
        voice: Optional voice selection
    
    Returns:
        Audio stream
    """
    try:
        audio_stream = tts_service.synthesize_stream(text, voice=voice)
        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")


@router.post("/query", response_model=TextQueryResponse)
async def text_query(request: TextQueryRequest):
    """
    Process a text query and return a text response.
    This is a fallback for testing without audio.
    
    Args:
        request: Query request with text and session ID
    
    Returns:
        Text response from LLM
    """
    try:
        # Get or create session
        session = session_manager.get_or_create_session(request.session_id)
        
        # Get conversation history
        history = memory_service.get_history(request.session_id)
        
        # Process with LLM
        response = await llm_service.generate_response(
            query=request.text,
            conversation_history=history,
            session_id=request.session_id
        )
        
        # Store in memory
        memory_service.add_exchange(
            session_id=request.session_id,
            user_message=request.text,
            assistant_message=response
        )
        
        return TextQueryResponse(
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.websocket("/stream")
async def voice_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice streaming.
    
    Handles bidirectional audio:
    - Client sends audio chunks for STT
    - Server sends audio responses via TTS
    """
    await websocket.accept()
    session_id = None
    selected_model = None  # Store model preference per session
    current_request_key = None  # Unique key for current request - must be echoed in responses
    
    try:
        # Wait for initial session setup
        init_data = await websocket.receive_json()
        session_id = init_data.get("session_id")
        selected_model = init_data.get("model")  # Get model from init
        
        # Get or create session and extract user_id
        if not session_id:
            session_id = session_manager.create_session()
        else:
            session_manager.get_or_create_session(session_id)
        
        # Get session to retrieve user_id
        session = session_manager.get_session(session_id)
        user_id = session.get("user_id") if session else settings.default_user_id
        
        # Send session confirmation
        await websocket.send_json({
            "type": "session_ready",
            "session_id": session_id
        })
        
        logger.info(f"Voice stream started for session: {session_id}, user: {user_id}, model: {selected_model}")
        
        # Main processing loop
        while True:
            try:
                # Receive message (can be audio data or control message)
                message = await websocket.receive()
                
                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        # Audio data received
                        audio_chunk = message["bytes"]
                        # Get current session to retrieve user_id (in case it changed)
                        current_session = session_manager.get_session(session_id)
                        current_user_id = current_session.get("user_id") if current_session else user_id
                        await process_audio_chunk(websocket, session_id, audio_chunk, selected_model, current_user_id, current_request_key)
                    elif "text" in message:
                        # Control message
                        import json
                        data = json.loads(message["text"])
                        # Update model and request_key if set_model message
                        if data.get("type") == "set_model":
                            selected_model = data.get("model")
                            current_request_key = data.get("request_key")
                            logger.info(f"Model updated to {selected_model} for session {session_id}, request_key: {current_request_key}")
                        else:
                            # Get current session to retrieve user_id
                            current_session = session_manager.get_session(session_id)
                            current_user_id = current_session.get("user_id") if current_session else user_id
                            await handle_control_message(websocket, session_id, data, current_user_id)
                        
            except asyncio.CancelledError:
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        if session_id:
            interrupt_handler.cancel_session(session_id)


async def process_audio_chunk(websocket: WebSocket, session_id: str, audio_chunk: bytes, model: str = None, user_id: str = None, request_key: str = None):
    """Process an incoming audio chunk through the STT → LLM → TTS pipeline.
    
    Args:
        websocket: The WebSocket connection
        session_id: Session identifier
        audio_chunk: Audio data bytes
        model: LLM model to use
        user_id: User identifier
        request_key: Unique key for this request - must be included in all responses
    """
    
    async def send_response(data: dict):
        """Helper to send JSON response with request_key included."""
        if request_key:
            data["request_key"] = request_key
        await websocket.send_json(data)
    
    try:
        # Use default user_id if not provided
        if user_id is None:
            user_id = settings.default_user_id
        
        # Check minimum audio size (roughly 0.1 seconds minimum for Whisper)
        # WebM/Opus encoding: ~1KB per 0.1 seconds is reasonable minimum
        MIN_AUDIO_SIZE = 1000  # bytes
        
        if len(audio_chunk) < MIN_AUDIO_SIZE:
            logger.debug(f"Audio chunk too small ({len(audio_chunk)} bytes), ignoring")
            await send_response({"type": "status", "status": "no_speech"})
            return
        
        # Check for interrupt - if set, clear it and continue with new audio
        # This handles the case where user interrupted and is now sending new audio
        if interrupt_handler.is_interrupted(session_id):
            interrupt_handler.clear_interrupt(session_id)
            logger.debug(f"Cleared stale interrupt flag for session {session_id}, processing new audio")
        
        # Send processing status
        await send_response({"type": "status", "status": "processing"})
        
        # Transcribe audio
        transcription = await stt_service.transcribe(audio_chunk)
        text = transcription["text"]
        
        if not text.strip():
            await send_response({"type": "status", "status": "no_speech"})
            return
        
        # Send transcription
        await send_response({
            "type": "transcription",
            "text": text
        })
        
        # Send thinking status
        await send_response({"type": "status", "status": "thinking"})
        
        # Get conversation history
        history = memory_service.get_history(session_id)
        
        # Retrieve vector DB context (past conversations + indexed content)
        vectordb_context = None
        try:
            # Send UI indicator for vector DB query
            await send_response({
                "type": "system_message",
                "message": "🔍 Querying vector database...",
                "category": "vectordb"
            })
            
            vectordb_context = await memory_service.get_combined_context(
                query=text,
                session_id=session_id,
                include_memories=True,
                include_indexed_content=True,
                user_id=user_id
            )
            if vectordb_context:
                logger.info(f"Retrieved vector DB context ({len(vectordb_context)} chars) for query: '{text[:50]}...' (user_id: {user_id})")
                await send_response({
                    "type": "system_message",
                    "message": "✅ Found relevant context in vector database",
                    "category": "vectordb"
                })
            else:
                logger.info(f"No vector DB context found for query: '{text[:50]}...' (user_id: {user_id})")
                await send_response({
                    "type": "system_message",
                    "message": "ℹ️ No relevant context found in vector database",
                    "category": "vectordb"
                })
        except Exception as e:
            logger.warning(f"Could not retrieve vector DB context: {e}")
            await send_response({
                "type": "system_message",
                "message": "⚠️ Vector database query failed",
                "category": "vectordb"
            })
        
        # Retrieve GitHub context for construction-related queries
        # This reads fresh data directly from GitHub for real-time accuracy
        github_context = None
        try:
            # Check if we might need GitHub data
            if any(word in text.lower() for word in ["project", "construction", "task", "progress", "team", "status"]):
                await send_response({
                    "type": "system_message",
                    "message": "📁 Reading from GitHub repository...",
                    "category": "github"
                })
            
            github_context = await get_github_context(text, vectordb_context)
            if github_context:
                logger.debug(f"Retrieved GitHub context ({len(github_context)} chars)")
                await send_response({
                    "type": "system_message",
                    "message": "✅ Retrieved data from GitHub",
                    "category": "github"
                })
        except Exception as e:
            logger.warning(f"Could not retrieve GitHub context: {e}")
            await send_response({
                "type": "system_message",
                "message": "⚠️ GitHub read failed",
                "category": "github"
            })
        
        # Combine contexts: GitHub context takes priority for fresh data
        # Add capability awareness to context
        capability_info = """
=== CURRENT SYSTEM STATUS ===
- Vector Database: Available (provides past conversation context and indexed content)
- GitHub Repository: Available (provides real-time project data)
- Write Operations: Enabled (can update tasks, add progress notes)
- Auto-Push: Enabled (all commits automatically pushed to remote)
- Default Repository: {repo}

=== AVAILABLE OPERATIONS ===
1. Read Operations:
   - Query tasks.json for task information
   - Read progress-report.md for latest updates
   - Access project-overview.md for project details
   - Search team-responsibilities.md for team info

2. Write Operations:
   - Update task status/progress: "Update task TASK-001 to completed"
   - Update task progress: "Set task TASK-002 to 75% complete"
   - Add progress notes: "Add a note that foundation is complete"
   - All writes automatically commit and push

3. Memory Operations:
   - Access past conversation summaries
   - Search indexed GitHub content semantically
   - Combine historical context with fresh data

=== IMPORTANT REMINDERS ===
- Always confirm what you're about to change before executing writes
- If task ID not found, list available tasks
- If write fails, explain the error clearly
- Use GitHub data for current/latest information
- Use vector DB for historical context and past discussions
""".format(repo=settings.github_default_repo or "FlankaLanka/sample-jarvis-read-write-repo")
        
        combined_context = None
        if github_context and vectordb_context:
            combined_context = f"{capability_info}\n\n=== FRESH DATA FROM GITHUB ===\n{github_context}\n\n=== HISTORICAL CONTEXT FROM VECTOR DB ===\n{vectordb_context}"
        elif github_context:
            combined_context = f"{capability_info}\n\n=== FRESH DATA FROM GITHUB ===\n{github_context}"
        elif vectordb_context:
            combined_context = f"{capability_info}\n\n=== HISTORICAL CONTEXT FROM VECTOR DB ===\n{vectordb_context}"
        else:
            # Even without context, provide capability info
            combined_context = capability_info
        
        # Send speaking status (we'll start speaking as soon as we have first sentence)
        await send_response({"type": "status", "status": "speaking"})
        
        # Stream LLM response and generate TTS incrementally
        # Use sequential processing to maintain correct order while keeping latency low
        import asyncio
        sentence_queue = asyncio.Queue()  # Queue to hold sentences in order
        tts_complete = asyncio.Event()  # Signal when all TTS is done
        
        full_response = ""
        current_sentence = ""
        first_sentence_sent = False
        last_char_was_ending = False
        last_token_was_ending = False  # Track if last token was ending punctuation
        
        async def tts_processor():
            """Process sentences sequentially to maintain order."""
            nonlocal first_sentence_sent
            sentence_index = 0
            while True:
                try:
                    # Check for interrupt before waiting for next sentence
                    if interrupt_handler.is_interrupted(session_id):
                        logger.debug("TTS processor interrupted, exiting")
                        break
                    
                    # Get next sentence from queue with timeout to allow interrupt checks
                    try:
                        item = await asyncio.wait_for(sentence_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        # Check for interrupt during timeout
                        if interrupt_handler.is_interrupted(session_id):
                            logger.debug("TTS processor interrupted during queue wait")
                            break
                        continue
                    
                    # None signals end of sentences
                    if item is None:
                        break
                    
                    sentence, is_first = item
                    if not sentence or not sentence.strip():
                        sentence_queue.task_done()
                        continue
                    
                    # Check interrupt before TTS generation
                    if interrupt_handler.is_interrupted(session_id):
                        sentence_queue.task_done()
                        break
                    
                    try:
                        # Generate complete TTS audio for this sentence
                        # Buffer all chunks, then send as one complete audio
                        audio_chunks = []
                        async for audio_chunk in tts_service.synthesize_stream_async(sentence):
                            if interrupt_handler.is_interrupted(session_id):
                                break
                            audio_chunks.append(audio_chunk)
                        
                        # Combine all chunks into one complete audio
                        if audio_chunks:
                            complete_audio = b''.join(audio_chunks)
                            
                            # Signal start of a new audio segment
                            await send_response({
                                "type": "audio_start",
                                "sentence_index": sentence_index,
                                "size": len(complete_audio)
                            })
                            
                            # Send the complete audio for this sentence
                            await websocket.send_bytes(complete_audio)
                            
                            # Signal end of this audio segment
                            await send_response({
                                "type": "audio_end",
                                "sentence_index": sentence_index
                            })
                            
                            logger.debug(f"Sent complete audio for sentence {sentence_index + 1} ({len(complete_audio)} bytes)")
                        
                        if is_first and not first_sentence_sent:
                            first_sentence_sent = True
                            logger.debug(f"First sentence TTS generated and sent (sentence {sentence_index + 1})")
                        
                        sentence_index += 1
                    except Exception as e:
                        logger.error(f"Error generating TTS for sentence {sentence_index + 1}: {e}")
                    
                    sentence_queue.task_done()
                except Exception as e:
                    logger.error(f"Error in TTS processor: {e}")
                    break
            
            tts_complete.set()
            logger.debug("TTS processor completed")
        
        # Start the TTS processor task
        tts_processor_task = asyncio.create_task(tts_processor())
        
        # Collect tool calls while streaming
        tool_calls_to_execute: List[ToolCall] = []
        
        async for item in llm_service.generate_response_stream_with_tools(
            query=text,
            conversation_history=history,
            session_id=session_id,
            model=model,
            context=combined_context,
            use_tools=True
        ):
            if interrupt_handler.is_interrupted(session_id):
                break
            
            # Check if this is a tool call or text
            if isinstance(item, ToolCall):
                # Collect tool call for execution after streaming
                tool_calls_to_execute.append(item)
                logger.info(f"Tool call received: {item.name}({item.arguments})")
                
                # Notify frontend that a tool will be executed
                await send_response({
                    "type": "tool_call",
                    "name": item.name,
                    "arguments": item.arguments
                })
                continue
            
            # It's a text token
            token = item
            full_response += token
            current_sentence += token
            
            # Send token to frontend for display
            await send_response({
                "type": "response_chunk",
                "text": token
            })
            
            # Check for sentence boundaries
            sentence_endings = ['.', '!', '?']
            token_is_ending = token in sentence_endings
            token_is_space = token == ' ' or token == '\n'
            
            # If we just got a sentence ending, mark it
            if token_is_ending:
                last_char_was_ending = True
                last_token_was_ending = True
            elif token_is_space and last_char_was_ending:
                # We have a complete sentence (ending + space)
                sentence_to_speak = current_sentence.strip()
                if sentence_to_speak:
                    # Add sentence to queue for sequential processing
                    is_first = not first_sentence_sent
                    await sentence_queue.put((sentence_to_speak, is_first))
                    logger.debug(f"Queued sentence for TTS: '{sentence_to_speak[:50]}...'")
                    
                    # Reset for next sentence
                    current_sentence = ""
                    last_char_was_ending = False
                    last_token_was_ending = False
            elif not token_is_space:
                # Reset ending flag if we get a non-space character
                if not last_token_was_ending:
                    last_char_was_ending = False
                last_token_was_ending = False
            
            # Fallback: If we have >80 chars without a sentence end, generate TTS anyway
            # This prevents long delays for responses without punctuation
            if len(current_sentence) > 80 and not any(ending in current_sentence[-10:] for ending in sentence_endings):
                sentence_to_speak = current_sentence.strip()
                if sentence_to_speak and len(sentence_to_speak) > 20:  # Only if substantial
                    is_first = not first_sentence_sent
                    await sentence_queue.put((sentence_to_speak + " ", is_first))
                    logger.debug(f"Queued long sentence for TTS (fallback): '{sentence_to_speak[:50]}...'")
                    
                    current_sentence = ""
                    last_char_was_ending = False
                    last_token_was_ending = False
        
        # CRITICAL: Process any remaining text - this ensures ALL text gets TTS
        # Check if we were interrupted during LLM streaming
        was_interrupted = interrupt_handler.is_interrupted(session_id)
        
        if not was_interrupted:
            # This handles cases where:
            # 1. Text ends without punctuation
            # 2. Text ends with punctuation but no space after
            # 3. Text ends mid-sentence
            remaining_text = current_sentence.strip()
            if remaining_text:
                logger.debug(f"Processing remaining text for TTS: {remaining_text[:50]}...")
                is_first = not first_sentence_sent
                await sentence_queue.put((remaining_text, is_first))
        
        # Signal end of sentences to TTS processor (always send to allow clean exit)
        await sentence_queue.put(None)
        
        # Wait for TTS processor to complete (with timeout to handle interrupt case)
        if not was_interrupted:
            logger.debug("Waiting for TTS processor to complete all sentences...")
            try:
                await asyncio.wait_for(tts_complete.wait(), timeout=30.0)
                logger.debug("All TTS processing completed")
            except asyncio.TimeoutError:
                logger.warning("TTS processor timed out, cancelling...")
                tts_processor_task.cancel()
        else:
            # Interrupted - cancel TTS processor immediately
            logger.debug("Processing was interrupted, cancelling TTS processor")
            tts_processor_task.cancel()
            try:
                await asyncio.wait_for(tts_processor_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        
        # Only send audio_complete if not interrupted
        if not was_interrupted:
            # Signal that all audio has been sent - do this BEFORE any other operations
            await send_response({"type": "audio_complete"})
            logger.debug("Sent audio_complete signal")
        else:
            logger.debug("Skipped audio_complete signal due to interrupt")
        
        # Execute any tool calls - we need to wait for results to store in memory
        # This gives Jarvis contextual awareness of what it just did
        if tool_calls_to_execute and not was_interrupted:
            logger.info(f"Executing {len(tool_calls_to_execute)} tool calls")
            
            # Execute tools and wait for results (needed for memory context)
            # We already sent the main TTS, so this won't block user experience
            tool_results_for_memory = []
            
            async def execute_tools_with_results():
                nonlocal tool_results_for_memory
                for tool_call in tool_calls_to_execute:
                    try:
                        # Check for interruption before executing tool
                        if interrupt_handler.is_interrupted(session_id):
                            logger.info(f"Tool execution interrupted for session {session_id}")
                            break
                        
                        await send_response({
                            "type": "system_message",
                            "message": f"🔧 Executing: {tool_call.name}...",
                            "category": "github"
                        })
                        
                        # Generate verbal feedback BEFORE execution
                        action_phrases = {
                            "update_task": "Updating the task now.",
                            "add_progress_note": "Adding that note to the progress report.",
                            "create_task": "Creating the new task now.",
                            "delete_task": "Deleting the task now."
                        }
                        action_phrase = action_phrases.get(tool_call.name, "Working on that now.")
                        
                        # Send TTS for the action announcement
                        try:
                            audio_chunks = []
                            async for chunk in tts_service.synthesize_stream_async(action_phrase):
                                audio_chunks.append(chunk)
                            if audio_chunks:
                                complete_audio = b''.join(audio_chunks)
                                await send_response({"type": "audio_start", "sentence_index": 100, "size": len(complete_audio)})
                                await websocket.send_bytes(complete_audio)
                                await send_response({"type": "audio_end", "sentence_index": 100})
                        except Exception as tts_err:
                            logger.warning(f"Failed to generate action TTS: {tts_err}")
                        
                        # Execute the tool
                        result = await execute_tool(tool_call)
                        
                        # Store result for memory
                        tool_results_for_memory.append({
                            "name": result.name,
                            "success": result.success,
                            "message": result.message,
                            "data": result.data
                        })
                        
                        if result.success:
                            await send_response({
                                "type": "tool_result",
                                "success": True,
                                "name": result.name,
                                "message": result.message,
                                "data": result.data
                            })
                            await send_response({
                                "type": "system_message",
                                "message": f"✅ {result.message}",
                                "category": "github"
                            })
                            logger.info(f"Tool {result.name} succeeded: {result.message}")
                            
                            # Generate verbal feedback AFTER success
                            try:
                                success_phrase = "Done."
                                audio_chunks = []
                                async for chunk in tts_service.synthesize_stream_async(success_phrase):
                                    audio_chunks.append(chunk)
                                if audio_chunks:
                                    complete_audio = b''.join(audio_chunks)
                                    await send_response({"type": "audio_start", "sentence_index": 101, "size": len(complete_audio)})
                                    await websocket.send_bytes(complete_audio)
                                    await send_response({"type": "audio_end", "sentence_index": 101})
                            except Exception as tts_err:
                                logger.warning(f"Failed to generate success TTS: {tts_err}")
                        else:
                            await send_response({
                                "type": "tool_result",
                                "success": False,
                                "name": result.name,
                                "message": result.message
                            })
                            await send_response({
                                "type": "system_message",
                                "message": f"❌ {result.message}",
                                "category": "github"
                            })
                            logger.warning(f"Tool {result.name} failed: {result.message}")
                            
                            # Generate verbal feedback AFTER failure
                            try:
                                # Keep failure message short for voice
                                failure_phrase = f"Sorry, that didn't work."
                                audio_chunks = []
                                async for chunk in tts_service.synthesize_stream_async(failure_phrase):
                                    audio_chunks.append(chunk)
                                if audio_chunks:
                                    complete_audio = b''.join(audio_chunks)
                                    await send_response({"type": "audio_start", "sentence_index": 102, "size": len(complete_audio)})
                                    await websocket.send_bytes(complete_audio)
                                    await send_response({"type": "audio_end", "sentence_index": 102})
                            except Exception as tts_err:
                                logger.warning(f"Failed to generate failure TTS: {tts_err}")
                            
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_call.name}: {e}")
                        tool_results_for_memory.append({
                            "name": tool_call.name,
                            "success": False,
                            "message": f"Error: {str(e)}",
                            "data": None
                        })
                        await send_response({
                            "type": "tool_result",
                            "success": False,
                            "name": tool_call.name,
                            "message": f"Error: {str(e)}"
                        })
                        await send_response({
                            "type": "system_message",
                            "message": f"❌ Tool failed: {str(e)}",
                            "category": "github"
                        })
                        
                        # Generate verbal feedback for exception
                        try:
                            error_phrase = "Sorry, something went wrong."
                            audio_chunks = []
                            async for chunk in tts_service.synthesize_stream_async(error_phrase):
                                audio_chunks.append(chunk)
                            if audio_chunks:
                                complete_audio = b''.join(audio_chunks)
                                await send_response({"type": "audio_start", "sentence_index": 103, "size": len(complete_audio)})
                                await websocket.send_bytes(complete_audio)
                                await send_response({"type": "audio_end", "sentence_index": 103})
                        except Exception as tts_err:
                            logger.warning(f"Failed to generate error TTS: {tts_err}")
            
            # Wait for tool execution with timeout
            try:
                await asyncio.wait_for(execute_tools_with_results(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Tool execution timed out")
        
        # Prepare tool calls for memory (simplified format)
        tool_calls_for_memory = None
        if tool_calls_to_execute:
            tool_calls_for_memory = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in tool_calls_to_execute
            ]
        
        # Store complete response in memory (only if not interrupted and we have a response)
        if full_response.strip() and not was_interrupted:
            memory_service.add_exchange(
                session_id, 
                text, 
                full_response,
                tool_calls=tool_calls_for_memory,
                tool_results=tool_results_for_memory if tool_calls_to_execute else None
            )
            
            # Auto-save conversation to vector DB with user_id filtering
            # Check if this is a "remember" command or important information
            is_remember_command = any(word in text.lower() for word in [
                "remember", "save", "note that", "keep in mind", "don't forget",
                "x equals", "x is", "x =", "equals", "is equal to"
            ])
            
            # Auto-save if:
            # 1. It's a remember command (force save immediately)
            # 2. Or we've reached the threshold
            try:
                if is_remember_command:
                    # Force save immediately for important information
                    await memory_service.save_to_vectordb(session_id, user_id=user_id, force=True)
                    logger.info(f"Auto-saved conversation due to remember command (user: {user_id})")
                    await send_response({
                        "type": "system_message",
                        "message": "💾 Saved to memory",
                        "category": "vectordb"
                    })
                elif len(memory_service.get_history(session_id)) >= settings.conversation_summary_threshold:
                    # Auto-save when threshold is met
                    await memory_service.save_to_vectordb(session_id, user_id=user_id, force=False)
                    logger.info(f"Auto-saved conversation (threshold met, user: {user_id})")
            except Exception as e:
                logger.warning(f"Error auto-saving conversation: {e}")
                # Don't fail the whole operation if save fails
            
            # Send complete response text
            await send_response({
                "type": "response",
                "text": full_response
            })
        elif was_interrupted:
            logger.debug("Skipped memory save due to interrupt")
        
        # Note: Tool execution happens in background task (execute_tools_and_notify)
        # Audio_complete was already sent right after TTS completed
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        await send_response({
            "type": "error",
            "message": str(e)
        })


async def handle_control_message(websocket: WebSocket, session_id: str, data: dict, user_id: str = None):
    """Handle control messages from the client."""
    # Use default user_id if not provided
    if user_id is None:
        user_id = settings.default_user_id
    
    message_type = data.get("type")
    
    if message_type == "interrupt":
        # User wants to interrupt current processing
        # Set the interrupt flag - any running processing will check this and stop
        interrupt_handler.interrupt(session_id)
        # Send acknowledgment to frontend
        await websocket.send_json({"type": "interrupted"})
        # Note: Flag is NOT cleared here - it will be cleared when next audio arrives
        # This ensures any in-flight processing sees the flag and stops
        
    elif message_type == "clear_history":
        # Clear conversation history
        memory_service.clear_history(session_id)
        await websocket.send_json({"type": "history_cleared"})
        
    elif message_type == "save_to_memory":
        # Save current conversation to vector database
        try:
            success = await memory_service.save_to_vectordb(
                session_id=session_id,
                user_id=user_id,
                force=data.get("force", False)
            )
            await websocket.send_json({
                "type": "memory_saved",
                "success": success
            })
        except Exception as e:
            logger.error(f"Error saving to memory: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to save memory: {str(e)}"
            })
    
    elif message_type == "search_memory":
        # Search vector database for relevant context
        query = data.get("query", "")
        if query:
            try:
                memories = await memory_service.search_relevant_memories(query=query, user_id=user_id)
                await websocket.send_json({
                    "type": "memory_search_results",
                    "results": memories
                })
            except Exception as e:
                logger.error(f"Error searching memory: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Memory search failed: {str(e)}"
                })
        
    elif message_type == "ping":
        # Keep-alive ping
        await websocket.send_json({"type": "pong"})


# Vector Database API Endpoints

class IndexGitHubRequest(BaseModel):
    """Request model for indexing GitHub content."""
    repo: str
    path: str = ""
    file_patterns: Optional[List[str]] = None


class IndexTextRequest(BaseModel):
    """Request model for indexing text content."""
    content: str
    content_type: str
    title: Optional[str] = None
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    """Request model for vector search."""
    query: str
    collection: Optional[str] = None
    top_k: Optional[int] = None
    content_type: Optional[str] = None
    repo: Optional[str] = None


@router.post("/vectordb/index/github")
async def index_github_content(request: IndexGitHubRequest):
    """
    Index GitHub repository content into the vector database.
    
    Args:
        request: GitHub indexing request with repo, path, and file patterns
    
    Returns:
        Indexing statistics
    """
    try:
        stats = await indexing_service.index_github_directory(
            repo=request.repo,
            path=request.path,
            file_patterns=request.file_patterns
        )
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"GitHub indexing error: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.post("/vectordb/index/text")
async def index_text_content(request: IndexTextRequest):
    """
    Index arbitrary text content into the vector database.
    
    Args:
        request: Text indexing request
    
    Returns:
        Success status
    """
    try:
        success = await indexing_service.index_text_content(
            content=request.content,
            content_type=request.content_type,
            title=request.title,
            metadata=request.metadata
        )
        return {
            "status": "success" if success else "failed",
            "indexed": success
        }
    except Exception as e:
        logger.error(f"Text indexing error: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.post("/vectordb/search")
async def search_vectordb(request: SearchRequest):
    """
    Search the vector database for relevant content.
    
    Args:
        request: Search request with query and filters
    
    Returns:
        Search results
    """
    try:
        if request.collection:
            # Search specific collection
            if request.collection == "conversation_summaries":
                results = await memory_service.search_relevant_memories(
                    query=request.query,
                    top_k=request.top_k
                )
            elif request.collection == "indexed_content":
                results = await memory_service.search_indexed_content(
                    query=request.query,
                    content_type=request.content_type,
                    repo=request.repo,
                    top_k=request.top_k
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown collection: {request.collection}")
        else:
            # Search all collections
            results = await vectordb_service.search_all(
                query=request.query,
                top_k=request.top_k
            )
            # Convert to dict for JSON response
            results = [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                    "collection": r.collection
                }
                for r in results
            ]
        
        return {
            "status": "success",
            "results": results,
            "count": len(results)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/vectordb/stats")
async def get_vectordb_stats():
    """
    Get vector database statistics.
    
    Returns:
        Database statistics
    """
    try:
        stats = vectordb_service.get_stats()
        
        # Add document counts if possible
        if vectordb_service.is_available or hasattr(vectordb_service, '_in_memory_store'):
            try:
                summaries = await vectordb_service.list_documents("conversation_summaries", limit=1000)
                content = await vectordb_service.list_documents("indexed_content", limit=1000)
                stats["conversation_summaries_count"] = len(summaries)
                stats["indexed_content_count"] = len(content)
            except Exception:
                pass
        
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/vectordb/documents/{collection}")
async def list_documents(collection: str, limit: int = 100):
    """
    List documents in a vector database collection.
    
    Args:
        collection: Collection name
        limit: Maximum documents to return
    
    Returns:
        List of documents
    """
    try:
        documents = await vectordb_service.list_documents(collection, limit=limit)
        return {
            "status": "success",
            "collection": collection,
            "documents": documents,
            "count": len(documents)
        }
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.delete("/vectordb/documents/{collection}/{doc_id}")
async def delete_document(collection: str, doc_id: str):
    """
    Delete a document from the vector database.
    
    Args:
        collection: Collection name
        doc_id: Document ID
    
    Returns:
        Success status
    """
    try:
        success = await vectordb_service.delete_document(collection, doc_id)
        return {
            "status": "success" if success else "failed",
            "deleted": success
        }
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


# GitHub Write API Endpoints

class GitHubWriteRequest(BaseModel):
    """Request model for writing to GitHub."""
    repo: str
    file_path: str
    content: str
    commit_message: str
    branch: str = "main"


class GitHubUpdateTaskRequest(BaseModel):
    """Request model for updating construction tasks."""
    task_id: str
    updates: dict
    commit_message: Optional[str] = None


@router.post("/github/write")
async def write_to_github(request: GitHubWriteRequest):
    """
    Write or update a file in a GitHub repository.
    
    Args:
        request: Write request with repo, file path, content, and commit message
    
    Returns:
        Commit SHA if successful
    """
    try:
        commit_sha = github_service.create_or_update_file(
            repo_name=request.repo,
            file_path=request.file_path,
            content=request.content,
            commit_message=request.commit_message,
            branch=request.branch
        )
        
        if commit_sha:
            return {
                "status": "success",
                "commit_sha": commit_sha,
                "message": f"File {request.file_path} updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to write file")
            
    except Exception as e:
        logger.error(f"GitHub write error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to write file: {str(e)}")


@router.post("/github/update-task")
async def update_construction_task(request: GitHubUpdateTaskRequest):
    """
    Update a construction task in the tasks.json file.
    
    This endpoint reads the current tasks.json, updates the specified task,
    and commits the change back to GitHub.
    
    Args:
        request: Update request with task ID and updates
    
    Returns:
        Commit SHA if successful
    """
    try:
        repo = settings.github_default_repo or "FlankaLanka/sample-jarvis-read-write-repo"
        tasks_file = "tasks.json"
        
        # Read current tasks file
        file_content = github_service.get_file_content(repo, tasks_file)
        if not file_content:
            raise HTTPException(status_code=404, detail="tasks.json not found in repository")
        
        # Parse JSON
        import json
        tasks_data = json.loads(file_content.content)
        
        # Find and update task
        task_found = False
        for task in tasks_data.get("tasks", []):
            if task.get("id") == request.task_id:
                task.update(request.updates)
                task_found = True
                break
        
        if not task_found:
            raise HTTPException(status_code=404, detail=f"Task {request.task_id} not found")
        
        # Update last_updated timestamp
        from datetime import datetime
        tasks_data["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        # Write back to GitHub
        commit_message = request.commit_message or f"Update task {request.task_id}"
        commit_sha = github_service.create_or_update_file(
            repo_name=repo,
            file_path=tasks_file,
            content=json.dumps(tasks_data, indent=2),
            commit_message=commit_message,
            branch="main"
        )
        
        if commit_sha:
            # Auto-push after task update (GitHub API commits are automatically pushed)
            push_result = None
            try:
                push_result = github_service.auto_push_after_write(
                    commit_sha=commit_sha,
                    repo_name=repo,
                    branch="main"
                )
            except Exception as e:
                logger.warning(f"Auto-push after task update failed: {e}")
            
            return {
                "status": "success",
                "commit_sha": commit_sha,
                "task_id": request.task_id,
                "message": f"Task {request.task_id} updated successfully",
                "push_result": push_result
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update task")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update task error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update task: {str(e)}")


@router.post("/github/add-progress-note")
async def add_progress_note(note: str, repo: Optional[str] = None):
    """
    Add a progress note to the progress-report.md file.
    
    Args:
        note: Progress note to add
        repo: Repository name (defaults to configured default)
    
    Returns:
        Commit SHA if successful
    """
    try:
        repo_name = repo or settings.github_default_repo or "FlankaLanka/sample-jarvis-read-write-repo"
        progress_file = "progress-report.md"
        
        # Read current progress file
        file_content = github_service.get_file_content(repo_name, progress_file)
        if not file_content:
            raise HTTPException(status_code=404, detail="progress-report.md not found")
        
        # Add note to file
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        new_note = f"\n\n## Note - {timestamp}\n{note}\n"
        updated_content = file_content.content + new_note
        
        # Write back
        commit_message = f"Add progress note: {note[:50]}..."
        commit_sha = github_service.create_or_update_file(
            repo_name=repo_name,
            file_path=progress_file,
            content=updated_content,
            commit_message=commit_message,
            branch="main"
        )
        
        if commit_sha:
            # Auto-push after adding note (GitHub API commits are automatically pushed)
            push_result = None
            try:
                push_result = github_service.auto_push_after_write(
                    commit_sha=commit_sha,
                    repo_name=repo_name,
                    branch="main"
                )
            except Exception as e:
                logger.warning(f"Auto-push after progress note failed: {e}")
            
            return {
                "status": "success",
                "commit_sha": commit_sha,
                "message": "Progress note added successfully",
                "push_result": push_result
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add progress note")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add progress note error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add progress note: {str(e)}")


@router.post("/github/push")
async def push_to_github(
    branch: str = "main",
    remote: str = "origin",
    force: bool = False,
    repo_path: Optional[str] = None
):
    """
    Push commits to GitHub repository using git commands.
    
    Note: GitHub API commits are automatically pushed. This endpoint is for
    local git repositories that need explicit push.
    
    Args:
        branch: Branch name to push (default: main)
        remote: Remote name (default: origin)
        force: Whether to force push (default: False)
        repo_path: Optional path to local git repository
        
    Returns:
        Push result with success status
    """
    try:
        result = github_service.push_to_remote(
            repo_path=repo_path,
            branch=branch,
            remote=remote,
            force=force
        )
        return result
    except Exception as e:
        logger.error(f"Push error: {e}")
        raise HTTPException(status_code=500, detail=f"Push failed: {str(e)}")

