"""
Jarvis Voice Assistant - Voice API Endpoints

Handles voice interactions including audio streaming, STT, TTS, and LLM processing.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import asyncio

from loguru import logger

from app.services.stt import stt_service
from app.services.tts import tts_service
from app.services.llm import llm_service
from app.services.memory import memory_service
from app.services.interrupt import interrupt_handler
from app.services.session import session_manager


router = APIRouter()


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
    
    try:
        # Wait for initial session setup
        init_data = await websocket.receive_json()
        session_id = init_data.get("session_id")
        
        if not session_id:
            session_id = session_manager.create_session()
        else:
            session_manager.get_or_create_session(session_id)
        
        # Send session confirmation
        await websocket.send_json({
            "type": "session_ready",
            "session_id": session_id
        })
        
        logger.info(f"Voice stream started for session: {session_id}")
        
        # Main processing loop
        while True:
            try:
                # Receive message (can be audio data or control message)
                message = await websocket.receive()
                
                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        # Audio data received
                        audio_chunk = message["bytes"]
                        await process_audio_chunk(websocket, session_id, audio_chunk)
                    elif "text" in message:
                        # Control message
                        import json
                        data = json.loads(message["text"])
                        await handle_control_message(websocket, session_id, data)
                        
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


async def process_audio_chunk(websocket: WebSocket, session_id: str, audio_chunk: bytes):
    """Process an incoming audio chunk through the STT → LLM → TTS pipeline."""
    try:
        # Check minimum audio size (roughly 0.1 seconds minimum for Whisper)
        # WebM/Opus encoding: ~1KB per 0.1 seconds is reasonable minimum
        MIN_AUDIO_SIZE = 1000  # bytes
        
        if len(audio_chunk) < MIN_AUDIO_SIZE:
            logger.debug(f"Audio chunk too small ({len(audio_chunk)} bytes), ignoring")
            await websocket.send_json({"type": "status", "status": "no_speech"})
            return
        
        # Check for interrupt
        if interrupt_handler.is_interrupted(session_id):
            interrupt_handler.clear_interrupt(session_id)
            await websocket.send_json({"type": "interrupted"})
            return
        
        # Send processing status
        await websocket.send_json({"type": "status", "status": "processing"})
        
        # Transcribe audio
        transcription = await stt_service.transcribe(audio_chunk)
        text = transcription["text"]
        
        if not text.strip():
            await websocket.send_json({"type": "status", "status": "no_speech"})
            return
        
        # Send transcription
        await websocket.send_json({
            "type": "transcription",
            "text": text
        })
        
        # Send thinking status
        await websocket.send_json({"type": "status", "status": "thinking"})
        
        # Get conversation history
        history = memory_service.get_history(session_id)
        
        # Generate LLM response
        response = await llm_service.generate_response(
            query=text,
            conversation_history=history,
            session_id=session_id
        )
        
        # Store in memory
        memory_service.add_exchange(session_id, text, response)
        
        # Send response text
        await websocket.send_json({
            "type": "response",
            "text": response
        })
        
        # Send speaking status
        await websocket.send_json({"type": "status", "status": "speaking"})
        
        # Stream TTS audio
        chunk_count = 0
        async for audio_chunk in tts_service.synthesize_stream_async(response):
            if interrupt_handler.is_interrupted(session_id):
                break
            await websocket.send_bytes(audio_chunk)
            chunk_count += 1
        
        # Signal that all audio chunks have been sent
        await websocket.send_json({"type": "audio_complete"})
        
        # Send completion status (will be handled after audio finishes playing)
        # Don't send ready immediately - let frontend handle it when audio ends
        
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })


async def handle_control_message(websocket: WebSocket, session_id: str, data: dict):
    """Handle control messages from the client."""
    message_type = data.get("type")
    
    if message_type == "interrupt":
        # User wants to interrupt current processing
        interrupt_handler.interrupt(session_id)
        await websocket.send_json({"type": "interrupted"})
        
    elif message_type == "clear_history":
        # Clear conversation history
        memory_service.clear_history(session_id)
        await websocket.send_json({"type": "history_cleared"})
        
    elif message_type == "ping":
        # Keep-alive ping
        await websocket.send_json({"type": "pong"})

