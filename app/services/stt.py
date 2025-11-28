"""
Jarvis Voice Assistant - Speech-to-Text Service

Uses OpenAI Whisper API for real-time transcription.
"""

import io
from typing import Dict, Any

from openai import AsyncOpenAI
from loguru import logger

from app.config import settings


class STTService:
    """Speech-to-text service using OpenAI Whisper."""
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client
    
    async def transcribe(self, audio_data: bytes, language: str = "en") -> Dict[str, Any]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw audio bytes (WAV, MP3, etc.)
            language: Language code for transcription
            
        Returns:
            Dictionary with transcription result
        """
        try:
            # Create a file-like object from the audio bytes
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav"  # OpenAI needs a filename
            
            logger.debug(f"Transcribing audio, size: {len(audio_data)} bytes")
            
            response = await self.client.audio.transcriptions.create(
                model=settings.stt_model,
                file=audio_file,
                language=language,
                response_format="verbose_json"
            )
            
            result = {
                "text": response.text,
                "language": getattr(response, "language", language),
                "duration": getattr(response, "duration", None),
                "confidence": 1.0,  # Whisper doesn't provide confidence scores
            }
            
            logger.debug(f"Transcription result: {result['text'][:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    async def transcribe_stream(self, audio_chunks):
        """
        Transcribe streaming audio chunks.
        
        Note: Whisper API doesn't support true streaming, so this
        buffers chunks and transcribes when sufficient audio is collected.
        
        Args:
            audio_chunks: Async generator of audio chunk bytes
            
        Yields:
            Partial transcription results
        """
        buffer = bytearray()
        min_buffer_size = 32000  # ~2 seconds at 16kHz mono
        
        async for chunk in audio_chunks:
            buffer.extend(chunk)
            
            if len(buffer) >= min_buffer_size:
                result = await self.transcribe(bytes(buffer))
                yield result
                buffer.clear()
        
        # Process remaining buffer
        if buffer:
            result = await self.transcribe(bytes(buffer))
            yield result


# Global STT service instance
stt_service = STTService()

