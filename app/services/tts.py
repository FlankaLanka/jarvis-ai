"""
Jarvis Voice Assistant - Text-to-Speech Service

Uses OpenAI TTS API for speech synthesis.
"""

from typing import AsyncGenerator, Generator, Optional

from openai import AsyncOpenAI, OpenAI
from loguru import logger

from app.config import settings


class TTSService:
    """Text-to-speech service using OpenAI TTS."""
    
    def __init__(self):
        self._client = None
        self._async_client = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy initialization of sync OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client
    
    @property
    def async_client(self) -> AsyncOpenAI:
        """Lazy initialization of async OpenAI client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._async_client
    
    def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to convert to speech
            voice: Voice selection (alloy, echo, fable, onyx, nova, shimmer)
            
        Returns:
            Audio bytes (MP3 format)
        """
        voice = voice or settings.tts_voice
        
        try:
            logger.debug(f"Synthesizing speech, text length: {len(text)}, voice: {voice}")
            
            response = self.client.audio.speech.create(
                model=settings.tts_model,
                voice=voice,
                input=text,
                response_format="mp3"
            )
            
            audio_data = response.content
            logger.debug(f"Generated audio, size: {len(audio_data)} bytes")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise
    
    def synthesize_stream(self, text: str, voice: Optional[str] = None) -> Generator[bytes, None, None]:
        """
        Synthesize text to speech with streaming.
        
        Args:
            text: Text to convert to speech
            voice: Voice selection
            
        Yields:
            Audio chunks (MP3 format)
        """
        voice = voice or settings.tts_voice
        
        try:
            logger.debug(f"Streaming speech synthesis, text length: {len(text)}")
            
            response = self.client.audio.speech.create(
                model=settings.tts_model,
                voice=voice,
                input=text,
                response_format="mp3"
            )
            
            # OpenAI TTS returns the full response, chunk it for streaming
            audio_data = response.content
            chunk_size = 4096
            
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i + chunk_size]
                
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            raise
    
    async def synthesize_async(self, text: str, voice: Optional[str] = None) -> bytes:
        """
        Synthesize text to speech asynchronously.
        
        Args:
            text: Text to convert to speech
            voice: Voice selection
            
        Returns:
            Audio bytes (MP3 format)
        """
        voice = voice or settings.tts_voice
        
        try:
            logger.debug(f"Async synthesizing speech, text length: {len(text)}")
            
            response = await self.async_client.audio.speech.create(
                model=settings.tts_model,
                voice=voice,
                input=text,
                response_format="mp3"
            )
            
            audio_data = response.content
            logger.debug(f"Generated audio, size: {len(audio_data)} bytes")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Async TTS error: {e}")
            raise
    
    async def synthesize_stream_async(
        self, 
        text: str, 
        voice: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text to speech with async streaming.
        
        Args:
            text: Text to convert to speech
            voice: Voice selection
            
        Yields:
            Audio chunks (MP3 format)
        """
        voice = voice or settings.tts_voice
        
        try:
            logger.debug(f"Async streaming speech synthesis, text length: {len(text)}")
            
            response = await self.async_client.audio.speech.create(
                model=settings.tts_model,
                voice=voice,
                input=text,
                response_format="mp3"
            )
            
            # Chunk the response for streaming
            audio_data = response.content
            chunk_size = 4096
            
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i + chunk_size]
                
        except Exception as e:
            logger.error(f"Async TTS streaming error: {e}")
            raise


# Global TTS service instance
tts_service = TTSService()

