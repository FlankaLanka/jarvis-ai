"""
Jarvis Voice Assistant - Embedding Service

Generates vector embeddings using OpenAI's embedding API for semantic search.
"""

from typing import List, Optional, Dict
import hashlib

from openai import AsyncOpenAI
from loguru import logger

from app.config import settings


class EmbeddingService:
    """
    Service for generating vector embeddings using OpenAI's API.
    
    Features:
    - Async embedding generation
    - Batch processing for multiple texts
    - Simple in-memory cache for frequently accessed text
    """
    
    def __init__(self, cache_size: int = 1000):
        """
        Initialize the embedding service.
        
        Args:
            cache_size: Maximum number of embeddings to cache
        """
        self._client: Optional[AsyncOpenAI] = None
        self._cache: Dict[str, List[float]] = {}
        self._cache_size = cache_size
        self._cache_keys: List[str] = []  # Track insertion order for LRU
    
    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self._client
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _add_to_cache(self, text: str, embedding: List[float]):
        """Add embedding to cache with LRU eviction."""
        key = self._get_cache_key(text)
        
        # Evict oldest if at capacity
        if len(self._cache) >= self._cache_size and key not in self._cache:
            oldest_key = self._cache_keys.pop(0)
            del self._cache[oldest_key]
        
        # Add to cache
        if key not in self._cache:
            self._cache_keys.append(key)
        self._cache[key] = embedding
    
    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if available."""
        key = self._get_cache_key(text)
        return self._cache.get(key)
    
    async def generate_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: The text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Vector embedding as a list of floats
        """
        # Check cache first
        if use_cache:
            cached = self._get_from_cache(text)
            if cached is not None:
                logger.debug("Using cached embedding")
                return cached
        
        try:
            # Truncate text if too long (OpenAI has 8191 token limit for embeddings)
            # Rough estimate: 1 token ≈ 4 chars
            max_chars = 30000  # Conservative limit
            if len(text) > max_chars:
                text = text[:max_chars]
                logger.warning(f"Text truncated to {max_chars} characters for embedding")
            
            response = await self.client.embeddings.create(
                model=settings.embedding_model,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Cache the result
            if use_cache:
                self._add_to_cache(text, embedding)
            
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            List of vector embeddings
        """
        if not texts:
            return []
        
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        texts_to_embed: List[tuple] = []  # (index, text)
        
        # Check cache first
        if use_cache:
            for i, text in enumerate(texts):
                cached = self._get_from_cache(text)
                if cached is not None:
                    embeddings[i] = cached
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = [(i, text) for i, text in enumerate(texts)]
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            try:
                # Truncate texts if needed
                max_chars = 30000
                processed_texts = []
                for _, text in texts_to_embed:
                    if len(text) > max_chars:
                        processed_texts.append(text[:max_chars])
                    else:
                        processed_texts.append(text)
                
                response = await self.client.embeddings.create(
                    model=settings.embedding_model,
                    input=processed_texts
                )
                
                # Map results back to original indices
                for (orig_idx, orig_text), embedding_data in zip(texts_to_embed, response.data):
                    embedding = embedding_data.embedding
                    embeddings[orig_idx] = embedding
                    
                    if use_cache:
                        self._add_to_cache(orig_text, embedding)
                
                logger.debug(f"Generated {len(texts_to_embed)} embeddings in batch")
                
            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                raise
        
        return embeddings
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_keys.clear()
        logger.debug("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._cache_size
        }


# Global embedding service instance
embedding_service = EmbeddingService()


