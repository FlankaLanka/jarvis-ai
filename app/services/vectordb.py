"""
Jarvis Voice Assistant - Vector Database Service

Provides semantic search capabilities using Firebase Firestore with vector embeddings.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import math
import os

from loguru import logger

from app.config import settings
from app.services.embeddings import embedding_service


@dataclass
class VectorSearchResult:
    """Represents a vector search result."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    collection: str


class VectorDBService:
    """
    Vector database service using Firebase Firestore.
    
    Features:
    - Document storage with vector embeddings
    - Semantic search using cosine similarity
    - Collection management for different content types
    - Graceful fallback when Firebase is unavailable
    """
    
    def __init__(self):
        self._db = None
        self._initialized = False
        self._firebase_available = False
    
    def _initialize(self):
        """Initialize Firebase connection."""
        if self._initialized:
            return
        
        self._initialized = True
        
        # Check if Firebase credentials are configured
        if not settings.firebase_credentials_path or not settings.firebase_project_id:
            logger.warning("Firebase not configured - vector DB will use in-memory fallback")
            self._firebase_available = False
            self._in_memory_store: Dict[str, Dict[str, Any]] = {}
            return
        
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
            
            # Check if already initialized
            try:
                firebase_admin.get_app()
            except ValueError:
                # Initialize Firebase
                cred_path = settings.firebase_credentials_path
                if not os.path.isabs(cred_path):
                    cred_path = os.path.join(os.getcwd(), cred_path)
                
                if not os.path.exists(cred_path):
                    logger.warning(f"Firebase credentials file not found: {cred_path}")
                    self._firebase_available = False
                    self._in_memory_store = {}
                    return
                
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': settings.firebase_project_id
                })
            
            self._db = firestore.client()
            self._firebase_available = True
            logger.info("Firebase Firestore initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Firebase packages not installed: {e}")
            self._firebase_available = False
            self._in_memory_store = {}
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            self._firebase_available = False
            self._in_memory_store = {}
    
    @property
    def db(self):
        """Get Firestore client with lazy initialization."""
        self._initialize()
        return self._db
    
    @property
    def is_available(self) -> bool:
        """Check if Firebase is available."""
        self._initialize()
        return self._firebase_available
    
    def _get_collection_name(self, collection: str) -> str:
        """Get full collection name with prefix."""
        return f"{settings.vector_db_collection_prefix}_{collection}"
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def add_document(
        self,
        collection: str,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Add a document with its embedding to the vector database.
        
        Args:
            collection: Collection name (e.g., 'conversation_summaries', 'indexed_content')
            doc_id: Document ID
            content: Text content to store
            metadata: Additional metadata
            embedding: Pre-computed embedding (will generate if not provided)
            
        Returns:
            True if successful
        """
        self._initialize()
        
        try:
            # Generate embedding if not provided
            if embedding is None:
                embedding = await embedding_service.generate_embedding(content)
            
            document = {
                "id": doc_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if self._firebase_available:
                collection_name = self._get_collection_name(collection)
                self.db.collection(collection_name).document(doc_id).set(document)
                logger.debug(f"Added document {doc_id} to Firebase collection {collection_name}")
            else:
                # In-memory fallback
                if collection not in self._in_memory_store:
                    self._in_memory_store[collection] = {}
                self._in_memory_store[collection][doc_id] = document
                logger.debug(f"Added document {doc_id} to in-memory store")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    async def update_document(
        self,
        collection: str,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing document.
        
        Args:
            collection: Collection name
            doc_id: Document ID
            content: New content (will regenerate embedding)
            metadata: Updated metadata
            
        Returns:
            True if successful
        """
        self._initialize()
        
        try:
            updates = {"updated_at": datetime.utcnow().isoformat()}
            
            if content is not None:
                embedding = await embedding_service.generate_embedding(content)
                updates["content"] = content
                updates["embedding"] = embedding
            
            if metadata is not None:
                updates["metadata"] = metadata
            
            if self._firebase_available:
                collection_name = self._get_collection_name(collection)
                self.db.collection(collection_name).document(doc_id).update(updates)
                logger.debug(f"Updated document {doc_id}")
            else:
                if collection in self._in_memory_store and doc_id in self._in_memory_store[collection]:
                    self._in_memory_store[collection][doc_id].update(updates)
                    logger.debug(f"Updated document {doc_id} in in-memory store")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False
    
    async def delete_document(self, collection: str, doc_id: str) -> bool:
        """
        Delete a document from the database.
        
        Args:
            collection: Collection name
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        self._initialize()
        
        try:
            if self._firebase_available:
                collection_name = self._get_collection_name(collection)
                self.db.collection(collection_name).document(doc_id).delete()
                logger.debug(f"Deleted document {doc_id}")
            else:
                if collection in self._in_memory_store and doc_id in self._in_memory_store[collection]:
                    del self._in_memory_store[collection][doc_id]
                    logger.debug(f"Deleted document {doc_id} from in-memory store")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    async def search(
        self,
        query: str,
        collection: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query text
            collection: Collection to search in
            top_k: Number of results to return (default from settings)
            threshold: Minimum similarity threshold (default from settings)
            metadata_filter: Filter by metadata fields
            
        Returns:
            List of search results sorted by relevance
        """
        self._initialize()
        
        top_k = top_k or settings.vector_search_top_k
        threshold = threshold or settings.vector_similarity_threshold
        
        try:
            # Generate query embedding
            query_embedding = await embedding_service.generate_embedding(query)
            
            results: List[VectorSearchResult] = []
            
            if self._firebase_available:
                # Query Firestore
                collection_name = self._get_collection_name(collection)
                docs = self.db.collection(collection_name).stream()
                
                for doc in docs:
                    doc_data = doc.to_dict()
                    
                    # Apply metadata filter
                    if metadata_filter:
                        skip = False
                        for key, value in metadata_filter.items():
                            if doc_data.get("metadata", {}).get(key) != value:
                                skip = True
                                break
                        if skip:
                            continue
                    
                    # Calculate similarity
                    doc_embedding = doc_data.get("embedding", [])
                    if doc_embedding:
                        similarity = self._cosine_similarity(query_embedding, doc_embedding)
                        
                        if similarity >= threshold:
                            results.append(VectorSearchResult(
                                id=doc_data.get("id", doc.id),
                                content=doc_data.get("content", ""),
                                score=similarity,
                                metadata=doc_data.get("metadata", {}),
                                collection=collection
                            ))
            else:
                # In-memory search
                if collection in self._in_memory_store:
                    for doc_id, doc_data in self._in_memory_store[collection].items():
                        # Apply metadata filter
                        if metadata_filter:
                            skip = False
                            for key, value in metadata_filter.items():
                                if doc_data.get("metadata", {}).get(key) != value:
                                    skip = True
                                    break
                            if skip:
                                continue
                        
                        doc_embedding = doc_data.get("embedding", [])
                        if doc_embedding:
                            similarity = self._cosine_similarity(query_embedding, doc_embedding)
                            
                            if similarity >= threshold:
                                results.append(VectorSearchResult(
                                    id=doc_data.get("id", doc_id),
                                    content=doc_data.get("content", ""),
                                    score=similarity,
                                    metadata=doc_data.get("metadata", {}),
                                    collection=collection
                                ))
            
            # Sort by similarity score and return top_k
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]
            
            logger.debug(f"Found {len(results)} results for query in collection {collection}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector DB: {e}")
            return []
    
    async def search_all(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[VectorSearchResult]:
        """
        Search across multiple collections.
        
        Args:
            query: Search query text
            collections: List of collections to search (default: all)
            top_k: Total number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            Combined search results from all collections
        """
        self._initialize()
        
        top_k = top_k or settings.vector_search_top_k
        
        # Default collections
        if collections is None:
            collections = ["conversation_summaries", "indexed_content"]
        
        all_results: List[VectorSearchResult] = []
        
        for collection in collections:
            results = await self.search(
                query=query,
                collection=collection,
                top_k=top_k,  # Get top_k from each collection
                threshold=threshold
            )
            all_results.extend(results)
        
        # Sort combined results and return top_k overall
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
    
    async def get_document(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.
        
        Args:
            collection: Collection name
            doc_id: Document ID
            
        Returns:
            Document data or None if not found
        """
        self._initialize()
        
        try:
            if self._firebase_available:
                collection_name = self._get_collection_name(collection)
                doc = self.db.collection(collection_name).document(doc_id).get()
                if doc.exists:
                    return doc.to_dict()
            else:
                if collection in self._in_memory_store:
                    return self._in_memory_store[collection].get(doc_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None
    
    async def list_documents(
        self,
        collection: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List documents in a collection.
        
        Args:
            collection: Collection name
            limit: Maximum number of documents to return
            
        Returns:
            List of documents
        """
        self._initialize()
        
        try:
            documents = []
            
            if self._firebase_available:
                collection_name = self._get_collection_name(collection)
                docs = self.db.collection(collection_name).limit(limit).stream()
                
                for doc in docs:
                    doc_data = doc.to_dict()
                    # Don't include embedding in list (too large)
                    doc_data.pop("embedding", None)
                    documents.append(doc_data)
            else:
                if collection in self._in_memory_store:
                    for doc_id, doc_data in list(self._in_memory_store[collection].items())[:limit]:
                        doc_copy = doc_data.copy()
                        doc_copy.pop("embedding", None)
                        documents.append(doc_copy)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics."""
        self._initialize()
        
        stats = {
            "firebase_available": self._firebase_available,
            "initialized": self._initialized
        }
        
        if not self._firebase_available:
            stats["in_memory_collections"] = list(self._in_memory_store.keys()) if hasattr(self, '_in_memory_store') else []
            stats["total_documents"] = sum(
                len(docs) for docs in self._in_memory_store.values()
            ) if hasattr(self, '_in_memory_store') else 0
        
        return stats


# Global vector database service instance
vectordb_service = VectorDBService()


