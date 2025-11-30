"""
Jarvis Voice Assistant - Indexing Service

Handles automatic indexing of conversation summaries and GitHub content
for semantic search capabilities.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib

from loguru import logger

from app.config import settings
from app.services.vectordb import vectordb_service
from app.services.embeddings import embedding_service


# Collections
CONVERSATION_SUMMARIES = "conversation_summaries"
INDEXED_CONTENT = "indexed_content"


class IndexingService:
    """
    Service for indexing content into the vector database.
    
    Features:
    - Conversation summary generation and indexing
    - GitHub content indexing (docs, README, code patterns)
    - Batch indexing operations
    - Index management
    """
    
    def __init__(self):
        self._llm_service = None  # Lazy import to avoid circular dependency
    
    @property
    def llm_service(self):
        """Lazy import of LLM service to avoid circular dependencies."""
        if self._llm_service is None:
            from app.services.llm import llm_service
            self._llm_service = llm_service
        return self._llm_service
    
    def _generate_doc_id(self, prefix: str, *parts: str) -> str:
        """Generate a unique document ID."""
        combined = "_".join(str(p) for p in parts)
        hash_suffix = hashlib.md5(combined.encode()).hexdigest()[:8]
        return f"{prefix}_{hash_suffix}"
    
    async def generate_conversation_summary(
        self,
        history: List[dict],
        session_id: str
    ) -> str:
        """
        Generate a summary of a conversation using the LLM.
        
        Args:
            history: List of conversation exchanges
            session_id: Session ID for context
            
        Returns:
            Generated summary text
        """
        if not history:
            return ""
        
        # Build conversation text
        conversation_text = "\n".join([
            f"User: {exchange.get('user', '')}\nAssistant: {exchange.get('assistant', '')}"
            for exchange in history
        ])
        
        # Use LLM to generate summary - but make it preserve specific facts
        summary_prompt = f"""Summarize this conversation, preserving ALL specific facts, values, and data mentioned.
Include exact numbers, names, values like "x = 1" or "project deadline is March 15".
Keep it to 2-4 sentences but don't lose any specific information.

Conversation:
{conversation_text}

Summary (preserve all specific values):"""
        
        try:
            summary = await self.llm_service.generate_response(
                query=summary_prompt,
                session_id=session_id
            )
            
            # Also append the raw conversation for better searchability
            # This ensures specific terms like "x equals 1" are indexed
            raw_content = " | ".join([
                f"User said: {ex.get('user', '')} Assistant said: {ex.get('assistant', '')}"
                for ex in history[-3:]  # Last 3 exchanges
            ])
            
            combined = f"{summary.strip()}\n\nKey exchanges: {raw_content}"
            return combined
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            # Fallback to raw conversation
            return f"Conversation: " + " | ".join(
                f"User: {ex.get('user', '')} Assistant: {ex.get('assistant', '')}"
                for ex in history
            )
    
    async def extract_topics(self, text: str) -> List[str]:
        """
        Extract key topics from text using LLM.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of topic strings
        """
        topic_prompt = f"""Extract 3-5 key topics or keywords from the following text. 
Return them as a comma-separated list, nothing else.

Text:
{text}

Topics:"""
        
        try:
            response = await self.llm_service.generate_response(query=topic_prompt)
            topics = [t.strip() for t in response.split(",")]
            return topics[:5]  # Limit to 5 topics
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    async def index_conversation_summary(
        self,
        session_id: str,
        history: List[dict],
        user_id: Optional[str] = None
    ) -> bool:
        """
        Index a conversation summary into the vector database.
        
        Args:
            session_id: Session ID
            history: Conversation history
            user_id: Optional user ID for filtering
            
        Returns:
            True if successful
        """
        if not history:
            logger.debug("No conversation history to index")
            return False
        
        try:
            # Generate summary
            summary = await self.generate_conversation_summary(history, session_id)
            
            if not summary:
                logger.debug("Could not generate summary")
                return False
            
            # Extract topics
            topics = await self.extract_topics(summary)
            
            # Generate document ID
            timestamp = datetime.utcnow().isoformat()
            doc_id = self._generate_doc_id("summary", session_id, timestamp)
            
            # Prepare metadata
            metadata = {
                "session_id": session_id,
                "exchanges_count": len(history),
                "topics": topics,
                "timestamp": timestamp,
                "type": "conversation_summary"
            }
            
            if user_id:
                metadata["user_id"] = user_id
            
            # Index the summary
            logger.info(f"Calling vectordb_service.add_document for summary {doc_id}")
            logger.info(f"Summary length: {len(summary)} chars, metadata: {metadata}")
            
            success = await vectordb_service.add_document(
                collection=CONVERSATION_SUMMARIES,
                doc_id=doc_id,
                content=summary,
                metadata=metadata
            )
            
            if success:
                logger.info(f"✅ Indexed conversation summary {doc_id} with {len(history)} exchanges (user_id: {user_id})")
            else:
                logger.error(f"❌ Failed to index conversation summary {doc_id} - vectordb_service.add_document returned False")
            
            return success
            
        except Exception as e:
            logger.error(f"Error indexing conversation summary: {e}", exc_info=True)
            return False
    
    async def index_github_file(
        self,
        repo: str,
        path: str,
        content: str,
        language: Optional[str] = None
    ) -> bool:
        """
        Index a GitHub file into the vector database.
        
        Args:
            repo: Repository name (owner/repo)
            path: File path in repository
            content: File content
            language: Programming language
            
        Returns:
            True if successful
        """
        try:
            # Generate document ID based on repo and path
            doc_id = self._generate_doc_id("github", repo, path)
            
            # Truncate content if too long
            max_content_length = 10000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "\n... (truncated)"
            
            # Prepare metadata
            metadata = {
                "repo": repo,
                "path": path,
                "content_type": "github_file",
                "language": language or self._detect_language(path),
                "size": len(content),
                "indexed_at": datetime.utcnow().isoformat()
            }
            
            # Index the file
            success = await vectordb_service.add_document(
                collection=INDEXED_CONTENT,
                doc_id=doc_id,
                content=content,
                metadata=metadata
            )
            
            if success:
                logger.info(f"Indexed GitHub file {repo}/{path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error indexing GitHub file: {e}")
            return False
    
    async def index_github_directory(
        self,
        repo: str,
        path: str = "",
        file_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Index all matching files in a GitHub directory.
        
        Args:
            repo: Repository name (owner/repo)
            path: Directory path (empty for root)
            file_patterns: File patterns to match (e.g., ['*.md', 'README*'])
            
        Returns:
            Statistics about indexed files
        """
        # Import GitHub service
        from app.integrations.github import github_service
        
        default_patterns = ["*.md", "README*", "*.txt", "docs/*"]
        patterns = file_patterns or default_patterns
        
        stats = {
            "total_files": 0,
            "indexed": 0,
            "failed": 0,
            "skipped": 0
        }
        
        try:
            # List directory contents
            contents = github_service.list_directory(repo, path)
            
            for item in contents:
                stats["total_files"] += 1
                
                if item["type"] == "dir":
                    # Recursively index subdirectories
                    sub_stats = await self.index_github_directory(
                        repo,
                        item["path"],
                        file_patterns
                    )
                    stats["indexed"] += sub_stats["indexed"]
                    stats["failed"] += sub_stats["failed"]
                    stats["skipped"] += sub_stats["skipped"]
                    
                elif item["type"] == "file":
                    # Check if file matches patterns
                    if not self._matches_pattern(item["name"], patterns):
                        stats["skipped"] += 1
                        continue
                    
                    # Get file content
                    file_content = github_service.get_file_content(repo, item["path"])
                    
                    if file_content:
                        success = await self.index_github_file(
                            repo=repo,
                            path=item["path"],
                            content=file_content.content
                        )
                        
                        if success:
                            stats["indexed"] += 1
                        else:
                            stats["failed"] += 1
                    else:
                        stats["failed"] += 1
            
            logger.info(f"Indexed {stats['indexed']} files from {repo}/{path}")
            return stats
            
        except Exception as e:
            logger.error(f"Error indexing GitHub directory: {e}")
            return stats
    
    async def index_text_content(
        self,
        content: str,
        content_type: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Index arbitrary text content.
        
        Args:
            content: Text content to index
            content_type: Type of content (e.g., 'documentation', 'knowledge')
            title: Optional title for the content
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            # Generate document ID
            doc_id = self._generate_doc_id("content", content_type, title or content[:50])
            
            # Prepare metadata
            doc_metadata = {
                "content_type": content_type,
                "title": title,
                "indexed_at": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            success = await vectordb_service.add_document(
                collection=INDEXED_CONTENT,
                doc_id=doc_id,
                content=content,
                metadata=doc_metadata
            )
            
            if success:
                logger.info(f"Indexed text content: {title or content[:30]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Error indexing text content: {e}")
            return False
    
    async def remove_indexed_content(
        self,
        doc_id: Optional[str] = None,
        repo: Optional[str] = None,
        path: Optional[str] = None
    ) -> bool:
        """
        Remove indexed content.
        
        Args:
            doc_id: Specific document ID to remove
            repo: Repository name (removes all files from this repo)
            path: File path (used with repo)
            
        Returns:
            True if successful
        """
        try:
            if doc_id:
                return await vectordb_service.delete_document(
                    INDEXED_CONTENT,
                    doc_id
                )
            
            if repo:
                # Generate doc_id for specific file or remove all from repo
                if path:
                    doc_id = self._generate_doc_id("github", repo, path)
                    return await vectordb_service.delete_document(
                        INDEXED_CONTENT,
                        doc_id
                    )
                else:
                    # Would need to list and delete all - for now just log
                    logger.warning("Removing all content from repo not yet implemented")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing indexed content: {e}")
            return False
    
    def _detect_language(self, path: str) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".html": "html",
            ".css": "css",
            ".txt": "text",
            ".sh": "shell",
            ".bash": "shell",
        }
        
        for ext, lang in extension_map.items():
            if path.endswith(ext):
                return lang
        
        return "unknown"
    
    def _matches_pattern(self, filename: str, patterns: List[str]) -> bool:
        """Check if filename matches any of the patterns."""
        import fnmatch
        
        for pattern in patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        
        return False


# Global indexing service instance
indexing_service = IndexingService()

