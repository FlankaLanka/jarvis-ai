#!/usr/bin/env python3
"""
Script to index GitHub repository content into the vector database.

Usage:
    python scripts/index_repo.py [--repo REPO_NAME]
"""

import os
import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.integrations.github import github_service
from app.services.indexing import indexing_service
from app.services.vectordb import vectordb_service
from app.config import settings
from loguru import logger


DEFAULT_REPO = "FlankaLanka/sample-jarvis-read-write-repo"
FILE_PATTERNS = ["*.md", "*.json", "*.txt"]


async def index_repository(repo_name: str = None):
    """Index a GitHub repository into the vector database."""
    
    repo = repo_name or settings.github_default_repo or DEFAULT_REPO
    
    if not settings.github_token:
        logger.error("GitHub token not configured. Set GITHUB_TOKEN in .env")
        return False
    
    logger.info(f"Indexing repository: {repo}")
    logger.info(f"File patterns: {FILE_PATTERNS}")
    
    try:
        # Index the repository
        stats = await indexing_service.index_github_directory(
            repo=repo,
            path="",
            file_patterns=FILE_PATTERNS
        )
        
        logger.info(f"\n📊 Indexing Summary:")
        logger.info(f"  Total files found: {stats['total_files']}")
        logger.info(f"  ✅ Successfully indexed: {stats['indexed']}")
        logger.info(f"  ❌ Failed: {stats['failed']}")
        logger.info(f"  ⏭️ Skipped: {stats['skipped']}")
        
        # Verify indexing
        if stats['indexed'] > 0:
            logger.info("\n🔍 Verifying indexed content...")
            
            # Test search
            results = await vectordb_service.search(
                query="construction project status",
                collection="indexed_content",
                top_k=3
            )
            
            logger.info(f"  Found {len(results)} results for test query")
            for r in results:
                logger.info(f"    - Score: {r.score:.3f} | {r.metadata.get('path', 'unknown')}")
            
            # Get stats
            db_stats = vectordb_service.get_stats()
            logger.info(f"\n📈 Vector DB Stats:")
            logger.info(f"  Firebase available: {db_stats.get('firebase_available', False)}")
            
            logger.info(f"\n🎉 Repository indexed successfully!")
            return True
        else:
            logger.warning("No files were indexed")
            return False
            
    except Exception as e:
        logger.error(f"Error indexing repository: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def list_indexed_content():
    """List all indexed content in the vector database."""
    logger.info("Listing indexed content...")
    
    try:
        documents = await vectordb_service.list_documents("indexed_content", limit=100)
        
        logger.info(f"\n📚 Indexed Content ({len(documents)} documents):")
        for doc in documents:
            path = doc.get("metadata", {}).get("path", "unknown")
            repo = doc.get("metadata", {}).get("repo", "unknown")
            logger.info(f"  - {repo}/{path}")
        
        return documents
        
    except Exception as e:
        logger.error(f"Error listing content: {e}")
        return []


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Index GitHub repository into vector DB")
    parser.add_argument("--repo", help="Repository name (owner/repo)", default=None)
    parser.add_argument("--list", action="store_true", help="List indexed content")
    
    args = parser.parse_args()
    
    if args.list:
        await list_indexed_content()
    else:
        await index_repository(args.repo)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


