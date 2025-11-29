#!/usr/bin/env python3
"""
Script to populate GitHub repository with synthetic construction site data.

Usage:
    python scripts/populate_construction_repo.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.integrations.github import github_service
from app.config import settings
from loguru import logger


REPO_NAME = "FlankaLanka/sample-jarvis-read-write-repo"
DATA_DIR = Path(__file__).parent.parent / "construction-data"


def populate_repo():
    """Populate the GitHub repository with construction data files."""
    
    if not settings.github_token:
        logger.error("GitHub token not configured. Set GITHUB_TOKEN in .env")
        return False
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        return False
    
    logger.info(f"Populating repository {REPO_NAME} with construction data...")
    
    files_to_upload = [
        "README.md",
        "project-overview.md",
        "tasks.json",
        "team-responsibilities.md",
        "progress-report.md"
    ]
    
    success_count = 0
    error_count = 0
    
    for filename in files_to_upload:
        local_path = DATA_DIR / filename
        
        if not local_path.exists():
            logger.warning(f"File not found: {local_path}, skipping...")
            continue
        
        try:
            commit_message = f"Add {filename} - Construction site data"
            
            commit_sha = github_service.create_file_from_local(
                repo_name=REPO_NAME,
                file_path=filename,
                local_file_path=str(local_path),
                commit_message=commit_message,
                branch="main"
            )
            
            if commit_sha:
                logger.info(f"✅ Uploaded {filename} (commit: {commit_sha[:7]})")
                success_count += 1
            else:
                logger.error(f"❌ Failed to upload {filename}")
                error_count += 1
                
        except Exception as e:
            logger.error(f"❌ Error uploading {filename}: {e}")
            error_count += 1
    
    logger.info(f"\n📊 Summary:")
    logger.info(f"  ✅ Successfully uploaded: {success_count} files")
    logger.info(f"  ❌ Failed: {error_count} files")
    
    if success_count > 0:
        logger.info(f"\n🎉 Repository populated! View at: https://github.com/{REPO_NAME}")
    
    return success_count > 0


if __name__ == "__main__":
    try:
        success = populate_repo()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


