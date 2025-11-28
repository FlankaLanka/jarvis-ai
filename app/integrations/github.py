"""
Jarvis Voice Assistant - GitHub Integration

Provides read-only access to GitHub repositories for code search,
file reading, and commit history.
"""

import base64
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from github import Github, GithubException
from loguru import logger

from app.config import settings


@dataclass
class FileContent:
    """Represents a file's content from GitHub."""
    path: str
    content: str
    size: int
    sha: str


@dataclass
class CommitInfo:
    """Represents commit information."""
    sha: str
    message: str
    author: str
    date: str
    files_changed: int


@dataclass
class SearchResult:
    """Represents a code search result."""
    path: str
    repository: str
    score: float
    fragment: str


class GitHubService:
    """
    Service for GitHub API integration.
    
    Provides:
    1. File reading
    2. Code search
    3. Commit history
    4. Repository metadata
    """
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self) -> Github:
        """Lazy initialization of GitHub client."""
        if self._client is None:
            if not settings.github_token:
                raise ValueError("GitHub token not configured")
            self._client = Github(settings.github_token)
        return self._client
    
    def get_file_content(
        self,
        repo_name: str,
        file_path: str,
        ref: str = "main"
    ) -> Optional[FileContent]:
        """
        Get the content of a file from a repository.
        
        Args:
            repo_name: Repository in "owner/repo" format
            file_path: Path to the file in the repository
            ref: Branch, tag, or commit SHA
            
        Returns:
            FileContent object or None if not found
        """
        try:
            repo = self.client.get_repo(repo_name)
            file_contents = repo.get_contents(file_path, ref=ref)
            
            if isinstance(file_contents, list):
                # It's a directory, not a file
                logger.warning(f"Path {file_path} is a directory, not a file")
                return None
            
            # Decode content
            content = base64.b64decode(file_contents.content).decode("utf-8")
            
            return FileContent(
                path=file_contents.path,
                content=content,
                size=file_contents.size,
                sha=file_contents.sha
            )
            
        except GithubException as e:
            logger.error(f"GitHub error getting file: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting file content: {e}")
            return None
    
    def search_code(
        self,
        query: str,
        repo_name: Optional[str] = None,
        language: Optional[str] = None,
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Search for code across repositories.
        
        Args:
            query: Search query
            repo_name: Optional repository to limit search
            language: Optional language filter
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # Build search query
            search_query = query
            if repo_name:
                search_query += f" repo:{repo_name}"
            if language:
                search_query += f" language:{language}"
            
            logger.debug(f"Searching GitHub code: {search_query}")
            
            results = self.client.search_code(search_query)
            
            search_results = []
            for i, result in enumerate(results):
                if i >= max_results:
                    break
                
                search_results.append(SearchResult(
                    path=result.path,
                    repository=result.repository.full_name,
                    score=result.score,
                    fragment=result.text_matches[0]["fragment"] if result.text_matches else ""
                ))
            
            logger.debug(f"Found {len(search_results)} results")
            return search_results
            
        except GithubException as e:
            logger.error(f"GitHub search error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching code: {e}")
            return []
    
    def get_commit_history(
        self,
        repo_name: str,
        path: Optional[str] = None,
        max_commits: int = 10
    ) -> List[CommitInfo]:
        """
        Get commit history for a repository or file.
        
        Args:
            repo_name: Repository in "owner/repo" format
            path: Optional path to get history for specific file
            max_commits: Maximum number of commits to return
            
        Returns:
            List of commit information
        """
        try:
            repo = self.client.get_repo(repo_name)
            
            if path:
                commits = repo.get_commits(path=path)
            else:
                commits = repo.get_commits()
            
            commit_list = []
            for i, commit in enumerate(commits):
                if i >= max_commits:
                    break
                
                commit_list.append(CommitInfo(
                    sha=commit.sha[:7],
                    message=commit.commit.message.split("\n")[0][:100],
                    author=commit.commit.author.name,
                    date=commit.commit.author.date.isoformat(),
                    files_changed=len(commit.files) if commit.files else 0
                ))
            
            return commit_list
            
        except GithubException as e:
            logger.error(f"GitHub error getting commits: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting commit history: {e}")
            return []
    
    def get_repository_info(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a repository.
        
        Args:
            repo_name: Repository in "owner/repo" format
            
        Returns:
            Repository information dictionary
        """
        try:
            repo = self.client.get_repo(repo_name)
            
            return {
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description,
                "language": repo.language,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "open_issues": repo.open_issues_count,
                "default_branch": repo.default_branch,
                "updated_at": repo.updated_at.isoformat(),
            }
            
        except GithubException as e:
            logger.error(f"GitHub error getting repo info: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting repository info: {e}")
            return None
    
    def list_directory(
        self,
        repo_name: str,
        path: str = "",
        ref: str = "main"
    ) -> List[Dict[str, str]]:
        """
        List contents of a directory in a repository.
        
        Args:
            repo_name: Repository in "owner/repo" format
            path: Directory path
            ref: Branch, tag, or commit SHA
            
        Returns:
            List of file/directory information
        """
        try:
            repo = self.client.get_repo(repo_name)
            contents = repo.get_contents(path, ref=ref)
            
            if not isinstance(contents, list):
                contents = [contents]
            
            return [
                {
                    "name": item.name,
                    "path": item.path,
                    "type": item.type,
                    "size": item.size if item.type == "file" else None
                }
                for item in contents
            ]
            
        except GithubException as e:
            logger.error(f"GitHub error listing directory: {e}")
            return []
        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            return []


# Global GitHub service instance
github_service = GitHubService()

