"""
Jarvis Voice Assistant - GitHub Integration

Provides read and write access to GitHub repositories for code search,
file reading, writing, and commit history.
Also supports git push operations for local repositories.
"""

import base64
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from github import Github, GithubException, InputGitAuthor
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
    2. File writing (create/update)
    3. Code search
    4. Commit history
    5. Repository metadata
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
    
    def create_or_update_file(
        self,
        repo_name: str,
        file_path: str,
        content: str,
        commit_message: str,
        branch: str = "main",
        author_name: Optional[str] = None,
        author_email: Optional[str] = None
    ) -> Optional[str]:
        """
        Create or update a file in a repository.
        
        Args:
            repo_name: Repository in "owner/repo" format
            file_path: Path to the file in the repository
            content: File content (string)
            commit_message: Commit message
            branch: Branch name (default: main)
            author_name: Author name for commit
            author_email: Author email for commit
            
        Returns:
            Commit SHA if successful, None otherwise
        """
        try:
            repo = self.client.get_repo(repo_name)
            
            # Try to get existing file
            try:
                existing_file = repo.get_contents(file_path, ref=branch)
                sha = existing_file.sha
                update = True
            except GithubException:
                sha = None
                update = False
            
            # Prepare commit author
            author = None
            if author_name and author_email:
                author = InputGitAuthor(
                    name=author_name,
                    email=author_email
                )
            
            # Create or update file
            if update:
                if author:
                    result = repo.update_file(
                        path=file_path,
                        message=commit_message,
                        content=content,
                        sha=sha,
                        branch=branch,
                        author=author
                    )
                else:
                    result = repo.update_file(
                        path=file_path,
                        message=commit_message,
                        content=content,
                        sha=sha,
                        branch=branch
                    )
            else:
                if author:
                    result = repo.create_file(
                        path=file_path,
                        message=commit_message,
                        content=content,
                        branch=branch,
                        author=author
                    )
                else:
                    result = repo.create_file(
                        path=file_path,
                        message=commit_message,
                        content=content,
                        branch=branch
                    )
            
            logger.info(f"{'Updated' if update else 'Created'} file {file_path} in {repo_name}")
            return result["commit"].sha
            
        except GithubException as e:
            logger.error(f"GitHub error creating/updating file: {e}")
            logger.error(f"Error details: {e.data if hasattr(e, 'data') else 'No data'}")
            logger.error(f"Error status: {e.status if hasattr(e, 'status') else 'No status'}")
            return None
        except Exception as e:
            logger.error(f"Error creating/updating file: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def delete_file(
        self,
        repo_name: str,
        file_path: str,
        commit_message: str,
        branch: str = "main",
        author_name: Optional[str] = None,
        author_email: Optional[str] = None
    ) -> Optional[str]:
        """
        Delete a file from a repository.
        
        Args:
            repo_name: Repository in "owner/repo" format
            file_path: Path to the file to delete
            commit_message: Commit message
            branch: Branch name (default: main)
            author_name: Author name for commit
            author_email: Author email for commit
            
        Returns:
            Commit SHA if successful, None otherwise
        """
        try:
            repo = self.client.get_repo(repo_name)
            file = repo.get_contents(file_path, ref=branch)
            
            # Prepare commit author
            author = None
            if author_name and author_email:
                author = InputGitAuthor(
                    name=author_name,
                    email=author_email
                )
            
            result = repo.delete_file(
                path=file_path,
                message=commit_message,
                sha=file.sha,
                branch=branch,
                author=author
            )
            
            logger.info(f"Deleted file {file_path} from {repo_name}")
            return result["commit"].sha
            
        except GithubException as e:
            logger.error(f"GitHub error deleting file: {e}")
            return None
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return None
    
    def create_file_from_local(
        self,
        repo_name: str,
        file_path: str,
        local_file_path: str,
        commit_message: str,
        branch: str = "main"
    ) -> Optional[str]:
        """
        Create a file in repository from a local file.
        
        Args:
            repo_name: Repository in "owner/repo" format
            file_path: Path in repository
            local_file_path: Path to local file
            commit_message: Commit message
            branch: Branch name
            
        Returns:
            Commit SHA if successful, None otherwise
        """
        try:
            with open(local_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.create_or_update_file(
                repo_name=repo_name,
                file_path=file_path,
                content=content,
                commit_message=commit_message,
                branch=branch
            )
        except FileNotFoundError:
            logger.error(f"Local file not found: {local_file_path}")
            return None
        except Exception as e:
            logger.error(f"Error creating file from local: {e}")
            return None
    
    def push_to_remote(
        self,
        repo_path: Optional[str] = None,
        branch: str = "main",
        remote: str = "origin",
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Push commits to remote repository using git commands.
        
        Note: GitHub API commits are automatically pushed. This method is for
        local git repositories that need explicit push.
        
        Args:
            repo_path: Path to local git repository (defaults to current directory or configured path)
            branch: Branch name to push
            remote: Remote name (default: origin)
            force: Whether to force push (default: False)
            
        Returns:
            Dictionary with success status and message
        """
        try:
            # Determine repository path
            if repo_path:
                repo_dir = Path(repo_path)
            elif hasattr(settings, 'github_local_repo_path') and settings.github_local_repo_path:
                repo_dir = Path(settings.github_local_repo_path)
            else:
                # Try to find git repo in current directory or parent
                current_dir = Path.cwd()
                repo_dir = None
                for path in [current_dir, current_dir.parent]:
                    if (path / '.git').exists():
                        repo_dir = path
                        break
                
                if not repo_dir:
                    return {
                        "success": False,
                        "message": "No local git repository found. GitHub API commits are automatically pushed."
                    }
            
            if not (repo_dir / '.git').exists():
                return {
                    "success": False,
                    "message": f"Not a git repository: {repo_dir}"
                }
            
            # Check if there are commits to push
            result = subprocess.run(
                ['git', 'rev-list', '--count', f'{remote}/{branch}..HEAD'],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            commits_ahead = int(result.stdout.strip()) if result.returncode == 0 else 0
            
            if commits_ahead == 0:
                return {
                    "success": True,
                    "message": "Repository is up to date. No commits to push.",
                    "commits_pushed": 0
                }
            
            # Push to remote
            push_cmd = ['git', 'push', remote, branch]
            if force:
                push_cmd.append('--force')
            
            result = subprocess.run(
                push_cmd,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully pushed {commits_ahead} commit(s) to {remote}/{branch}")
                return {
                    "success": True,
                    "message": f"Successfully pushed {commits_ahead} commit(s) to {remote}/{branch}",
                    "commits_pushed": commits_ahead,
                    "branch": branch,
                    "remote": remote
                }
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                logger.error(f"Git push failed: {error_msg}")
                return {
                    "success": False,
                    "message": f"Git push failed: {error_msg}",
                    "error": error_msg
                }
                
        except subprocess.TimeoutExpired:
            logger.error("Git push timed out")
            return {
                "success": False,
                "message": "Git push timed out"
            }
        except FileNotFoundError:
            logger.error("Git command not found. Make sure git is installed.")
            return {
                "success": False,
                "message": "Git command not found. Make sure git is installed."
            }
        except Exception as e:
            logger.error(f"Error pushing to remote: {e}")
            return {
                "success": False,
                "message": f"Error pushing to remote: {str(e)}"
            }
    
    def auto_push_after_write(
        self,
        commit_sha: Optional[str] = None,
        repo_name: Optional[str] = None,
        branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Automatically push after a GitHub API write operation.
        
        Note: GitHub API commits are already pushed automatically.
        This method can be used to verify or trigger additional operations.
        
        Args:
            commit_sha: Commit SHA from the write operation
            repo_name: Repository name
            branch: Branch name
            
        Returns:
            Dictionary with push status
        """
        # GitHub API commits are automatically pushed, so we just confirm
        if commit_sha:
            logger.info(f"Commit {commit_sha[:7]} automatically pushed via GitHub API")
            return {
                "success": True,
                "message": f"Commit {commit_sha[:7]} is already on remote (GitHub API auto-push)",
                "commit_sha": commit_sha,
                "method": "github_api"
            }
        
        # If no commit SHA, try local git push as fallback
        return self.push_to_remote(branch=branch)


# Global GitHub service instance
github_service = GitHubService()

