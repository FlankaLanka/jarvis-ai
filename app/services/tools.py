"""
Jarvis Voice Assistant - Tool Definitions and Executors

Defines OpenAI function calling tools for GitHub operations.
The LLM can call these tools to perform actions like updating tasks
and adding progress notes.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from app.config import settings


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    tool_call_id: str
    name: str
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


# =============================================================================
# TOOL DEFINITIONS (OpenAI Function Calling Schema)
# =============================================================================

GITHUB_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_task",
            "description": "Update any field of a construction task. Use this when the user wants to modify a task's status, progress, assignee, priority, dates, description, or any other task attribute.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID in format TASK-XXX (e.g., TASK-001, TASK-002). If user says 'task 1', convert to TASK-001."
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "blocked"],
                        "description": "The new status for the task"
                    },
                    "progress": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Progress percentage (0-100). Set to 100 if marking as completed."
                    },
                    "assigned_to": {
                        "type": "string",
                        "description": "Person responsible for the task. Format: 'Name (Role)' e.g., 'John Martinez (Structural Lead)'"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Priority level of the task"
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Due date in YYYY-MM-DD format (e.g., '2024-03-15')"
                    },
                    "title": {
                        "type": "string",
                        "description": "The task title/name"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the task"
                    },
                    "location": {
                        "type": "string",
                        "description": "Physical location of the task (e.g., 'Floor 5', 'Parking Level 1')"
                    },
                    "estimated_hours": {
                        "type": "number",
                        "description": "Estimated hours to complete the task"
                    },
                    "actual_hours": {
                        "type": "number",
                        "description": "Actual hours spent on the task so far"
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of task IDs this task depends on (e.g., ['TASK-001', 'TASK-002'])"
                    }
                },
                "required": ["task_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_progress_note",
            "description": "Add a note to the construction project's progress report. Use this when the user wants to record progress, document something, or add a note about the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": "The content of the progress note to add"
                    }
                },
                "required": ["note"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Create a new construction task. Use this when the user wants to add a new task to the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The task title/name"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the task"
                    },
                    "assigned_to": {
                        "type": "string",
                        "description": "Person responsible for the task. Format: 'Name (Role)'"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Priority level of the task"
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Due date in YYYY-MM-DD format"
                    },
                    "location": {
                        "type": "string",
                        "description": "Physical location of the task"
                    },
                    "estimated_hours": {
                        "type": "number",
                        "description": "Estimated hours to complete the task"
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of task IDs this task depends on"
                    }
                },
                "required": ["title", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_task",
            "description": "Delete a construction task. Use this when the user wants to remove a task from the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to delete in format TASK-XXX"
                    }
                },
                "required": ["task_id"]
            }
        }
    }
]


# =============================================================================
# TOOL EXECUTORS
# =============================================================================

async def execute_tool(tool_call: ToolCall) -> ToolResult:
    """
    Execute a tool call and return the result.
    
    Args:
        tool_call: The tool call to execute
        
    Returns:
        ToolResult with success status and message
    """
    logger.info(f"Executing tool: {tool_call.name} with args: {tool_call.arguments}")
    
    try:
        if tool_call.name == "update_task":
            return await _execute_update_task(tool_call)
        elif tool_call.name == "add_progress_note":
            return await _execute_add_progress_note(tool_call)
        elif tool_call.name == "create_task":
            return await _execute_create_task(tool_call)
        elif tool_call.name == "delete_task":
            return await _execute_delete_task(tool_call)
        else:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                message=f"Unknown tool: {tool_call.name}"
            )
    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            success=False,
            message=f"Tool execution failed: {str(e)}"
        )


def _normalize_task_id(task_id: str) -> Optional[str]:
    """Normalize task ID to TASK-XXX format."""
    import re
    if not task_id:
        return None
    if not task_id.upper().startswith("TASK-"):
        digits = re.search(r'\d+', task_id)
        if digits:
            return f"TASK-{digits.group().zfill(3)}"
        return None
    return task_id.upper()


async def _execute_update_task(tool_call: ToolCall) -> ToolResult:
    """Execute the update_task tool."""
    from app.integrations.github import github_service
    
    args = tool_call.arguments
    task_id = _normalize_task_id(args.get("task_id", ""))
    
    if not task_id:
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            success=False,
            message=f"Invalid task ID: {args.get('task_id')}"
        )
    
    # Build updates dict - support all task fields
    updatable_fields = [
        "status", "progress", "assigned_to", "priority", "due_date",
        "title", "description", "location", "estimated_hours", 
        "actual_hours", "dependencies"
    ]
    
    updates = {}
    for field in updatable_fields:
        if field in args:
            updates[field] = args[field]
    
    # Auto-set status based on progress if progress is provided
    if "progress" in updates:
        if updates["progress"] == 100 and "status" not in updates:
            updates["status"] = "completed"
        elif updates["progress"] > 0 and "status" not in updates:
            updates["status"] = "in_progress"
    
    if not updates:
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            success=False,
            message="No updates specified. You can update: status, progress, assigned_to, priority, due_date, title, description, location, estimated_hours, actual_hours, dependencies"
        )
    
    try:
        repo = settings.github_default_repo or "FlankaLanka/sample-jarvis-read-write-repo"
        tasks_file = "tasks.json"
        
        # Read current tasks file
        file_content = github_service.get_file_content(repo, tasks_file)
        if not file_content:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                message="tasks.json not found in repository"
            )
        
        # Parse JSON
        tasks_data = json.loads(file_content.content)
        
        # Find and update task
        task_found = False
        for task in tasks_data.get("tasks", []):
            if task.get("id") == task_id:
                task.update(updates)
                task_found = True
                break
        
        if not task_found:
            # List available tasks for helpful error message
            available_tasks = [t.get("id") for t in tasks_data.get("tasks", [])]
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                message=f"Task {task_id} not found. Available tasks: {', '.join(available_tasks)}"
            )
        
        # Update last_updated timestamp
        tasks_data["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        # Write back to GitHub
        commit_message = f"Update task {task_id}: {updates}"
        commit_sha = github_service.create_or_update_file(
            repo_name=repo,
            file_path=tasks_file,
            content=json.dumps(tasks_data, indent=2),
            commit_message=commit_message,
            branch="main"
        )
        
        if commit_sha:
            logger.info(f"Task {task_id} updated successfully, commit: {commit_sha}")
            
            # Auto-reindex the tasks.json file in vector DB
            try:
                from app.services.indexing import indexing_service
                await indexing_service.index_github_file(
                    repo=repo,
                    path=tasks_file,
                    content=json.dumps(tasks_data, indent=2),
                    language="json"
                )
                logger.info(f"Reindexed {tasks_file} after update")
            except Exception as reindex_err:
                logger.warning(f"Failed to reindex {tasks_file} after update: {reindex_err}")
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=True,
                message=f"Task {task_id} updated successfully",
                data={"commit_sha": commit_sha, "task_id": task_id, "updates": updates}
            )
        else:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                message="Failed to commit changes to GitHub"
            )
            
    except Exception as e:
        logger.error(f"Error updating task: {e}")
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            success=False,
            message=f"Error updating task: {str(e)}"
        )


async def _execute_add_progress_note(tool_call: ToolCall) -> ToolResult:
    """Execute the add_progress_note tool."""
    from app.integrations.github import github_service
    
    args = tool_call.arguments
    note = args.get("note", "").strip()
    
    if not note:
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            success=False,
            message="No note content provided"
        )
    
    try:
        repo = settings.github_default_repo or "FlankaLanka/sample-jarvis-read-write-repo"
        progress_file = "progress-report.md"
        
        # Read current progress file
        file_content = github_service.get_file_content(repo, progress_file)
        
        # Format the note with timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        formatted_note = f"\n\n### {timestamp}\n{note}"
        
        if file_content:
            new_content = file_content.content + formatted_note
        else:
            new_content = f"# Progress Report\n{formatted_note}"
        
        # Write back to GitHub
        commit_message = f"Add progress note: {note[:50]}..."
        commit_sha = github_service.create_or_update_file(
            repo_name=repo,
            file_path=progress_file,
            content=new_content,
            commit_message=commit_message,
            branch="main"
        )
        
        if commit_sha:
            logger.info(f"Progress note added successfully, commit: {commit_sha}")
            
            # Auto-reindex the progress-report.md file in vector DB
            try:
                from app.services.indexing import indexing_service
                await indexing_service.index_github_file(
                    repo=repo,
                    path=progress_file,
                    content=new_content,
                    language="markdown"
                )
                logger.info(f"Reindexed {progress_file} after update")
            except Exception as reindex_err:
                logger.warning(f"Failed to reindex {progress_file} after update: {reindex_err}")
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=True,
                message="Progress note added successfully",
                data={"commit_sha": commit_sha, "note": note[:100]}
            )
        else:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                message="Failed to commit progress note to GitHub"
            )
            
    except Exception as e:
        logger.error(f"Error adding progress note: {e}")
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            success=False,
            message=f"Error adding progress note: {str(e)}"
        )


async def _execute_create_task(tool_call: ToolCall) -> ToolResult:
    """Execute the create_task tool."""
    from app.integrations.github import github_service
    
    args = tool_call.arguments
    title = args.get("title", "").strip()
    description = args.get("description", "").strip()
    
    if not title or not description:
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            success=False,
            message="Both title and description are required to create a task"
        )
    
    try:
        repo = settings.github_default_repo or "FlankaLanka/sample-jarvis-read-write-repo"
        tasks_file = "tasks.json"
        
        # Read current tasks file
        file_content = github_service.get_file_content(repo, tasks_file)
        if file_content:
            tasks_data = json.loads(file_content.content)
        else:
            tasks_data = {"tasks": [], "last_updated": None}
        
        # Generate new task ID
        existing_ids = [t.get("id", "") for t in tasks_data.get("tasks", [])]
        max_num = 0
        for tid in existing_ids:
            if tid.startswith("TASK-"):
                try:
                    num = int(tid.split("-")[1])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
        new_id = f"TASK-{str(max_num + 1).zfill(3)}"
        
        # Create new task with all provided fields
        new_task = {
            "id": new_id,
            "title": title,
            "description": description,
            "status": "pending",
            "priority": args.get("priority", "medium"),
            "assigned_to": args.get("assigned_to", "Unassigned"),
            "due_date": args.get("due_date", ""),
            "progress": 0,
            "dependencies": args.get("dependencies", []),
            "location": args.get("location", ""),
            "estimated_hours": args.get("estimated_hours", 0),
            "actual_hours": 0
        }
        
        tasks_data["tasks"].append(new_task)
        tasks_data["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        # Write back to GitHub
        commit_message = f"Create task {new_id}: {title}"
        commit_sha = github_service.create_or_update_file(
            repo_name=repo,
            file_path=tasks_file,
            content=json.dumps(tasks_data, indent=2),
            commit_message=commit_message,
            branch="main"
        )
        
        if commit_sha:
            logger.info(f"Task {new_id} created successfully, commit: {commit_sha}")
            
            # Auto-reindex
            try:
                from app.services.indexing import indexing_service
                await indexing_service.index_github_file(
                    repo=repo,
                    path=tasks_file,
                    content=json.dumps(tasks_data, indent=2),
                    language="json"
                )
                logger.info(f"Reindexed {tasks_file} after create")
            except Exception as reindex_err:
                logger.warning(f"Failed to reindex after create: {reindex_err}")
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=True,
                message=f"Task {new_id} created successfully: {title}",
                data={"commit_sha": commit_sha, "task_id": new_id, "task": new_task}
            )
        else:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                message="Failed to commit new task to GitHub"
            )
            
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            success=False,
            message=f"Error creating task: {str(e)}"
        )


async def _execute_delete_task(tool_call: ToolCall) -> ToolResult:
    """Execute the delete_task tool."""
    from app.integrations.github import github_service
    
    args = tool_call.arguments
    task_id = _normalize_task_id(args.get("task_id", ""))
    
    if not task_id:
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            success=False,
            message=f"Invalid task ID: {args.get('task_id')}"
        )
    
    try:
        repo = settings.github_default_repo or "FlankaLanka/sample-jarvis-read-write-repo"
        tasks_file = "tasks.json"
        
        # Read current tasks file
        file_content = github_service.get_file_content(repo, tasks_file)
        if not file_content:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                message="tasks.json not found in repository"
            )
        
        tasks_data = json.loads(file_content.content)
        
        # Find and remove task
        original_count = len(tasks_data.get("tasks", []))
        tasks_data["tasks"] = [t for t in tasks_data.get("tasks", []) if t.get("id") != task_id]
        
        if len(tasks_data["tasks"]) == original_count:
            available_tasks = [t.get("id") for t in tasks_data.get("tasks", [])]
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                message=f"Task {task_id} not found. Available tasks: {', '.join(available_tasks)}"
            )
        
        tasks_data["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        # Write back to GitHub
        commit_message = f"Delete task {task_id}"
        commit_sha = github_service.create_or_update_file(
            repo_name=repo,
            file_path=tasks_file,
            content=json.dumps(tasks_data, indent=2),
            commit_message=commit_message,
            branch="main"
        )
        
        if commit_sha:
            logger.info(f"Task {task_id} deleted successfully, commit: {commit_sha}")
            
            # Auto-reindex
            try:
                from app.services.indexing import indexing_service
                await indexing_service.index_github_file(
                    repo=repo,
                    path=tasks_file,
                    content=json.dumps(tasks_data, indent=2),
                    language="json"
                )
                logger.info(f"Reindexed {tasks_file} after delete")
            except Exception as reindex_err:
                logger.warning(f"Failed to reindex after delete: {reindex_err}")
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=True,
                message=f"Task {task_id} deleted successfully",
                data={"commit_sha": commit_sha, "task_id": task_id}
            )
        else:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                success=False,
                message="Failed to commit deletion to GitHub"
            )
            
    except Exception as e:
        logger.error(f"Error deleting task: {e}")
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            success=False,
            message=f"Error deleting task: {str(e)}"
        )


async def execute_tools(tool_calls: List[ToolCall]) -> List[ToolResult]:
    """
    Execute multiple tool calls.
    
    Args:
        tool_calls: List of tool calls to execute
        
    Returns:
        List of tool results
    """
    results = []
    for tool_call in tool_calls:
        result = await execute_tool(tool_call)
        results.append(result)
    return results

