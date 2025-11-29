# Construction Repository Setup Guide

This guide explains how to populate the GitHub repository with synthetic construction site data and enable Jarvis to read and write to it.

## Overview

Jarvis can now:
- **Read** construction data from GitHub (tasks, progress, team info)
- **Write** updates to GitHub based on voice conversations
- **Index** repository content into the vector database for semantic search

## Step 1: Populate the Repository

Run the provided script to populate the repository with synthetic construction data:

```bash
python scripts/populate_construction_repo.py
```

This will upload:
- `README.md` - Repository overview
- `project-overview.md` - Project details, milestones, budget
- `tasks.json` - Construction tasks with status, assignments, progress
- `team-responsibilities.md` - Team structure and responsibilities
- `progress-report.md` - Weekly progress updates

## Step 2: Verify Repository

Check that files were uploaded:
```bash
# View repository on GitHub
open https://github.com/FlankaLanka/sample-jarvis-read-write-repo
```

## Step 3: Index Repository Content

Index the repository content into the vector database for semantic search:

```bash
curl -X POST http://localhost:8000/api/vectordb/index/github \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "FlankaLanka/sample-jarvis-read-write-repo",
    "path": "",
    "file_patterns": ["*.md", "*.json"]
  }'
```

## Step 4: Test Reading

Ask Jarvis about the construction project:

**Voice Examples:**
- "What's the status of the construction project?"
- "Who is responsible for the structural work?"
- "What tasks are in progress?"
- "What's the project completion percentage?"

Jarvis will:
1. Search the vector database for relevant content
2. Read files from GitHub if needed
3. Provide accurate answers based on the data

## Step 5: Test Writing

Jarvis can automatically update the repository based on voice commands:

**Voice Examples:**
- "Update task TASK-001 to completed"
- "Mark task TASK-002 as in progress"
- "Task TASK-003 is 75% complete"
- "Add a note: Floor 5 structural frame completed ahead of schedule"

Jarvis will:
1. Detect the write request from your voice command
2. Update the appropriate file in GitHub
3. Create a commit with a descriptive message
4. Confirm the update was successful

## API Endpoints

### Write to GitHub

```bash
# Update a construction task
curl -X POST http://localhost:8000/api/github/update-task \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "TASK-001",
    "updates": {"status": "completed", "progress": 100},
    "commit_message": "Mark task TASK-001 as completed"
  }'

# Add a progress note
curl -X POST http://localhost:8000/api/github/add-progress-note \
  -H "Content-Type: application/json" \
  -d '{"note": "Floor 5 structural frame completed successfully"}'

# Write any file
curl -X POST http://localhost:8000/api/github/write \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "FlankaLanka/sample-jarvis-read-write-repo",
    "file_path": "custom-file.md",
    "content": "# Custom Content\n\nThis is custom content.",
    "commit_message": "Add custom file"
  }'
```

## Configuration

Make sure your `.env` file includes:

```env
GITHUB_TOKEN=your_github_token_here
GITHUB_DEFAULT_REPO=FlankaLanka/sample-jarvis-read-write-repo
```

## How It Works

### Reading Flow
1. User asks a question about construction
2. Jarvis searches vector DB for relevant context
3. If needed, reads specific files from GitHub
4. Combines information to provide accurate answer

### Writing Flow
1. User requests an update (e.g., "update task TASK-001")
2. Jarvis detects the write intent
3. Reads current file from GitHub
4. Updates the data (JSON parsing for tasks.json)
5. Writes updated content back to GitHub
6. Creates commit with descriptive message
7. Confirms success to user

## Data Structure

### tasks.json
```json
{
  "tasks": [
    {
      "id": "TASK-001",
      "title": "Complete Floor 5 Structural Frame",
      "status": "in_progress",
      "progress": 65,
      "assigned_to": "John Martinez",
      "due_date": "2024-03-15"
    }
  ]
}
```

### Supported Updates
- Task status: `pending`, `in_progress`, `completed`
- Task progress: 0-100 percentage
- Task assignments: Change `assigned_to` field
- Progress notes: Append to `progress-report.md`

## Troubleshooting

**Issue:** Script fails with "GitHub token not configured"
- **Solution:** Set `GITHUB_TOKEN` in your `.env` file

**Issue:** Write operations fail
- **Solution:** Ensure your GitHub token has write permissions for the repository

**Issue:** Vector DB search returns no results
- **Solution:** Run the indexing endpoint to populate the vector database

## Next Steps

1. Populate the repository with the script
2. Index the content into vector DB
3. Test reading with voice queries
4. Test writing with voice commands
5. Customize the data structure as needed

Enjoy using Jarvis with your construction repository!


