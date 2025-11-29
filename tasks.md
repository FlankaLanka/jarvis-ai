# Jarvis — Engineering Tasks List

## P0 — Core Must-Have Tasks

### 1. Audio Layer
- [x] Implement WebRTC bidirectional audio.
- [x] Build real-time STT pipeline (Whisper/Transcribe).
- [x] Integrate low-latency TTS engine.
- [x] Build audio feedback/cue system for status communication.

### 2. LLM / Intelligence Layer
- [x] Create zero-hallucination wrapper.
- [x] Build verification layer for GitHub/API data.
- [x] Implement conversation memory (in-memory for MVP, Redis/DynamoDB for production).
- [x] Build interruptibility module.
- [x] Add follow-up context stitching logic.

### 3. Integrations
- [x] GitHub integration (search, read files, metadata).
- [x] Public API connector + refresh scheduler.
- [x] Data validator module.

### 4. Server Infrastructure
- [x] Setup FastAPI server to serve static frontend files.
- [x] Configure static file mounting (frontend build output).
- [x] Implement API endpoints under `/api/` route.
- [x] Add session store (in-memory for MVP, Redis/DynamoDB for production).
- [x] Setup local development environment.
- [ ] (Later) Setup AWS deploy pipeline (ECS/Lambda) for production.

### 5. Web Application (Frontend)
- [x] Create minimal web app (HTML/TypeScript).
- [x] Configure build pipeline to output to `static/` directory.
- [x] Implement Web Audio API for microphone input.
- [x] Implement WebRTC audio streaming to server API.
- [x] Add push-to-talk button (minimal UI, optional visual fallback).
- [x] **Build audio status feedback system** (primary method):
  - [x] Audio cues for "listening" state (e.g., brief tone or "ready").
  - [x] Audio cues for "thinking/processing" state (e.g., subtle chime).
  - [x] Audio cues for "speaking" state (natural voice response).
  - [x] Audio cues for errors or connection issues.
  - [x] Audio cues for long-running tasks (progress chimes).
- [x] Optional: Add minimal visual status indicator (fallback only, not primary).
- [x] Ensure frontend builds correctly and is served by FastAPI.

### 6. Vector Database / Persistent Memory
- [x] Setup Firebase Firestore integration.
- [x] Create embedding service with OpenAI embeddings API.
- [x] Create vector database service with semantic search.
- [x] Implement conversation summary indexing.
- [x] Implement GitHub content indexing.
- [x] Integrate vector DB context into LLM flow.
- [x] Add vector DB API endpoints for management.
- [x] Implement graceful fallback to in-memory when Firebase unavailable.

---

## P1 — Should-Have Tasks
- [x] Add system self-awareness responses (communicated via voice).
- [x] Implement comprehensive audio feedback system:
  - [x] Audio confirmation for all user actions.
  - [x] Audio notifications for system events.
  - [x] Audio error messages and recovery guidance.
- [x] Implement long-task audible chime system (progress indicators).
- [x] Implement rate limiting.

---

## P2 — Nice-to-Have Tasks
- [ ] Add passive listening mode (always-on in browser).
- [ ] Auto-open GitHub PRs for issue resolution.
- [ ] Support multi-user personalization profiles.
- [ ] Advanced vector DB features:
  - [ ] Automatic periodic indexing of GitHub repos.
  - [ ] Conversation summary scheduling (daily/weekly).
  - [ ] Vector DB analytics dashboard.
