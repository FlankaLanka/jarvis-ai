# Jarvis — Engineering Tasks List

## P0 — Core Must-Have Tasks

### 1. Audio Layer
- [ ] Implement WebRTC bidirectional audio.
- [ ] Build real-time STT pipeline (Whisper/Transcribe).
- [ ] Integrate low-latency TTS engine.
- [ ] Build audio feedback/cue system for status communication.

### 2. LLM / Intelligence Layer
- [ ] Create zero-hallucination wrapper.
- [ ] Build verification layer for GitHub/API data.
- [ ] Implement conversation memory (in-memory for MVP, Redis/DynamoDB for production).
- [ ] Build interruptibility module.
- [ ] Add follow-up context stitching logic.

### 3. Integrations
- [ ] GitHub integration (search, read files, metadata).
- [ ] Public API connector + refresh scheduler.
- [ ] Data validator module.

### 4. Server Infrastructure
- [ ] Setup FastAPI server to serve static frontend files.
- [ ] Configure static file mounting (frontend build output).
- [ ] Implement API endpoints under `/api/` route.
- [ ] Add session store (in-memory for MVP, Redis/DynamoDB for production).
- [ ] Setup local development environment.
- [ ] (Later) Setup AWS deploy pipeline (ECS/Lambda) for production.

### 5. Web Application (Frontend)
- [ ] Create minimal web app (HTML/TypeScript).
- [ ] Configure build pipeline to output to `static/` directory.
- [ ] Implement Web Audio API for microphone input.
- [ ] Implement WebRTC audio streaming to server API.
- [ ] Add push-to-talk button (minimal UI, optional visual fallback).
- [ ] **Build audio status feedback system** (primary method):
  - [ ] Audio cues for "listening" state (e.g., brief tone or "ready").
  - [ ] Audio cues for "thinking/processing" state (e.g., subtle chime).
  - [ ] Audio cues for "speaking" state (natural voice response).
  - [ ] Audio cues for errors or connection issues.
  - [ ] Audio cues for long-running tasks (progress chimes).
- [ ] Optional: Add minimal visual status indicator (fallback only, not primary).
- [ ] Ensure frontend builds correctly and is served by FastAPI.

---

## P1 — Should-Have Tasks
- [ ] Add system self-awareness responses (communicated via voice).
- [ ] Implement comprehensive audio feedback system:
  - [ ] Audio confirmation for all user actions.
  - [ ] Audio notifications for system events.
  - [ ] Audio error messages and recovery guidance.
- [ ] Implement long-task audible chime system (progress indicators).
- [ ] Implement rate limiting.

---

## P2 — Nice-to-Have Tasks
- [ ] Add passive listening mode (always-on in browser).
- [ ] Auto-open GitHub PRs for issue resolution.
- [ ] Support multi-user personalization profiles.
