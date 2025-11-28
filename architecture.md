# Jarvis — System Architecture

```mermaid
flowchart TD

    subgraph WebApp["🌐 Web Application (Browser)"]
        A1[Microphone Input (Web Audio API)]
        A2[Speaker Output (Web Audio API)]
        A3[Push-to-Talk Button]
        A4[Audio Status Feedback System]
    end

    subgraph Server["🖥️ Jarvis Server (FastAPI - Local/AWS)"]
        subgraph AudioService["🎤 Real-Time Audio Service"]
            B1[WebRTC Gateway]
            B2[Streaming STT Engine]
            B3[Streaming TTS Engine]
        end
        
        subgraph CoreBackend["🧠 Core Backend Services"]
            C1[Request Router]
            C2[Context Memory (In-Memory/Redis)]
            C3[Zero-Hallucination Guardrails]
            C4[LLM Engine (Python Wrapper)]
            C5[Task Manager / Interrupt Handler]
        end
        
        E1[Static File Server]
    end

    subgraph Integrations["🔗 External Integrations"]
        D1[GitHub API]
        D2[Public APIs]
    end

    A1 -->|Audio Stream| B1
    B1 --> B2
    B2 -->|Transcript| C1
    C1 --> C2
    C1 --> C3
    C3 --> C4
    C4 -->|Valid Response| C5
    C5 --> B3
    B3 -->|Audio Output| A2
    C5 -->|Status Updates| A4
    A4 -->|Audio Cues| A2

    C4 -->|Data Lookup| D1
    C4 -->|API Data| D2
    
    E1 -->|Serves Frontend| WebApp
