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
            C2[Context Memory (In-Memory)]
            C3[Zero-Hallucination Guardrails]
            C4[LLM Engine (Python Wrapper)]
            C5[Task Manager / Interrupt Handler]
            C6[Embedding Service]
            C7[Indexing Service]
        end
        
        E1[Static File Server]
    end

    subgraph Storage["💾 Persistent Storage"]
        F1[Firebase Firestore]
        F2[Vector DB Collections]
        F3[Conversation Summaries]
        F4[Indexed Content]
    end

    subgraph Integrations["🔗 External Integrations"]
        D1[GitHub API]
        D2[Public APIs]
        D3[OpenAI Embeddings API]
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
    
    C6 -->|Generate Embeddings| D3
    C7 -->|Index Content| F2
    
    C4 -->|Semantic Search| F2
    F2 --> F3
    F2 --> F4
    F1 --> F2
    
    D1 -->|File Content| C7
    
    E1 -->|Serves Frontend| WebApp
```

## Component Details

### Vector Database Layer

The system now includes a persistent vector database using Firebase Firestore for semantic search capabilities:

**Embedding Service** (`app/services/embeddings.py`)
- Generates vector embeddings using OpenAI's `text-embedding-3-small` model
- Implements LRU caching for frequently accessed text
- Batch embedding generation for efficiency

**VectorDB Service** (`app/services/vectordb.py`)
- Firebase Firestore integration with vector search
- Document storage with embeddings
- Cosine similarity-based semantic search
- Graceful fallback to in-memory storage when Firebase unavailable

**Indexing Service** (`app/services/indexing.py`)
- Automatic conversation summary generation
- GitHub content indexing (docs, README, code patterns)
- Topic extraction from conversations
- Index management (add, update, delete)

### Data Collections

**conversation_summaries**
- Stores summarized conversation history
- Enables semantic search of past conversations
- Indexed after conversation threshold (default: 10 exchanges)

**indexed_content**
- Stores indexed GitHub files and documentation
- Supports multiple content types (github_file, documentation, knowledge)
- Filtered by repo, path, and content type

### Query Flow with Vector DB

1. User speaks → STT transcription
2. Query vector DB for relevant context:
   - Search conversation_summaries for past discussions
   - Search indexed_content for relevant docs/code
3. Combine context with conversation history
4. LLM generates response with full context
5. TTS speaks response
6. Conversation saved to in-memory + optionally indexed to vector DB
