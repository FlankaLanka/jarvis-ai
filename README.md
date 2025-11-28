# Jarvis Voice Assistant

Real-time, audio-first voice assistant built for frontline workers.

## Features

- **Voice-Only Interaction**: Push-to-talk interface with audio feedback for all status updates
- **Real-Time Processing**: Sub-500ms latency target with streaming STT/TTS
- **Context Memory**: Maintains conversation context across interactions
- **Interruptible**: Users can interrupt at any time during responses
- **Zero-Hallucination**: Verified responses with guardrails against AI hallucinations
- **GitHub Integration**: Search code, read files, view commit history
- **Audio Feedback**: All status communicated via audio cues (designed for hardware deployment)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (Frontend)                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Microphone │  │ Audio Cues   │  │  Audio Playback  │   │
│  │  (Web Audio)│  │ (State Tones)│  │  (TTS Response)  │   │
│  └──────┬──────┘  └──────────────┘  └────────▲─────────┘   │
│         │                                     │             │
│         └──────────── WebSocket ──────────────┘             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  FastAPI Server (Python)                     │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │ STT Service│  │ LLM Service │  │ TTS Service          │ │
│  │ (Whisper)  │→ │ (GPT-4)     │→ │ (OpenAI TTS)         │ │
│  └────────────┘  └──────┬──────┘  └──────────────────────┘ │
│                         │                                   │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │  Memory │ Guardrails │ Context │ GitHub │ Interrupt │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- OpenAI API key

### Setup

1. **Clone and configure**:
   ```bash
   cd jarvis-ai
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

2. **Install backend dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Build frontend**:
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

4. **Run the server**:
   ```bash
   python main.py
   ```

5. **Open browser**: Navigate to `http://localhost:8000`

### Development Mode

For development with hot reload:

**Terminal 1 - Backend**:
```bash
python main.py  # Runs with reload enabled when DEBUG=true
```

**Terminal 2 - Frontend** (optional, for frontend dev):
```bash
cd frontend
npm run dev  # Runs on port 3000, proxies API to 8000
```

## Configuration

Environment variables (`.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `GITHUB_TOKEN` | GitHub access token (optional) | - |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `DEBUG` | Enable debug mode | `true` |
| `TTS_VOICE` | TTS voice (alloy/echo/fable/onyx/nova/shimmer) | `alloy` |
| `LLM_MODEL` | LLM model | `gpt-4` |

## Usage

### Push-to-Talk

1. **Click and hold** the microphone button (or hold **Space** key)
2. **Speak** your question
3. **Release** when done
4. Listen for the response

### Audio Feedback

All status is communicated via audio:
- **Ready**: Brief descending tone
- **Listening**: Ascending tone
- **Thinking**: Subtle chime
- **Speaking**: Natural voice response
- **Error**: Low warning tone
- **Long tasks**: Periodic progress chimes

### Interruption

You can interrupt Jarvis at any time:
- Press the button while Jarvis is speaking
- Press **Space** key
- Jarvis will stop and listen to your new input

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/voice/stream` | WebSocket | Voice streaming |
| `/api/voice/transcribe` | POST | Transcribe audio |
| `/api/voice/synthesize` | POST | Text-to-speech |
| `/api/voice/query` | POST | Text query (testing) |
| `/api/session/create` | POST | Create session |
| `/api/session/{id}` | GET | Get session info |

## Project Structure

```
jarvis-ai/
├── frontend/               # TypeScript frontend
│   ├── src/
│   │   ├── audio/         # Audio handling (mic, playback, cues)
│   │   ├── webrtc/        # WebSocket client
│   │   ├── types/         # TypeScript types
│   │   ├── state.ts       # State management
│   │   └── main.ts        # Entry point
│   └── index.html         # Main HTML
├── app/                    # Python backend
│   ├── api/               # API endpoints
│   ├── services/          # Core services (STT, TTS, LLM, etc.)
│   ├── integrations/      # External integrations (GitHub, APIs)
│   └── middleware/        # Middleware (rate limiting)
├── static/                 # Built frontend (generated)
├── main.py                 # FastAPI entry point
├── requirements.txt        # Python dependencies
└── .env.example           # Environment template
```

## License

Proprietary - Frontier Audio

