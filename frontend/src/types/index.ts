/**
 * Jarvis Voice Assistant - TypeScript Types
 */

// Application states
export type AppState = 'idle' | 'listening' | 'thinking' | 'speaking' | 'error';

// WebSocket message types
export interface WSMessage {
  type: string;
  [key: string]: unknown;
}

export interface SessionReadyMessage extends WSMessage {
  type: 'session_ready';
  session_id: string;
}

export interface StatusMessage extends WSMessage {
  type: 'status';
  status: 'processing' | 'thinking' | 'speaking' | 'ready' | 'no_speech';
}

export interface TranscriptionMessage extends WSMessage {
  type: 'transcription';
  text: string;
}

export interface ResponseMessage extends WSMessage {
  type: 'response';
  text: string;
}

export interface ErrorMessage extends WSMessage {
  type: 'error';
  message: string;
}

// Audio cue types
export type AudioCueType = 'ready' | 'listening' | 'thinking' | 'speaking' | 'error' | 'progress';

// Event handlers
export interface StateChangeHandler {
  (newState: AppState, oldState: AppState): void;
}

export interface TranscriptionHandler {
  (text: string): void;
}

export interface ResponseHandler {
  (text: string): void;
}

export interface ErrorHandler {
  (error: Error | string): void;
}

// Configuration
export interface JarvisConfig {
  wsUrl?: string;
  apiUrl?: string;
  enableVisualFeedback?: boolean;
  enableAudioFeedback?: boolean;
}

// Session info
export interface SessionInfo {
  session_id: string;
  created_at: string;
  expires_at: string;
  message_count: number;
}

