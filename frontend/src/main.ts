/**
 * Jarvis Voice Assistant - Main Application Entry Point
 */

import { microphone } from './audio/microphone';
import { audioPlayback } from './audio/playback';
import { audioCues } from './audio/cues';
import { wsClient } from './webrtc/client';
import { stateManager, updateUI } from './state';
import type { WSMessage, StatusMessage } from './types';

// Application state
let isHolding = false;
let progressChimeStop: (() => void) | null = null;

/**
 * Initialize the application.
 */
async function init(): Promise<void> {
  console.log('Initializing Jarvis...');

  // Check browser support
  if (!microphone.constructor.prototype.constructor.name) {
    if (!('mediaDevices' in navigator)) {
      showError('Your browser does not support audio recording.');
      return;
    }
  }

  // Set up state change handler
  stateManager.onStateChange((newState) => {
    updateUI(newState);
  });

  // Set up WebSocket handlers
  setupWebSocketHandlers();

  // Set up UI event handlers
  setupUIHandlers();

  // Request microphone permission
  const hasPermission = await microphone.requestPermission();
  if (!hasPermission) {
    showError('Microphone permission is required.');
    return;
  }

  // Connect to server
  try {
    await wsClient.connect();
    console.log('Connected to server');
  } catch (error) {
    console.error('Failed to connect:', error);
    showError('Failed to connect to server.');
  }

  console.log('Jarvis ready');
}

/**
 * Set up WebSocket message handlers.
 */
function setupWebSocketHandlers(): void {
  // Handle JSON messages
  wsClient.onMessage((message: WSMessage) => {
    console.log('Received:', message.type);

    switch (message.type) {
      case 'status':
        handleStatusMessage(message as StatusMessage);
        break;

      case 'transcription':
        const transcriptionText = (message as unknown as { text: string }).text;
        console.log('Transcription:', transcriptionText);
        addChatMessage('user', transcriptionText);
        break;

      case 'response':
        const responseText = (message as unknown as { text: string }).text;
        console.log('Response:', responseText);
        addChatMessage('assistant', responseText);
        break;

      case 'error':
        console.error('Server error:', (message as unknown as { message: string }).message);
        stateManager.toError();
        stopProgressChimes();
        break;

      case 'interrupted':
        console.log('Processing interrupted');
        audioPlayback.clear();
        stateManager.toIdle();
        stopProgressChimes();
        break;

      case 'session_ready':
        console.log('Session ready');
        break;

      case 'pong':
        // Keep-alive response
        break;
        
      case 'audio_complete':
        // All audio chunks have been sent, flush buffer to start playing
        console.log('All audio chunks received, flushing buffer');
        audioPlayback.flush();
        break;
    }
  });

  // Handle binary audio data (TTS response)
  wsClient.onBinary((data: ArrayBuffer) => {
    audioPlayback.queue(data);
  });

  // Handle connection events
  wsClient.onConnect(() => {
    audioCues.playNotification();
  });

  wsClient.onDisconnect(() => {
    stateManager.toError();
    audioCues.playError();
  });

  wsClient.onError(() => {
    stateManager.toError();
  });

  // Handle playback start
  audioPlayback.onStart(() => {
    // Ensure we're in speaking state when audio actually starts
    if (!stateManager.isSpeaking) {
      stateManager.toSpeaking();
    }
  });

  // Handle playback end
  audioPlayback.onEnd(() => {
    // Transition back to idle when audio finishes
    if (stateManager.isSpeaking) {
      stateManager.toIdle();
    }
  });
}

/**
 * Handle status messages from server.
 */
function handleStatusMessage(message: StatusMessage): void {
  console.log('Status update:', message.status);
  
  switch (message.status) {
    case 'processing':
      console.log('→ Transitioning to thinking (processing)');
      stateManager.toThinking();
      startProgressChimes();
      break;

    case 'thinking':
      console.log('→ Transitioning to thinking');
      stateManager.toThinking();
      break;

    case 'speaking':
      console.log('→ Transitioning to speaking');
      stateManager.toSpeaking();
      stopProgressChimes();
      // Don't flush here - wait for audio_complete message
      break;

    case 'ready':
      console.log('→ Ready status received, isPlaying:', audioPlayback.isPlaying);
      // Only go to idle if we're not currently playing audio
      if (!audioPlayback.isPlaying) {
        stateManager.toIdle();
      } else {
        console.log('→ Audio still playing, keeping speaking state');
      }
      stopProgressChimes();
      break;

    case 'no_speech':
      console.log('No speech detected');
      stateManager.toIdle();
      stopProgressChimes();
      break;
  }
}

/**
 * Set up UI event handlers.
 */
function setupUIHandlers(): void {
  const button = document.getElementById('ptt-button');
  if (!button) return;

  // Handle mouse/touch events for push-to-talk
  // Use pointer events for better cross-device support
  button.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    button.setPointerCapture(e.pointerId);
    handlePTTStart();
  });
  
  button.addEventListener('pointerup', (e) => {
    e.preventDefault();
    handlePTTEnd();
  });
  
  button.addEventListener('pointercancel', (e) => {
    e.preventDefault();
    handlePTTEnd();
  });
  
  // Fallback for older browsers
  button.addEventListener('mousedown', (e) => {
    e.preventDefault();
    handlePTTStart();
  });
  
  button.addEventListener('mouseup', (e) => {
    e.preventDefault();
    handlePTTEnd();
  });
  
  // Only end on mouseleave if button was pressed
  let mousePressed = false;
  button.addEventListener('mousedown', () => { mousePressed = true; });
  button.addEventListener('mouseup', () => { mousePressed = false; });
  button.addEventListener('mouseleave', (e) => {
    if (mousePressed) {
      e.preventDefault();
      handlePTTEnd();
      mousePressed = false;
    }
  });
  
  // Touch events
  button.addEventListener('touchstart', (e) => {
    e.preventDefault();
    handlePTTStart();
  });
  
  button.addEventListener('touchend', (e) => {
    e.preventDefault();
    handlePTTEnd();
  });
  
  button.addEventListener('touchcancel', (e) => {
    e.preventDefault();
    handlePTTEnd();
  });

  // Handle keyboard events
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && !e.repeat) {
      e.preventDefault();
      handlePTTStart();
    }
  });

  document.addEventListener('keyup', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      handlePTTEnd();
    }
  });

  // Resume audio context on first interaction
  const resumeAudio = async () => {
    await audioCues.resume();
    await audioPlayback.resume();
    document.removeEventListener('click', resumeAudio);
    document.removeEventListener('touchstart', resumeAudio);
  };

  document.addEventListener('click', resumeAudio);
  document.addEventListener('touchstart', resumeAudio);
}

/**
 * Handle push-to-talk start.
 */
function handlePTTStart(): void {
  if (isHolding) return;
  isHolding = true;

  // If currently speaking, interrupt
  if (stateManager.isSpeaking) {
    wsClient.interrupt();
    audioPlayback.clear();
  }

  // Start recording (don't send chunks during recording)
  stateManager.toListening();
  microphone.startRecording(); // No callback - we'll send when released
}

/**
 * Handle push-to-talk end.
 */
async function handlePTTEnd(): Promise<void> {
  if (!isHolding) return;
  isHolding = false;

  if (!stateManager.isListening) return;

  // Stop recording and send complete audio
  const audioBlob = await microphone.stopRecording();
  
  if (audioBlob && audioBlob.size > 0) {
    // Check minimum size (roughly 0.1 seconds at 16kHz mono)
    // WebM with Opus encoding: ~1KB per 0.1 seconds is reasonable
    const minSize = 1000; // bytes
    
    if (audioBlob.size < minSize) {
      console.log(`Audio too short (${audioBlob.size} bytes), ignoring`);
      stateManager.toIdle();
      return;
    }
    
    const audioData = await audioBlob.arrayBuffer();
    wsClient.sendAudio(audioData);
    stateManager.toThinking();
    startProgressChimes();
  } else {
    stateManager.toIdle();
  }
}

/**
 * Start progress chimes for long-running tasks.
 */
function startProgressChimes(): void {
  stopProgressChimes();
  
  // Start chimes after 3 seconds
  const timeout = setTimeout(() => {
    progressChimeStop = audioCues.startProgressChimes(2000);
  }, 3000);

  progressChimeStop = () => {
    clearTimeout(timeout);
    if (progressChimeStop) {
      // This will be replaced by the actual stop function
    }
  };
}

/**
 * Stop progress chimes.
 */
function stopProgressChimes(): void {
  if (progressChimeStop) {
    progressChimeStop();
    progressChimeStop = null;
  }
}

/**
 * Show error message to user.
 */
function showError(message: string): void {
  console.error(message);
  stateManager.toError();
  
  const statusEl = document.getElementById('status');
  if (statusEl) {
    statusEl.textContent = message;
  }
}

/**
 * Add a message to the chat log.
 */
function addChatMessage(role: 'user' | 'assistant', text: string): void {
  const chatLog = document.getElementById('chat-log');
  if (!chatLog) return;
  
  // Remove empty message if present
  const emptyMsg = chatLog.querySelector('.chat-empty');
  if (emptyMsg) {
    emptyMsg.remove();
  }
  
  const messageDiv = document.createElement('div');
  messageDiv.className = `chat-message chat-message-${role}`;
  
  const roleLabel = document.createElement('div');
  roleLabel.className = 'chat-role';
  roleLabel.textContent = role === 'user' ? 'You' : 'Jarvis';
  
  const textContent = document.createElement('div');
  textContent.className = 'chat-text';
  textContent.textContent = text;
  
  messageDiv.appendChild(roleLabel);
  messageDiv.appendChild(textContent);
  
  chatLog.appendChild(messageDiv);
  
  // Scroll to bottom
  chatLog.scrollTop = chatLog.scrollHeight;
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);

