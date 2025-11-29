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
let listeningIndicatorStop: (() => void) | null = null;
let thinkingIndicatorStop: (() => void) | null = null;
let ignoreIncomingAudio = false; // Flag to ignore audio when user is speaking
let currentRequestKey: string | null = null; // Unique key for current request - responses must match

/**
 * Generate a unique request key.
 */
function generateRequestKey(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
}

/**
 * Check if a response's request key matches the current expected key.
 * Returns true if the response should be accepted, false if rejected.
 */
function isValidRequestKey(responseKey: string | undefined): boolean {
  // If no current key is set (user interrupted), reject everything
  if (currentRequestKey === null) {
    console.log('Rejecting response: no active request key');
    return false;
  }
  
  // If response has no key, reject it (old format)
  if (!responseKey) {
    console.log('Rejecting response: missing request key in response');
    return false;
  }
  
  // Check if keys match
  if (responseKey !== currentRequestKey) {
    console.log(`Rejecting response: key mismatch (got ${responseKey}, expected ${currentRequestKey})`);
    return false;
  }
  
  return true;
}

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

  // Set up model selector change handler
  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  if (modelSelect) {
    modelSelect.addEventListener('change', () => {
      const selectedModel = modelSelect.value;
      console.log('Model changed to:', selectedModel);
      // Send model update to server if connected
      if (wsClient.isConnected) {
        wsClient.sendMessage({ type: 'set_model', model: selectedModel });
      }
    });
  }

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
    
    // Extract request_key from message if present
    const msgWithKey = message as unknown as { request_key?: string };
    const responseKey = msgWithKey.request_key;

    switch (message.type) {
      case 'status':
        // Status messages MUST have valid key to be accepted
        if (!isValidRequestKey(responseKey)) {
          console.log(`Ignoring status message with invalid key`);
          return;
        }
        handleStatusMessage(message as StatusMessage);
        break;

      case 'transcription':
        // Transcription must have valid key
        if (!isValidRequestKey(responseKey)) {
          console.log(`Ignoring transcription with invalid key`);
          return;
        }
        const transcriptionText = (message as unknown as { text: string }).text;
        console.log('Transcription:', transcriptionText);
        addChatMessage('user', transcriptionText);
        break;

      case 'response_chunk':
        // Response chunks must have valid key
        if (!isValidRequestKey(responseKey)) {
          return;
        }
        // Streaming response chunk - append to current assistant message
        const chunkText = (message as unknown as { text: string }).text;
        appendToLastAssistantMessage(chunkText);
        break;

      case 'response':
        // Complete response must have valid key
        if (!isValidRequestKey(responseKey)) {
          console.log(`Ignoring response with invalid key`);
          return;
        }
        // Complete response received - update the message
        const responseText = (message as unknown as { text: string }).text;
        console.log('Complete response:', responseText);
        updateLastAssistantMessage(responseText);
        break;

      case 'error':
        // Errors are always shown regardless of key
        console.error('Server error:', (message as unknown as { message: string }).message);
        stopListeningIndicator();
        stopThinkingIndicator();
        stopProgressChimes();
        stateManager.toError();
        break;

      case 'interrupted':
        // Interrupted acknowledgment from server
        // NOTE: Do NOT invalidate the key here - user already generated a new key
        // when they pressed the button to interrupt
        console.log('Processing interrupted (server acknowledged)');
        stopThinkingIndicator();
        stopProgressChimes();
        audioPlayback.clear();
        // Keep ignoring audio until new processing starts
        ignoreIncomingAudio = true;
        // Only go to idle if user is not currently speaking
        // If user is holding the button, stay in listening state
        if (!isHolding && !stateManager.isListening) {
          stateManager.toIdle();
        }
        break;

      case 'session_ready':
        console.log('Session ready');
        break;

      case 'pong':
        // Keep-alive response
        break;
        
      case 'audio_start':
        // Audio messages must have valid key
        if (!isValidRequestKey(responseKey)) {
          console.log(`Ignoring audio_start with invalid key`);
          return;
        }
        // Start of a new audio segment (sentence)
        const audioStartMsg = message as unknown as { sentence_index: number; size: number };
        console.log(`Audio segment ${audioStartMsg.sentence_index} starting (${audioStartMsg.size} bytes expected)`);
        break;
        
      case 'audio_end':
        // Audio messages must have valid key
        if (!isValidRequestKey(responseKey)) {
          return;
        }
        // End of an audio segment (sentence) - the audio data has already been received
        const audioEndMsg = message as unknown as { sentence_index: number };
        console.log(`Audio segment ${audioEndMsg.sentence_index} complete`);
        break;
        
      case 'audio_complete':
        // Audio complete must have valid key
        if (!isValidRequestKey(responseKey)) {
          console.log(`Ignoring audio_complete with invalid key`);
          return;
        }
        // All audio segments have been sent
        console.log('All audio segments received');
        audioPlayback.markAllReceived();
        break;
        
      case 'system_message':
        // System messages must have valid key
        if (!isValidRequestKey(responseKey)) {
          return;
        }
        // System messages for GitHub/VectorDB operations
        const systemMsg = message as unknown as { message: string; category?: string };
        addSystemMessage(systemMsg.message, systemMsg.category);
        break;
        
      case 'github_write':
        // GitHub write operation completed (legacy)
        if (!isValidRequestKey(responseKey)) {
          return;
        }
        const writeResult = message as unknown as { success: boolean; message: string };
        if (writeResult.success) {
          addSystemMessage(`✅ ${writeResult.message}`, 'github');
        } else {
          addSystemMessage(`⚠️ ${writeResult.message}`, 'github');
        }
        break;
        
      case 'tool_call':
        // LLM decided to call a tool
        if (!isValidRequestKey(responseKey)) {
          console.log(`Ignoring tool_call with invalid key`);
          return;
        }
        const toolCallMsg = message as unknown as { name: string; arguments: Record<string, unknown> };
        console.log('Tool call:', toolCallMsg.name, toolCallMsg.arguments);
        addSystemMessage(`🔧 Calling: ${toolCallMsg.name}`, 'github');
        // Play notification sound to alert user that a tool is being called
        audioCues.playNotification();
        break;
        
      case 'tool_result':
        // Tool execution completed
        if (!isValidRequestKey(responseKey)) {
          console.log(`Ignoring tool_result with invalid key`);
          return;
        }
        const toolResult = message as unknown as { success: boolean; name: string; message: string };
        console.log('Tool result:', toolResult);
        // System message already sent from server, no need to duplicate
        break;
    }
  });

  // Handle binary audio data (TTS response)
  wsClient.onBinary((data: ArrayBuffer) => {
    // CRITICAL: Ignore incoming audio when user is speaking or has interrupted
    // User must ALWAYS be able to override the AI
    if (ignoreIncomingAudio || stateManager.isListening) {
      console.log('Ignoring incoming audio - user is speaking or interrupted');
      return;
    }
    audioPlayback.queue(data);
  });

  // Set up audio playback callbacks
  audioPlayback.onEnd(() => {
    // When audio finishes playing, transition to idle if we're in speaking state
    if (stateManager.isSpeaking) {
      console.log('Audio playback ended, transitioning to idle');
      stateManager.toIdle();
    }
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
    // BUT only if user is not currently speaking (user always has priority)
    if (!stateManager.isSpeaking && !isHolding && !stateManager.isListening) {
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
      // Backend has received our NEW audio - now accept incoming audio for the NEW response
      // Only clear if user is not currently holding the button (recording)
      if (!isHolding) {
        ignoreIncomingAudio = false;
        console.log('→ Now accepting audio for new response');
      }
      stateManager.toThinking();
      startThinkingIndicator();
      startProgressChimes();
      break;

    case 'thinking':
      console.log('→ Transitioning to thinking');
      // Also clear here in case "processing" was missed
      if (!isHolding) {
        ignoreIncomingAudio = false;
      }
      stateManager.toThinking();
      startThinkingIndicator();
      break;

    case 'speaking':
      console.log('→ Transitioning to speaking');
      // Ensure we accept audio when speaking status is received
      if (!isHolding) {
        ignoreIncomingAudio = false;
      }
      stateManager.toSpeaking();
      stopThinkingIndicator();
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
      stopThinkingIndicator();
      stopProgressChimes();
      break;

    case 'no_speech':
      console.log('No speech detected');
      stopListeningIndicator();
      stopThinkingIndicator();
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

  // CRITICAL: Generate new request key IMMEDIATELY when user starts speaking
  // This ensures:
  // 1. Old responses (with old key) are rejected
  // 2. New responses (with this new key) will be accepted
  currentRequestKey = generateRequestKey();
  console.log(`Generated new request key on PTT start: ${currentRequestKey}`);
  
  // CRITICAL: Immediately ignore any incoming audio from AI
  // User must ALWAYS be able to override the AI
  ignoreIncomingAudio = true;
  
  // Stop any currently playing audio IMMEDIATELY
  audioPlayback.clear();
  
  // If currently speaking OR thinking, interrupt the backend
  if (stateManager.isSpeaking || stateManager.isThinking) {
    console.log(`Interrupting AI (was ${stateManager.state})`);
    wsClient.interrupt();
  }
  
  // Stop all indicators
  stopThinkingIndicator();
  stopProgressChimes();
  stopListeningIndicator();

  // Start recording (don't send chunks during recording)
  stateManager.toListening();
  
  // Start listening indicator audio feedback
  startListeningIndicator();
  
  microphone.startRecording(); // No callback - we'll send when released
}

/**
 * Handle push-to-talk end.
 */
async function handlePTTEnd(): Promise<void> {
  if (!isHolding) return;
  isHolding = false;

  if (!stateManager.isListening) return;
  
  // Stop listening indicator
  stopListeningIndicator();

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
    
    // Get selected model and send with audio
    const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
    const selectedModel = modelSelect?.value || 'gpt-3.5-turbo';
    
    // Use the request key that was generated when user started speaking (PTT start)
    // This key was already generated to invalidate old responses immediately
    console.log(`Sending audio with request key: ${currentRequestKey}`);
    
    // Send model preference with request key before audio
    wsClient.sendMessage({ 
      type: 'set_model', 
      model: selectedModel,
      request_key: currentRequestKey 
    });
    
    // Clear any previous assistant message tracking
    currentAssistantTextContent = null;
    
    const audioData = await audioBlob.arrayBuffer();
    wsClient.sendAudio(audioData);
    stateManager.toThinking();
    
    // Start thinking indicator and progress chimes
    startThinkingIndicator();
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
 * Start listening indicator audio feedback.
 */
function startListeningIndicator(): void {
  stopListeningIndicator();
  listeningIndicatorStop = audioCues.startListeningIndicator(1500);
}

/**
 * Stop listening indicator.
 */
function stopListeningIndicator(): void {
  if (listeningIndicatorStop) {
    listeningIndicatorStop();
    listeningIndicatorStop = null;
  }
}

/**
 * Start thinking indicator audio feedback.
 */
function startThinkingIndicator(): void {
  stopThinkingIndicator();
  thinkingIndicatorStop = audioCues.startThinkingIndicator(2000);
}

/**
 * Stop thinking indicator.
 */
function stopThinkingIndicator(): void {
  if (thinkingIndicatorStop) {
    thinkingIndicatorStop();
    thinkingIndicatorStop = null;
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

// Track current assistant message for streaming
let currentAssistantTextContent: HTMLElement | null = null;

/**
 * Add a system message to the chat log (for GitHub/VectorDB operations).
 */
function addSystemMessage(message: string, category?: string): void {
  const chatLog = document.getElementById('chat-log');
  if (!chatLog) return;
  
  // Remove empty message if present
  const emptyMsg = chatLog.querySelector('.chat-empty');
  if (emptyMsg) {
    emptyMsg.remove();
  }
  
  const messageDiv = document.createElement('div');
  messageDiv.className = 'chat-message chat-message-system';
  if (category) {
    messageDiv.setAttribute('data-category', category);
  }
  
  const textDiv = document.createElement('div');
  textDiv.className = 'chat-text';
  textDiv.textContent = message;
  
  messageDiv.appendChild(textDiv);
  chatLog.appendChild(messageDiv);
  
  // Auto-scroll to bottom
  chatLog.scrollTop = chatLog.scrollHeight;
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
  
  // Track assistant messages for streaming
  if (role === 'assistant') {
    currentAssistantTextContent = textContent;
  } else {
    currentAssistantTextContent = null;
  }
  
  // Scroll to bottom
  chatLog.scrollTop = chatLog.scrollHeight;
}

/**
 * Append text to the last assistant message (for streaming).
 */
function appendToLastAssistantMessage(text: string): void {
  if (currentAssistantTextContent) {
    currentAssistantTextContent.textContent += text;
    
    // Scroll to bottom
    const chatLog = document.getElementById('chat-log');
    if (chatLog) {
      chatLog.scrollTop = chatLog.scrollHeight;
    }
  } else {
    // No current message, create a new one
    addChatMessage('assistant', text);
  }
}

/**
 * Update the last assistant message with complete text.
 */
function updateLastAssistantMessage(text: string): void {
  if (currentAssistantTextContent) {
    currentAssistantTextContent.textContent = text;
  } else {
    // No current message, create a new one
    addChatMessage('assistant', text);
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);

