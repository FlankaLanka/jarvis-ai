/**
 * Jarvis Voice Assistant - State Management
 * 
 * Manages application state with event-driven updates.
 */

import type { AppState, StateChangeHandler } from './types';
import { audioCues } from './audio/cues';

class StateManager {
  private _state: AppState = 'idle';
  private _listeners: StateChangeHandler[] = [];
  private _audioFeedbackEnabled: boolean = true;

  get state(): AppState {
    return this._state;
  }

  get isIdle(): boolean {
    return this._state === 'idle';
  }

  get isListening(): boolean {
    return this._state === 'listening';
  }

  get isThinking(): boolean {
    return this._state === 'thinking';
  }

  get isSpeaking(): boolean {
    return this._state === 'speaking';
  }

  get isError(): boolean {
    return this._state === 'error';
  }

  /**
   * Enable or disable audio feedback for state changes.
   */
  setAudioFeedback(enabled: boolean): void {
    this._audioFeedbackEnabled = enabled;
  }

  /**
   * Transition to a new state.
   */
  setState(newState: AppState): void {
    if (newState === this._state) return;

    const oldState = this._state;
    this._state = newState;

    // Play audio cue for state transition
    if (this._audioFeedbackEnabled) {
      this._playStateCue(newState);
    }

    // Notify listeners
    this._listeners.forEach(handler => {
      try {
        handler(newState, oldState);
      } catch (e) {
        console.error('State change handler error:', e);
      }
    });

    console.log(`State: ${oldState} → ${newState}`);
  }

  /**
   * Subscribe to state changes.
   */
  onStateChange(handler: StateChangeHandler): () => void {
    this._listeners.push(handler);
    
    // Return unsubscribe function
    return () => {
      const index = this._listeners.indexOf(handler);
      if (index > -1) {
        this._listeners.splice(index, 1);
      }
    };
  }

  /**
   * Transition to idle state.
   */
  toIdle(): void {
    this.setState('idle');
  }

  /**
   * Transition to listening state.
   */
  toListening(): void {
    this.setState('listening');
  }

  /**
   * Transition to thinking state.
   */
  toThinking(): void {
    this.setState('thinking');
  }

  /**
   * Transition to speaking state.
   */
  toSpeaking(): void {
    this.setState('speaking');
  }

  /**
   * Transition to error state.
   */
  toError(): void {
    this.setState('error');
  }

  /**
   * Play audio cue for state.
   */
  private _playStateCue(state: AppState): void {
    switch (state) {
      case 'idle':
        audioCues.playReady();
        break;
      case 'listening':
        audioCues.playListening();
        break;
      case 'thinking':
        audioCues.playThinking();
        break;
      case 'speaking':
        // Speaking state doesn't need a cue - TTS audio serves this purpose
        break;
      case 'error':
        audioCues.playError();
        break;
    }
  }

  /**
   * Reset to initial state.
   */
  reset(): void {
    this._state = 'idle';
  }
}

// Export singleton instance
export const stateManager = new StateManager();

/**
 * Update UI elements based on state.
 */
export function updateUI(state: AppState): void {
  const button = document.getElementById('ptt-button');
  const statusEl = document.getElementById('status');

  if (!button || !statusEl) return;

  // Remove all state classes
  button.classList.remove('listening', 'thinking', 'speaking', 'error');
  statusEl.classList.remove('listening', 'thinking', 'speaking', 'error');

  // Add current state class and update text
  switch (state) {
    case 'idle':
      statusEl.textContent = 'Ready';
      break;
    case 'listening':
      button.classList.add('listening');
      statusEl.classList.add('listening');
      statusEl.textContent = 'Listening...';
      break;
    case 'thinking':
      button.classList.add('thinking');
      statusEl.classList.add('thinking');
      statusEl.textContent = 'Thinking...';
      break;
    case 'speaking':
      button.classList.add('speaking');
      statusEl.classList.add('speaking');
      statusEl.textContent = 'Speaking...';
      break;
    case 'error':
      button.classList.add('error');
      statusEl.classList.add('error');
      statusEl.textContent = 'Error occurred';
      break;
  }
}

