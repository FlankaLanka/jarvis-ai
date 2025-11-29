/**
 * Jarvis Voice Assistant - Audio Feedback Cues
 * 
 * Generates audio cues for state transitions using Web Audio API.
 * This is the PRIMARY feedback mechanism - visual UI is secondary.
 */

import type { AudioCueType } from '../types';

class AudioCues {
  private _audioContext: AudioContext | null = null;
  private _enabled: boolean = true;
  private _volume: number = 0.3;

  /**
   * Get or create the AudioContext.
   */
  private get audioContext(): AudioContext {
    if (!this._audioContext) {
      this._audioContext = new AudioContext();
    }
    return this._audioContext;
  }

  /**
   * Enable or disable audio cues.
   */
  setEnabled(enabled: boolean): void {
    this._enabled = enabled;
  }

  /**
   * Set volume (0.0 to 1.0).
   */
  setVolume(volume: number): void {
    this._volume = Math.max(0, Math.min(1, volume));
  }

  /**
   * Resume audio context (required after user interaction).
   */
  async resume(): Promise<void> {
    if (this._audioContext?.state === 'suspended') {
      await this._audioContext.resume();
    }
  }

  /**
   * Play "ready/idle" cue - brief descending tone.
   */
  playReady(): void {
    if (!this._enabled) return;
    this._playTone([440, 330], 0.1, 'sine');
  }

  /**
   * Play "listening" cue - brief ascending tone with confirmation.
   */
  playListening(): void {
    if (!this._enabled) return;
    // More noticeable ascending tone to indicate microphone is active
    this._playTone([440, 550, 660], 0.15, 'sine');
  }

  /**
   * Play "thinking/processing" cue - more noticeable chime.
   */
  playThinking(): void {
    if (!this._enabled) return;
    // More noticeable chime to indicate AI is processing
    this._playChime([523, 659, 784], 0.2);
  }
  
  /**
   * Start continuous listening indicator - subtle periodic beeps.
   */
  startListeningIndicator(intervalMs: number = 1500): () => void {
    if (!this._enabled) return () => {};
    
    // Play initial cue
    this.playListening();
    
    // Then play periodic subtle beeps
    const interval = setInterval(() => {
      if (this._enabled) {
        // Subtle beep to indicate still listening
        this._playTone([440], 0.05, 'sine');
      }
    }, intervalMs);
    
    return () => clearInterval(interval);
  }
  
  /**
   * Start continuous thinking indicator - periodic chimes.
   */
  startThinkingIndicator(intervalMs: number = 2000): () => void {
    if (!this._enabled) return () => {};
    
    // Play initial cue
    this.playThinking();
    
    // Then play periodic chimes
    const interval = setInterval(() => {
      if (this._enabled) {
        // Subtle thinking chime
        this._playChime([523, 659], 0.1);
      }
    }, intervalMs);
    
    return () => clearInterval(interval);
  }

  /**
   * Play "error" cue - low warning tone.
   */
  playError(): void {
    if (!this._enabled) return;
    this._playTone([200, 150], 0.2, 'sawtooth');
  }

  /**
   * Play progress chime for long-running tasks.
   */
  playProgress(): void {
    if (!this._enabled) return;
    this._playTone([440], 0.05, 'sine');
  }

  /**
   * Play a generic audio cue by type.
   */
  play(type: AudioCueType): void {
    switch (type) {
      case 'ready':
        this.playReady();
        break;
      case 'listening':
        this.playListening();
        break;
      case 'thinking':
        this.playThinking();
        break;
      case 'speaking':
        // No cue needed - TTS provides audio feedback
        break;
      case 'error':
        this.playError();
        break;
      case 'progress':
        this.playProgress();
        break;
    }
  }

  /**
   * Play a tone sequence.
   */
  private _playTone(
    frequencies: number[],
    duration: number,
    type: OscillatorType = 'sine'
  ): void {
    const ctx = this.audioContext;
    const now = ctx.currentTime;

    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();

    oscillator.type = type;
    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);

    // Set volume with fade out
    gainNode.gain.setValueAtTime(this._volume, now);
    gainNode.gain.exponentialRampToValueAtTime(0.01, now + duration);

    // Schedule frequency changes
    const timePerFreq = duration / frequencies.length;
    frequencies.forEach((freq, i) => {
      oscillator.frequency.setValueAtTime(freq, now + i * timePerFreq);
    });

    oscillator.start(now);
    oscillator.stop(now + duration);
  }

  /**
   * Play a chime (two overlapping tones).
   */
  private _playChime(frequencies: number[], duration: number): void {
    const ctx = this.audioContext;
    const now = ctx.currentTime;

    frequencies.forEach((freq, i) => {
      const oscillator = ctx.createOscillator();
      const gainNode = ctx.createGain();

      oscillator.type = 'sine';
      oscillator.frequency.value = freq;
      oscillator.connect(gainNode);
      gainNode.connect(ctx.destination);

      const startTime = now + i * 0.03;
      gainNode.gain.setValueAtTime(this._volume * 0.7, startTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, startTime + duration);

      oscillator.start(startTime);
      oscillator.stop(startTime + duration);
    });
  }

  /**
   * Play confirmation sound for user actions.
   */
  playConfirmation(): void {
    if (!this._enabled) return;
    this._playTone([600, 800], 0.08, 'sine');
  }

  /**
   * Play notification sound for system events.
   */
  playNotification(): void {
    if (!this._enabled) return;
    this._playChime([880, 1108], 0.12);
  }

  /**
   * Start periodic progress chimes for long tasks.
   */
  startProgressChimes(intervalMs: number = 2000): () => void {
    const interval = setInterval(() => {
      this.playProgress();
    }, intervalMs);

    // Return stop function
    return () => clearInterval(interval);
  }
}

// Export singleton instance
export const audioCues = new AudioCues();

