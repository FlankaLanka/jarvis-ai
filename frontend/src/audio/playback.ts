/**
 * Jarvis Voice Assistant - Audio Playback
 * 
 * Handles TTS audio playback and audio queue management.
 */

class AudioPlayback {
  private _audioContext: AudioContext | null = null;
  private _currentSource: AudioBufferSourceNode | null = null;
  private _audioQueue: ArrayBuffer[] = [];
  private _audioBuffer: Uint8Array | null = null; // Buffer for accumulating MP3 chunks
  private _isBuffering: boolean = false;
  private _isPlaying: boolean = false;
  private _onEndCallback: (() => void) | null = null;
  private _onStartCallback: (() => void) | null = null;
  private _volume: number = 1.0;
  private _gainNode: GainNode | null = null;
  private _bufferTimeout: number | null = null;

  get isPlaying(): boolean {
    return this._isPlaying;
  }

  /**
   * Get or create the AudioContext.
   */
  private get audioContext(): AudioContext {
    if (!this._audioContext) {
      this._audioContext = new AudioContext();
      this._gainNode = this._audioContext.createGain();
      this._gainNode.gain.value = this._volume;
      this._gainNode.connect(this._audioContext.destination);
    }
    return this._audioContext;
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
   * Set volume (0.0 to 1.0).
   */
  setVolume(volume: number): void {
    this._volume = Math.max(0, Math.min(1, volume));
    if (this._gainNode) {
      this._gainNode.gain.value = this._volume;
    }
  }

  /**
   * Set callback for when playback ends.
   */
  onEnd(callback: () => void): void {
    this._onEndCallback = callback;
  }

  /**
   * Set callback for when playback starts.
   */
  onStart(callback: () => void): void {
    this._onStartCallback = callback;
  }

  /**
   * Play audio from ArrayBuffer.
   * Handles MP3 format which requires complete file or valid frames.
   */
  async play(audioData: ArrayBuffer): Promise<void> {
    await this.resume();

    try {
      // Try to decode the audio data
      // For MP3, this should work if we have a complete or valid frame
      const audioBuffer = await this.audioContext.decodeAudioData(audioData.slice(0));
      
      // Stop current playback if any
      this.stop();

      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      
      if (this._gainNode) {
        source.connect(this._gainNode);
      } else {
        source.connect(this.audioContext.destination);
      }

      source.onended = () => {
        this._isPlaying = false;
        this._currentSource = null;
        this._playNext();
      };

      this._currentSource = source;
      this._isPlaying = true;
      
      // Notify that playback started
      if (this._onStartCallback) {
        this._onStartCallback();
      }
      
      source.start();

      console.log('Playing audio, duration:', audioBuffer.duration.toFixed(2), 'seconds');
    } catch (error) {
      console.error('Error playing audio:', error);
      console.error('Audio data size:', audioData.byteLength, 'bytes');
      this._isPlaying = false;
      this._playNext();
    }
  }

  /**
   * Add audio to queue.
   * For MP3, we need to buffer chunks and combine them before decoding.
   */
  queue(audioData: ArrayBuffer): void {
    console.log('Queueing audio chunk:', audioData.byteLength, 'bytes');
    
    // Start buffering if not already
    if (!this._isBuffering && !this._isPlaying) {
      this._isBuffering = true;
      this._audioBuffer = new Uint8Array(0);
    }
    
    // Append to buffer
    if (this._audioBuffer) {
      const newData = new Uint8Array(audioData);
      const combined = new Uint8Array(this._audioBuffer.length + newData.length);
      combined.set(this._audioBuffer);
      combined.set(newData, this._audioBuffer.length);
      this._audioBuffer = combined;
    }
    
    // Clear any existing timeout
    if (this._bufferTimeout) {
      clearTimeout(this._bufferTimeout);
    }
    
    // Wait a bit for more chunks, then play
    // This allows us to accumulate chunks before decoding
    // Note: We'll actually flush when we get the audio_complete message from server
    this._bufferTimeout = window.setTimeout(() => {
      // Only auto-flush if we've been buffering for a while (fallback)
      // Normally we wait for audio_complete message
      if (this._audioBuffer && this._audioBuffer.length > 5000 && !this._isPlaying) {
        console.log('Auto-flushing buffer after timeout');
        const completeAudio = new ArrayBuffer(this._audioBuffer.length);
        new Uint8Array(completeAudio).set(this._audioBuffer);
        this._audioBuffer = null;
        this._isBuffering = false;
        this._playCompleteAudio(completeAudio);
      }
    }, 500); // Fallback timeout - normally we wait for audio_complete
  }
  
  /**
   * Play complete audio buffer (for MP3).
   */
  private async _playCompleteAudio(audioData: ArrayBuffer): Promise<void> {
    console.log('Playing complete audio:', audioData.byteLength, 'bytes');
    await this.play(audioData);
  }

  /**
   * Play next item in queue.
   */
  private async _playNext(): Promise<void> {
    if (this._audioQueue.length === 0) {
      // Queue empty, notify end
      if (this._onEndCallback) {
        this._onEndCallback();
      }
      return;
    }

    const nextAudio = this._audioQueue.shift()!;
    await this.play(nextAudio);
  }

  /**
   * Stop current playback.
   */
  stop(): void {
    if (this._currentSource) {
      try {
        this._currentSource.stop();
      } catch {
        // Ignore errors from already stopped source
      }
      this._currentSource = null;
    }
    this._isPlaying = false;
  }

  /**
   * Stop playback and clear queue.
   */
  clear(): void {
    this.stop();
    this._audioQueue = [];
    this._audioBuffer = null;
    this._isBuffering = false;
    if (this._bufferTimeout) {
      clearTimeout(this._bufferTimeout);
      this._bufferTimeout = null;
    }
  }
  
  /**
   * Flush any buffered audio immediately.
   */
  flush(): void {
    if (this._bufferTimeout) {
      clearTimeout(this._bufferTimeout);
      this._bufferTimeout = null;
    }
    
    if (this._audioBuffer && this._audioBuffer.length > 0 && !this._isPlaying) {
      // Create a new ArrayBuffer from the Uint8Array
      const completeAudio = new ArrayBuffer(this._audioBuffer.length);
      new Uint8Array(completeAudio).set(this._audioBuffer);
      this._audioBuffer = null;
      this._isBuffering = false;
      this._playCompleteAudio(completeAudio);
    }
  }

  /**
   * Play audio from URL.
   */
  async playFromUrl(url: string): Promise<void> {
    try {
      const response = await fetch(url);
      const arrayBuffer = await response.arrayBuffer();
      await this.play(arrayBuffer);
    } catch (error) {
      console.error('Error loading audio from URL:', error);
    }
  }

  /**
   * Play audio from Blob.
   */
  async playFromBlob(blob: Blob): Promise<void> {
    const arrayBuffer = await blob.arrayBuffer();
    await this.play(arrayBuffer);
  }

  /**
   * Stream audio chunks.
   */
  async streamChunks(chunks: AsyncIterable<ArrayBuffer>): Promise<void> {
    for await (const chunk of chunks) {
      this.queue(chunk);
    }
  }

  /**
   * Release resources.
   */
  release(): void {
    this.clear();
    
    if (this._audioContext) {
      this._audioContext.close();
      this._audioContext = null;
    }
    
    this._gainNode = null;
  }
}

// Export singleton instance
export const audioPlayback = new AudioPlayback();

