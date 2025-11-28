/**
 * Jarvis Voice Assistant - Microphone Input
 * 
 * Handles microphone access and audio capture using Web Audio API.
 */

export interface MicrophoneOptions {
  sampleRate?: number;
  channelCount?: number;
  echoCancellation?: boolean;
  noiseSuppression?: boolean;
  autoGainControl?: boolean;
}

const DEFAULT_OPTIONS: MicrophoneOptions = {
  sampleRate: 16000,
  channelCount: 1,
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

class Microphone {
  private _stream: MediaStream | null = null;
  private _audioContext: AudioContext | null = null;
  private _mediaRecorder: MediaRecorder | null = null;
  private _chunks: Blob[] = [];
  private _isRecording: boolean = false;
  private _onDataCallback: ((data: Blob) => void) | null = null;

  get isRecording(): boolean {
    return this._isRecording;
  }

  get hasPermission(): boolean {
    return this._stream !== null;
  }

  /**
   * Request microphone permission and initialize stream.
   */
  async requestPermission(options: MicrophoneOptions = {}): Promise<boolean> {
    const opts = { ...DEFAULT_OPTIONS, ...options };

    try {
      this._stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: opts.sampleRate,
          channelCount: opts.channelCount,
          echoCancellation: opts.echoCancellation,
          noiseSuppression: opts.noiseSuppression,
          autoGainControl: opts.autoGainControl,
        },
      });

      console.log('Microphone permission granted');
      return true;
    } catch (error) {
      console.error('Microphone permission denied:', error);
      return false;
    }
  }

  /**
   * Start recording audio.
   */
  startRecording(onData?: (data: Blob) => void): void {
    if (!this._stream) {
      console.error('No microphone stream available');
      return;
    }

    if (this._isRecording) {
      console.warn('Already recording');
      return;
    }

    this._chunks = [];
    this._onDataCallback = onData || null;

    // Create MediaRecorder with WebM/Opus format
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';

    this._mediaRecorder = new MediaRecorder(this._stream, {
      mimeType,
      audioBitsPerSecond: 128000,
    });

    this._mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this._chunks.push(event.data);
        
        // Call callback with each chunk if provided
        if (this._onDataCallback) {
          this._onDataCallback(event.data);
        }
      }
    };

    this._mediaRecorder.onerror = (event) => {
      console.error('MediaRecorder error:', event);
    };

    // Request data every 500ms to get larger chunks
    // We'll send the complete recording when button is released
    this._mediaRecorder.start(500);
    this._isRecording = true;

    console.log('Recording started');
  }

  /**
   * Stop recording and return the audio blob.
   */
  async stopRecording(): Promise<Blob | null> {
    if (!this._mediaRecorder || !this._isRecording) {
      return null;
    }

    return new Promise((resolve) => {
      this._mediaRecorder!.onstop = () => {
        const blob = new Blob(this._chunks, {
          type: this._mediaRecorder!.mimeType,
        });
        
        this._chunks = [];
        this._isRecording = false;
        
        console.log(`Recording stopped, size: ${blob.size} bytes`);
        resolve(blob);
      };

      this._mediaRecorder!.stop();
    });
  }

  /**
   * Get the current audio stream.
   */
  getStream(): MediaStream | null {
    return this._stream;
  }

  /**
   * Get audio data as ArrayBuffer (for sending to API).
   */
  async getAudioData(): Promise<ArrayBuffer | null> {
    const blob = await this.stopRecording();
    if (!blob) return null;
    
    return blob.arrayBuffer();
  }

  /**
   * Release microphone resources.
   */
  release(): void {
    if (this._mediaRecorder && this._isRecording) {
      this._mediaRecorder.stop();
    }

    if (this._stream) {
      this._stream.getTracks().forEach(track => track.stop());
      this._stream = null;
    }

    if (this._audioContext) {
      this._audioContext.close();
      this._audioContext = null;
    }

    this._mediaRecorder = null;
    this._isRecording = false;
    this._chunks = [];

    console.log('Microphone released');
  }

  /**
   * Check if microphone is supported.
   */
  static isSupported(): boolean {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }
}

// Export singleton instance
export const microphone = new Microphone();

