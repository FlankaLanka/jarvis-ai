/**
 * Jarvis Voice Assistant - WebSocket Client
 * 
 * Handles WebSocket communication with the backend for voice streaming.
 * Note: Using WebSocket for simplicity. Can be upgraded to WebRTC for lower latency.
 */

import type { WSMessage } from '../types';

export interface WebSocketClientOptions {
  url?: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

type MessageHandler = (message: WSMessage) => void;
type BinaryHandler = (data: ArrayBuffer) => void;
type ConnectionHandler = () => void;
type ErrorHandler = (error: Event) => void;

class WebSocketClient {
  private _ws: WebSocket | null = null;
  private _sessionId: string | null = null;
  private _reconnectAttempts: number = 0;
  private _maxReconnectAttempts: number = 5;
  private _reconnectInterval: number = 3000;
  private _url: string;

  private _messageHandlers: MessageHandler[] = [];
  private _binaryHandlers: BinaryHandler[] = [];
  private _connectHandlers: ConnectionHandler[] = [];
  private _disconnectHandlers: ConnectionHandler[] = [];
  private _errorHandlers: ErrorHandler[] = [];

  constructor(options: WebSocketClientOptions = {}) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    this._url = options.url || `${protocol}//${host}/api/voice/stream`;
    this._reconnectInterval = options.reconnectInterval || 3000;
    this._maxReconnectAttempts = options.maxReconnectAttempts || 5;
  }

  get isConnected(): boolean {
    return this._ws?.readyState === WebSocket.OPEN;
  }

  get sessionId(): string | null {
    return this._sessionId;
  }

  /**
   * Connect to the WebSocket server.
   */
  async connect(sessionId?: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this._ws = new WebSocket(this._url);
        this._ws.binaryType = 'arraybuffer';

        this._ws.onopen = () => {
          console.log('WebSocket connected');
          this._reconnectAttempts = 0;

          // Get selected model from UI
          const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
          const selectedModel = modelSelect?.value || 'gpt-3.5-turbo';

          // Send session initialization with model preference
          this._ws!.send(JSON.stringify({
            type: 'init',
            session_id: sessionId || null,
            model: selectedModel,
          }));

          this._connectHandlers.forEach(h => h());
          resolve();
        };

        this._ws.onmessage = (event) => {
          if (event.data instanceof ArrayBuffer) {
            // Binary audio data
            this._binaryHandlers.forEach(h => h(event.data));
          } else {
            // JSON message
            try {
              const message = JSON.parse(event.data) as WSMessage;
              
              // Handle session ready
              if (message.type === 'session_ready') {
                this._sessionId = message.session_id as string;
                console.log('Session established:', this._sessionId);
              }

              this._messageHandlers.forEach(h => h(message));
            } catch (e) {
              console.error('Failed to parse message:', e);
            }
          }
        };

        this._ws.onclose = () => {
          console.log('WebSocket disconnected');
          this._disconnectHandlers.forEach(h => h());
          this._attemptReconnect();
        };

        this._ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this._errorHandlers.forEach(h => h(error));
          reject(error);
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the server.
   */
  disconnect(): void {
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
    this._sessionId = null;
  }

  /**
   * Send audio data to the server.
   */
  sendAudio(data: ArrayBuffer | Blob): void {
    if (!this.isConnected) {
      console.error('Not connected');
      return;
    }

    if (data instanceof Blob) {
      data.arrayBuffer().then(buffer => {
        this._ws!.send(buffer);
      });
    } else {
      this._ws!.send(data);
    }
  }

  /**
   * Send a control message.
   */
  sendMessage(message: WSMessage): void {
    if (!this.isConnected) {
      console.error('Not connected');
      return;
    }

    this._ws!.send(JSON.stringify(message));
  }

  /**
   * Send interrupt signal.
   */
  interrupt(): void {
    this.sendMessage({ type: 'interrupt' });
  }

  /**
   * Clear conversation history.
   */
  clearHistory(): void {
    this.sendMessage({ type: 'clear_history' });
  }

  /**
   * Send ping for keep-alive.
   */
  ping(): void {
    this.sendMessage({ type: 'ping' });
  }

  /**
   * Register message handler.
   */
  onMessage(handler: MessageHandler): () => void {
    this._messageHandlers.push(handler);
    return () => {
      const i = this._messageHandlers.indexOf(handler);
      if (i > -1) this._messageHandlers.splice(i, 1);
    };
  }

  /**
   * Register binary data handler.
   */
  onBinary(handler: BinaryHandler): () => void {
    this._binaryHandlers.push(handler);
    return () => {
      const i = this._binaryHandlers.indexOf(handler);
      if (i > -1) this._binaryHandlers.splice(i, 1);
    };
  }

  /**
   * Register connect handler.
   */
  onConnect(handler: ConnectionHandler): () => void {
    this._connectHandlers.push(handler);
    return () => {
      const i = this._connectHandlers.indexOf(handler);
      if (i > -1) this._connectHandlers.splice(i, 1);
    };
  }

  /**
   * Register disconnect handler.
   */
  onDisconnect(handler: ConnectionHandler): () => void {
    this._disconnectHandlers.push(handler);
    return () => {
      const i = this._disconnectHandlers.indexOf(handler);
      if (i > -1) this._disconnectHandlers.splice(i, 1);
    };
  }

  /**
   * Register error handler.
   */
  onError(handler: ErrorHandler): () => void {
    this._errorHandlers.push(handler);
    return () => {
      const i = this._errorHandlers.indexOf(handler);
      if (i > -1) this._errorHandlers.splice(i, 1);
    };
  }

  /**
   * Attempt to reconnect.
   */
  private _attemptReconnect(): void {
    if (this._reconnectAttempts >= this._maxReconnectAttempts) {
      console.log('Max reconnect attempts reached');
      return;
    }

    this._reconnectAttempts++;
    console.log(`Reconnecting (${this._reconnectAttempts}/${this._maxReconnectAttempts})...`);

    setTimeout(() => {
      this.connect(this._sessionId || undefined).catch(console.error);
    }, this._reconnectInterval);
  }
}

// Export singleton instance
export const wsClient = new WebSocketClient();

