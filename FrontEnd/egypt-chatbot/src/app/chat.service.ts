import { Injectable } from '@angular/core';

interface ChatRequest {
  query: string;
  history: Array<{ role: string; content: string }>;
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  API = 'http://localhost:8000/chat/stream';

  async streamReply(text: string, history: Array<{ role: string; content: string }>, onChunk: (chunk: string, started: boolean) => void) {
    const body: ChatRequest = { query: text, history };

    const response = await fetch(this.API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok || !response.body) throw new Error('Stream failed');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let started = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      onChunk(chunk, !started);
      started = true;
    }
  }
}
