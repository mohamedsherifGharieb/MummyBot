import { Component, viewChild, ElementRef, signal, effect, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { ChatBox } from './components/chatbox/chatbox';
import { ChatInput } from './components/chat-input/chat-input';
import { PyramidLoader } from './components/pyramid-loader/pyramid-loader';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

@Component({
  selector: 'app-root',
  imports: [
    FormsModule,
    ChatBox,
    ChatInput,
    PyramidLoader,
  ],
  templateUrl: './app.html',
  standalone: true,
  styleUrls: ['./app.scss'],
})
export class App {
  private http = inject(HttpClient);
  scrollRef = viewChild<ElementRef>('scrollRef');

  API = 'http://localhost:8000/chat/stream';

  messages = signal<Message[]>([]);
  isLoading = signal(false);

  suggestions = [
    'Who built the pyramids?',
    'Tell me about Cleopatra',
    'How did mummification work?',
  ];

  constructor() {
    effect(() => {
      this.messages();
      setTimeout(() => this.scrollToBottom(), 50);
    });
  }

  async handleSend(text: string) {
    if (!text.trim() || this.isLoading()) return;

    this.messages.update(msgs => [...msgs, { role: 'user', content: text }]);
    this.isLoading.set(true);

    const history = this.messages().slice(-5, -1).map(m => ({ role: m.role, content: m.content }));

    try {
      const response = await fetch(this.API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, history }),
      });

      if (!response.ok || !response.body) throw new Error('Stream failed');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let started = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        if (!started) {
          this.messages.update(msgs => [...msgs, { role: 'assistant', content: chunk }]);
          started = true;
        } else {
          this.messages.update(msgs => {
            const updated = [...msgs];
            const last = updated[updated.length - 1];
            updated[updated.length - 1] = { ...last, content: last.content + chunk };
            return updated;
          });
        }
      }
    } catch {
      this.messages.update(msgs => [
        ...msgs,
        { role: 'assistant', content: 'Could not reach the server. Make sure the backend is running.' },
      ]);
    } finally {
      this.isLoading.set(false);
    }
  }

  private scrollToBottom() {
    try {
      const el = this.scrollRef()?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    } catch {}
  }
}