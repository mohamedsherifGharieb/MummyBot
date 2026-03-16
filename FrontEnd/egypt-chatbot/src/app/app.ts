import { Component, viewChild, ElementRef, signal, effect, inject } from '@angular/core';
import { ChatService } from './chat.service';
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
  private chat = inject(ChatService);
  scrollRef = viewChild<ElementRef>('scrollRef');

  API = this.chat.API;

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

    // Send last 10 messages (before this new user message) so the backend
    // can resolve pronouns like "he/she" across longer conversations.
    const history = this.messages().slice(-11, -1).map(m => ({ role: m.role, content: m.content }));

    try {
      await this.chat.streamReply(text, history, (chunk, isFirst) => {
        if (isFirst) {
          this.messages.update(msgs => [...msgs, { role: 'assistant', content: chunk }]);
        } else {
          this.messages.update(msgs => {
            const updated = [...msgs];
            const last = updated[updated.length - 1];
            updated[updated.length - 1] = { ...last, content: last.content + chunk };
            return updated;
          });
        }
      });
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