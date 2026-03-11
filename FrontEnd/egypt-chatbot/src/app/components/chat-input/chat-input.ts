import { Component, input, output, viewChild, ElementRef, signal, effect } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-chat-input',
  imports: [FormsModule],
  templateUrl: './chat-input.html',
  styleUrl: './chat-input.scss',
})
export class ChatInput {
  disabled = input(false);
  sendMessage = output<string>();
  textareaRef = viewChild<ElementRef<HTMLTextAreaElement>>('textareaRef');

  value = signal('');

  constructor() {
    effect(() => {
      this.value();
      setTimeout(() => this.autoResize(), 0);
    });
  }

  autoResize() {
    const el = this.textareaRef()?.nativeElement;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }

  handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      this.submit();
    }
  }

  submit() {
    const text = this.value().trim();
    if (!text || this.disabled()) return;
    this.sendMessage.emit(text);
    this.value.set('');
  }
}