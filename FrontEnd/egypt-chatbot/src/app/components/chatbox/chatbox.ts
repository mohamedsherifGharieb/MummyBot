import { Component, input, computed } from '@angular/core';

@Component({
  selector: 'app-chat-message',
  templateUrl: './chatbox.html',
  styleUrl: './chatbox.scss',
})
export class ChatBox {
  role = input<'user' | 'assistant'>('user');
  content = input('');

  isAssistant = computed(() => this.role() === 'assistant');

  formattedContent = computed(() => {
    if (!this.isAssistant()) return this.escapeHtml(this.content());
    return this.renderMarkdown(this.content());
  });

  private escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  private renderMarkdown(text: string): string {
    let html = this.escapeHtml(text);
    html = html.replace(/^## (.+)$/gm, '<h3 class="md-h2">$1</h3>');
    html = html.replace(/^### (.+)$/gm, '<h4 class="md-h3">$1</h4>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
    html = html.replace(/\n/g, '<br>');
    html = html.replace(/<br>(<h[34])/g, '$1');
    html = html.replace(/(<\/h[34]>)<br>/g, '$1');
    html = html.replace(/<br>(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)<br>/g, '$1');
    return html;
  }
}