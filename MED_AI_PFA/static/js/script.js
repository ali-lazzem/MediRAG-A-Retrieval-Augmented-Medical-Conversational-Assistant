// FILE: static/js/script.js (modified to handle title updates & animations)
let currentSessionId = null;
let chats = [];

const elements = {
    sidebar: document.querySelector('.sidebar'),
    sidebarToggle: document.getElementById('sidebar-toggle'),
    sidebarClose: document.getElementById('sidebar-close'),
    newChatBtn: document.getElementById('new-chat-btn'),
    startChatBtn: document.getElementById('start-chat-btn'),
    chatList: document.getElementById('chat-list'),
    chatContainer: document.getElementById('chat-container'),
    welcomeMessage: document.getElementById('welcome-message'),
    messageInput: document.getElementById('message-input'),
    sendBtn: document.getElementById('send-btn'),
    typingIndicator: document.getElementById('typing-indicator')
};

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function addCsrf(options = {}) {
    const csrfToken = getCookie('csrftoken');
    if (csrfToken) {
        options.headers = options.headers || {};
        options.headers['X-CSRFToken'] = csrfToken;
    }
    return options;
}

async function fetchSessions() {
    const response = await fetch('/api/sessions/');
    if (response.ok) return await response.json();
    return [];
}

async function loadSessions() {
    chats = await fetchSessions();
    renderChatList();
    if (chats.length > 0 && !currentSessionId) {
        switchToSession(chats[0].session_id);
    } else if (chats.length === 0) {
        createNewSession();
    }
}

async function createNewSession() {
    try {
        const response = await fetch('/api/sessions/create/', addCsrf({
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: 'New Consultation' })
        }));
        if (!response.ok) throw new Error('Failed to create session');
        const data = await response.json();
        await loadSessions();
        switchToSession(data.session_id);
    } catch (error) {
        console.error('Create session error:', error);
        showToast('Could not create new conversation. Please try again.', 'error');
    }
}

async function switchToSession(sessionId) {
    currentSessionId = sessionId;
    const session = chats.find(s => s.session_id === sessionId);
    if (!session) return;
    try {
        const msgsResponse = await fetch(`/api/sessions/${sessionId}/`);
        if (!msgsResponse.ok) throw new Error('Failed to load messages');
        const messages = await msgsResponse.json();
        renderChatMessages(messages);
        elements.welcomeMessage.style.display = 'none';
        renderChatList(); // highlight active
        elements.messageInput.focus();
    } catch (error) {
        console.error('Switch session error:', error);
        showToast('Could not load conversation history.', 'error');
    }
}

async function renameSession(sessionId, newTitle) {
    try {
        const response = await fetch(`/api/sessions/${sessionId}/rename/`, addCsrf({
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newTitle })
        }));
        if (!response.ok) throw new Error('Rename failed');
        await loadSessions(); // refresh list
        // Add animation effect to the updated title
        const updatedItem = document.querySelector(`.chat-item[data-session-id="${sessionId}"] .chat-title`);
        if (updatedItem) {
            updatedItem.classList.add('update-animation');
            setTimeout(() => updatedItem.classList.remove('update-animation'), 300);
        }
    } catch (error) {
        console.error('Rename error:', error);
    }
}

async function deleteSession(sessionId) {
    if (!confirm('Delete this consultation? This action cannot be undone.')) return;
    try {
        const response = await fetch(`/api/sessions/${sessionId}/delete/`, addCsrf({ method: 'DELETE' }));
        if (!response.ok) throw new Error('Delete failed');
        chats = chats.filter(s => s.session_id !== sessionId);
        renderChatList();
        if (currentSessionId === sessionId) {
            if (chats.length > 0) {
                switchToSession(chats[0].session_id);
            } else {
                await createNewSession();
            }
        }
    } catch (error) {
        console.error('Delete error:', error);
        showToast('Could not delete conversation.', 'error');
    }
}

function renderChatList() {
    elements.chatList.innerHTML = '';
    chats.forEach(session => {
        const li = document.createElement('li');
        li.className = `chat-item ${session.session_id === currentSessionId ? 'active' : ''}`;
        li.setAttribute('data-session-id', session.session_id);
        let displayTitle = session.title;
        if (displayTitle.length > 35) displayTitle = displayTitle.substring(0, 35) + '…';
        li.innerHTML = `
            <span class="chat-title">${escapeHtml(displayTitle)}</span>
            <div class="chat-actions">
                <button class="chat-action-btn rename-chat" title="Rename"><i class="fas fa-edit"></i></button>
                <button class="chat-action-btn delete-chat" title="Delete"><i class="fas fa-trash"></i></button>
            </div>
        `;
        li.addEventListener('click', (e) => {
            if (!e.target.closest('.chat-action-btn')) switchToSession(session.session_id);
        });
        li.querySelector('.rename-chat').addEventListener('click', (e) => {
            e.stopPropagation();
            const newTitle = prompt('Rename consultation:', session.title);
            if (newTitle && newTitle.trim()) renameSession(session.session_id, newTitle.trim());
        });
        li.querySelector('.delete-chat').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteSession(session.session_id);
        });
        elements.chatList.appendChild(li);
    });
}

function renderChatMessages(messages) {
    elements.chatContainer.innerHTML = '';
    if (!messages || messages.length === 0) return;
    messages.forEach(msg => {
        const div = document.createElement('div');
        div.className = `message ${msg.role === 'bot' ? 'bot' : 'user'}`;
        const time = msg.timestamp
            ? new Date(msg.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            : '';
        div.innerHTML = `<div class="message-content">${escapeHtml(msg.content)}</div><div class="message-time">${time}</div>`;
        elements.chatContainer.appendChild(div);
    });
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
}

async function sendMessage() {
    const text = elements.messageInput.value.trim();
    if (!text) return;
    elements.messageInput.value = '';
    elements.messageInput.style.height = 'auto';
    // Add user message temporarily
    const tempUserDiv = document.createElement('div');
    tempUserDiv.className = 'message user';
    tempUserDiv.innerHTML = `<div class="message-content">${escapeHtml(text)}</div><div class="message-time"></div>`;
    elements.chatContainer.appendChild(tempUserDiv);
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    elements.typingIndicator.style.display = 'flex';
    try {
        const response = await fetch('/api/ask/', addCsrf({
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: text, top_k: 5, session_id: currentSessionId })
        }));
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || 'Request failed');
        }
        const data = await response.json();
        elements.typingIndicator.style.display = 'none';
        currentSessionId = data.session_id;
        await loadSessions(); // refreshes sidebar – will show updated title if backend renamed
        const msgsResponse = await fetch(`/api/sessions/${currentSessionId}/`);
        const messages = await msgsResponse.json();
        renderChatMessages(messages);
        console.log('Sources:', data.chunks_used);
    } catch (error) {
        elements.typingIndicator.style.display = 'none';
        console.error('Send message error:', error);
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message bot';
        errorDiv.innerHTML = `<div class="message-content">⚠️ An error occurred. Please ensure the backend is running and try again.</div><div class="message-time"></div>`;
        elements.chatContainer.appendChild(errorDiv);
        elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
        showToast('Failed to send message. Check console for details.', 'error');
    }
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.position = 'fixed';
    toast.style.bottom = '80px';
    toast.style.right = '20px';
    toast.style.backgroundColor = type === 'error' ? '#dc3545' : '#2c7da0';
    toast.style.color = 'white';
    toast.style.padding = '12px 20px';
    toast.style.borderRadius = '8px';
    toast.style.zIndex = '2000';
    toast.style.fontSize = '14px';
    toast.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function setupEventListeners() {
    elements.sidebarToggle?.addEventListener('click', () => elements.sidebar.classList.add('open'));
    elements.sidebarClose?.addEventListener('click', () => elements.sidebar.classList.remove('open'));
    elements.newChatBtn?.addEventListener('click', createNewSession);
    elements.startChatBtn?.addEventListener('click', createNewSession);
    elements.sendBtn?.addEventListener('click', sendMessage);
    elements.messageInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    elements.messageInput?.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
}

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadSessions();
});