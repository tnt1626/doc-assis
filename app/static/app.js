// State Management
let documentList = [];
let selectedDocumentId = ""; // Default to Search All
let sessionList = [];
let currentSessionId = "";

// DOM Elements
const docListContainer = document.getElementById('doc-list');
const docCountEl = document.getElementById('doc-count');
const docSelect = document.getElementById('doc-select');
const fileInput = document.getElementById('file-input');
const uploadDropzone = document.getElementById('upload-dropzone');
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const clearChatBtn = document.getElementById('clear-chat-btn');
const toastEl = document.getElementById('toast-notification');
const sessionListEl = document.getElementById('session-list');
const newChatBtn = document.getElementById('new-chat-btn');

// Initialize Icons
function initIcons() {
    if (window.lucide) {
        window.lucide.createIcons();
    }
}

// Show Toast Notifications
function showToast(message, type = 'info') {
    toastEl.textContent = message;
    toastEl.className = `toast show ${type}`;
    
    // Add icon to toast
    const iconName = type === 'success' ? 'check-circle' : type === 'error' ? 'alert-triangle' : 'info';
    const icon = document.createElement('i');
    icon.setAttribute('data-lucide', iconName);
    toastEl.insertBefore(icon, toastEl.firstChild);
    initIcons();

    setTimeout(() => {
        toastEl.classList.remove('show');
    }, 4000);
}

let pollInterval = null;

// Fetch Document List from API
async function fetchDocuments(showError = true) {
    try {
        const response = await fetch('/documents/?limit=50');
        if (!response.ok) throw new Error("Failed to fetch documents");
        documentList = await response.json();
        renderDocuments();
        updateDocumentDropdown();
        checkAndStartPolling();
    } catch (error) {
        console.error(error);
        if (showError) showToast("Error loading document list", "error");
    }
}

// Auto-poll document list when any document is processing or pending
function checkAndStartPolling() {
    const hasUnfinished = documentList.some(doc => {
        const s = (doc.status || '').toLowerCase();
        return s === 'processing' || s === 'pending';
    });
    if (hasUnfinished) {
        if (!pollInterval) {
            pollInterval = setInterval(() => fetchDocuments(false), 3000);
        }
    } else {
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
    }
}

// Fetch Sessions List from API
async function fetchSessions() {
    try {
        const response = await fetch('/session/?limit=20');
        if (!response.ok) throw new Error("Failed to fetch sessions");
        sessionList = await response.json();
        renderSessions();

        if (sessionList.length > 0 && !currentSessionId) {
            await selectSession(sessionList[0].id);
        } else if (sessionList.length === 0) {
            await createNewSession("New Conversation", false);
        }
    } catch (error) {
        console.error("Error loading sessions:", error);
    }
}

// Render Sessions in Sidebar
function renderSessions() {
    if (!sessionListEl) return;
    sessionListEl.innerHTML = '';

    if (sessionList.length === 0) {
        sessionListEl.innerHTML = `
            <div style="text-align: center; padding: 1rem; color: var(--text-secondary); font-size: 0.8rem;">
                No active sessions.
            </div>
        `;
        return;
    }

    sessionList.forEach(s => {
        const isActive = s.id === currentSessionId;
        const item = document.createElement('div');
        item.className = `session-item ${isActive ? 'active' : ''}`;
        item.onclick = () => selectSession(s.id);

        item.innerHTML = `
            <span class="session-title-text" title="${escapeHtml(s.title || 'Chat')}">${escapeHtml(s.title || 'Chat Session')}</span>
            <div style="display: flex; gap: 4px; align-items: center;">
                <button class="session-item-del" title="Rename session" onclick="event.stopPropagation(); renameSession('${s.id}', '${escapeHtml(s.title || '')}')">
                    <i data-lucide="edit-3" style="width: 12px; height: 12px;"></i>
                </button>
                <button class="session-item-del" title="Delete session" onclick="event.stopPropagation(); deleteSession('${s.id}')">
                    <i data-lucide="trash-2" style="width: 12px; height: 12px;"></i>
                </button>
            </div>
        `;
        sessionListEl.appendChild(item);
    });
    initIcons();
}

// Rename Chat Session
async function renameSession(sessionId, currentTitle) {
    const newTitle = prompt("Enter new title for this conversation:", currentTitle || "New Conversation");
    if (!newTitle || newTitle.trim() === "" || newTitle.trim() === currentTitle) return;

    try {
        const response = await fetch(`/session/${sessionId}/`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newTitle.trim() })
        });
        if (!response.ok) throw new Error("Failed to rename session");
        showToast("Session renamed", "success");
        await fetchSessions();
    } catch (error) {
        console.error(error);
        showToast("Failed to rename session", "error");
    }
}

// Create New Chat Session
async function createNewSession(title = "New Conversation", loadWelcome = true) {
    try {
        const response = await fetch('/session/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: title })
        });
        if (!response.ok) throw new Error("Failed to create session");
        const newSession = await response.json();
        currentSessionId = newSession.id;
        showToast("Started new chat session", "info");
        await fetchSessions();
        if (loadWelcome) {
            renderWelcomeMessage();
        }
    } catch (error) {
        console.error(error);
        showToast("Error creating session", "error");
    }
}

// Select a Chat Session
async function selectSession(sessionId) {
    currentSessionId = sessionId;
    renderSessions();
    await loadSessionMessages(sessionId);
}

// Load Session Message History from API
async function loadSessionMessages(sessionId) {
    try {
        const response = await fetch(`/session/${sessionId}/messages/?limit=50`);
        if (!response.ok) throw new Error("Failed to fetch messages");
        const messages = await response.json();
        renderSessionHistory(messages);
    } catch (error) {
        console.error("Error loading session messages:", error);
    }
}

// Render Historical Messages to Chat UI
function renderSessionHistory(messages) {
    chatBox.innerHTML = '';
    if (!messages || messages.length === 0) {
        renderWelcomeMessage();
        return;
    }

    messages.forEach(msg => {
        const isMsg = msg.type === 'message' || msg.type === 'MESSAGE';
        if (isMsg) {
            const contentText = typeof msg.content === 'object' ? (msg.content.text || '') : msg.content;
            if (contentText) {
                appendMessage(contentText, msg.role);
            }
        }
    });
}

function renderWelcomeMessage() {
    chatBox.innerHTML = `
        <div class="message assistant">
            <div class="message-bubble">
                Hello! I am your intelligent document assistant. Upload a text file on the left side to get started, or select a document to query. You can ask me to search specific documents or search globally across all uploaded knowledge.
            </div>
            <div class="message-meta">Agent • Just now</div>
        </div>
    `;
}

// Delete Chat Session
async function deleteSession(sessionId) {
    if (!confirm("Delete this chat session?")) return;
    try {
        const response = await fetch(`/session/${sessionId}/`, { method: 'DELETE' });
        if (!response.ok) throw new Error("Failed to delete session");
        showToast("Session deleted", "info");
        if (currentSessionId === sessionId) {
            currentSessionId = "";
        }
        await fetchSessions();
    } catch (error) {
        console.error(error);
        showToast("Failed to delete session", "error");
    }
}

// Render Documents in Sidebar
function renderDocuments() {
    docCountEl.textContent = documentList.length;
    docListContainer.innerHTML = '';

    if (documentList.length === 0) {
        docListContainer.innerHTML = `
            <div style="text-align: center; padding: 1.5rem; color: var(--text-secondary); font-size: 0.85rem;">
                No documents uploaded yet.
            </div>
        `;
        return;
    }

    documentList.forEach(doc => {
        const isActive = doc.id === selectedDocumentId;
        const item = document.createElement('div');
        item.className = `document-item ${isActive ? 'active' : ''}`;
        item.onclick = () => selectDocument(doc.id);

        const dateStr = new Date(doc.created_at).toLocaleString();

        const rawStatus = (doc.status || 'completed').toLowerCase();
        const statusClass = rawStatus;
        let statusLabel = rawStatus;
        let statusIcon = '';
        if (rawStatus === 'completed') {
            statusLabel = 'Ready';
            statusIcon = '<span style="width: 6px; height: 6px; border-radius: 50%; background-color: var(--accent-emerald);"></span>';
        } else if (rawStatus === 'processing') {
            statusLabel = 'Processing';
            statusIcon = '<div class="agent-spinner" style="width: 8px; height: 8px; border-width: 1px;"></div>';
        } else if (rawStatus === 'pending') {
            statusLabel = 'Pending';
            statusIcon = '<span style="width: 6px; height: 6px; border-radius: 50%; background-color: var(--accent-amber);"></span>';
        } else if (rawStatus === 'failed') {
            statusLabel = 'Failed';
            statusIcon = '<span style="color: var(--accent-rose); font-weight: bold;">✕</span>';
        }

        item.innerHTML = `
            <div class="doc-header">
                <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
                <div class="doc-actions">
                    <span class="status-badge-pill ${statusClass}">
                        ${statusIcon} ${statusLabel}
                    </span>
                    <button class="doc-btn delete-btn" title="Delete Document" onclick="event.stopPropagation(); deleteDoc('${doc.id}')">
                        <i data-lucide="trash-2" style="width: 14px; height: 14px;"></i>
                    </button>
                </div>
            </div>
            <div class="doc-info">
                <span>Uploaded: ${dateStr.split(',')[0]}</span>
            </div>
            <div>
                <span class="doc-id-badge" onclick="event.stopPropagation(); copyToClipboard('${doc.id}')" title="Click to copy UUID">
                    <i data-lucide="copy" style="width: 10px; height: 10px;"></i>
                    ${doc.id.substring(0, 8)}...
                </span>
            </div>
        `;
        docListContainer.appendChild(item);
    });
    initIcons();
}

// Select a document from sidebar
function selectDocument(docId) {
    const doc = documentList.find(d => d.id === docId);
    if (doc && doc.status && doc.status.toLowerCase() !== 'completed') {
        showToast(`Document status is '${doc.status}'. Only completed documents can be queried.`, "error");
        return;
    }

    if (selectedDocumentId === docId) {
        selectedDocumentId = ""; // Toggle off
    } else {
        selectedDocumentId = docId;
    }
    docSelect.value = selectedDocumentId;
    renderDocuments();
    showToast(selectedDocumentId ? `Selected document context.` : `Switched to global search.`, "info");
}

// Update the Document Selection Dropdown
function updateDocumentDropdown() {
    docSelect.innerHTML = '<option value="">🔍 Search All Documents</option>';
    documentList.forEach(doc => {
        const option = document.createElement('option');
        option.value = doc.id;
        const isNotCompleted = doc.status && doc.status.toLowerCase() !== 'completed';
        const statusText = isNotCompleted ? ` [${doc.status.toUpperCase()}]` : '';
        option.textContent = `📄 ${doc.filename}${statusText}`;
        if (isNotCompleted) {
            option.disabled = true;
        }
        docSelect.appendChild(option);
    });
    docSelect.value = selectedDocumentId;
}

// Dropdown Change Handler
docSelect.onchange = (e) => {
    const doc = documentList.find(d => d.id === e.target.value);
    if (doc && doc.status && doc.status.toLowerCase() !== 'completed') {
        showToast(`Document status is '${doc.status}'. Only completed documents can be queried.`, "error");
        docSelect.value = selectedDocumentId;
        return;
    }
    selectedDocumentId = e.target.value;
    renderDocuments();
};

// Copy Document UUID
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast("Copied Document ID to clipboard!", "success");
    }).catch(err => {
        console.error("Could not copy text: ", err);
    });
}

// Delete Document
async function deleteDoc(docId) {
    if (!confirm("Are you sure you want to delete this document? All embedded chunks will be deleted permanently.")) return;
    
    try {
        const response = await fetch(`/documents/${docId}/`, {
            method: 'DELETE'
        });
        if (!response.ok) throw new Error("Delete failed");
        
        showToast("Document deleted successfully", "success");
        if (selectedDocumentId === docId) selectedDocumentId = "";
        await fetchDocuments();
    } catch (error) {
        console.error(error);
        showToast("Failed to delete document", "error");
    }
}

// Drag and Drop File Upload
uploadDropzone.onclick = () => fileInput.click();
fileInput.onchange = () => handleFileUpload(fileInput.files[0]);

uploadDropzone.ondragover = (e) => {
    e.preventDefault();
    uploadDropzone.classList.add('dragover');
};

uploadDropzone.ondragleave = () => {
    uploadDropzone.classList.remove('dragover');
};

uploadDropzone.ondrop = (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
};

async function handleFileUpload(file) {
    if (!file) return;
    
    const ALLOWED_EXTENSIONS = ['.txt', '.md', '.html', '.css', '.csv', '.xml', '.json'];
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
        showToast("Unsupported file format! Supported: " + ALLOWED_EXTENSIONS.join(', '), "error");
        return;
    }

    showToast(`Uploading and embedding '${file.name}'...`, "info");

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/documents/?chunk_size=500&overlap=50', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errDetail = await response.json();
            throw new Error(errDetail.detail || "Upload failed");
        }

        const data = await response.json();
        showToast(`Document uploaded successfully! Indexing task scheduled.`, "success");
        await fetchDocuments();
    } catch (error) {
        console.error(error);
        showToast(`Upload failed: ${error.message}`, "error");
    }
}

// Chat Implementation
sendBtn.onclick = () => submitQuestion();
userInput.onkeypress = (e) => {
    if (e.key === 'Enter') submitQuestion();
};

if (newChatBtn) {
    newChatBtn.onclick = () => createNewSession("New Conversation", true);
}

async function submitQuestion() {
    const questionText = userInput.value.trim();
    if (!questionText) return;

    if (!currentSessionId) {
        await createNewSession("New Conversation", false);
    }

    // 1. Add User Message to Chat UI
    appendMessage(questionText, 'user');
    userInput.value = '';
    
    // Disable inputs during processing
    sendBtn.disabled = true;
    userInput.disabled = true;

    // 2. Add Loading Indicator
    const loadingMessageId = appendLoadingIndicator();

    try {
        // Prepare request body
        const requestBody = {
            question: questionText,
            session_id: currentSessionId,
            limit: 20
        };
        if (selectedDocumentId) {
            requestBody.document_id = selectedDocumentId;
        }

        const response = await fetch('/agent/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        // Remove loading indicator
        removeLoadingIndicator(loadingMessageId);

        if (!response.ok) {
            const errBody = await response.json();
            throw new Error(errBody.detail || "Server error");
        }

        // 3. Setup stream readers
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        // Create empty assistant bubble that we will populate dynamically
        const assistantMessageId = appendEmptyAssistantBubble();
        let accumulatedAnswer = "";
        let steps = [];

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Split buffer by double newline to separate SSE event packets
            const parts = buffer.split("\n\n");
            // Keep the last partial packet in the buffer
            buffer = parts.pop();

            for (const part of parts) {
                if (!part.trim()) continue;

                // Parse SSE format: event: ... \n data: ...
                const lines = part.split("\n");
                let eventType = "";
                let dataString = "";

                for (const line of lines) {
                    if (line.startsWith("event:")) {
                        eventType = line.replace("event:", "").trim();
                    } else if (line.startsWith("data:")) {
                        dataString = line.replace("data:", "").trim();
                    }
                }

                // Process events
                if (eventType === "thought" && dataString) {
                    try {
                        const step = JSON.parse(dataString);
                        steps.push(step);
                        // Force accordion to open while agent is actively thinking
                        updateAssistantThoughtTrace(assistantMessageId, steps, true);
                    } catch (e) {
                        console.error("Failed to parse thought step JSON:", e);
                    }
                } else if (eventType === "answer" && dataString) {
                    try {
                        const payload = JSON.parse(dataString);
                        // Collapse the thoughts trace when the final answer starts streaming
                        collapseAssistantThoughtTrace(assistantMessageId);
                        accumulatedAnswer += payload.text;
                        updateAssistantTextContent(assistantMessageId, accumulatedAnswer);
                    } catch (e) {
                        console.error("Failed to parse answer chunk JSON:", e);
                    }
                } else if (eventType === "done") {
                    console.log("Streaming completed");
                }
            }
        }

        // Remove cursor indicator
        removeStreamingCursor(assistantMessageId);
        await fetchSessions();

    } catch (error) {
        removeLoadingIndicator(loadingMessageId);
        console.error(error);
        appendMessage(`Sorry, I encountered an error: ${error.message}`, 'assistant');
        showToast("Error retrieving answer", "error");
    } finally {
        sendBtn.disabled = false;
        userInput.disabled = false;
        userInput.focus();
    }
}

// Create an empty assistant message bubble
function appendEmptyAssistantBubble() {
    const id = `msg-${Math.random().toString(36).substr(2, 9)}`;
    const messageContainer = document.createElement('div');
    messageContainer.className = 'message assistant';
    messageContainer.id = id;

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    messageContainer.innerHTML = `
        <div class="message-bubble markdown-body" id="${id}-bubble">
            <!-- Container for streaming final answer -->
            <div class="agent-text-answer" id="${id}-answer"></div>
        </div>
        <div class="message-meta">Agent • ${timestamp}</div>
    `;
    chatBox.appendChild(messageContainer);
    chatBox.scrollTop = chatBox.scrollHeight;
    return id;
}

// Update the assistant text bubble with markdown content, removing the active checklist loader
function updateAssistantTextContent(id, text) {
    const bubble = document.getElementById(`${id}-bubble`);
    if (!bubble) return;

    // Remove the active loading item from the checklist if present
    const activeItem = document.getElementById(`${id}-active-item`);
    if (activeItem) activeItem.remove();

    const checklist = document.getElementById(`${id}-checklist`);
    if (checklist && checklist.children.length === 0) {
        checklist.remove();
    }

    const parsedText = window.marked ? window.marked.parse(text) : escapeHtml(text);
    
    // Find or create the text answer container
    let textContainer = document.getElementById(`${id}-answer`);
    if (!textContainer) {
        textContainer = document.createElement('div');
        textContainer.className = 'agent-text-answer';
        textContainer.id = `${id}-answer`;
        
        // Insert it after checklist but before accordion
        const accordion = bubble.querySelector('.thought-accordion');
        if (accordion) {
            bubble.insertBefore(textContainer, accordion);
        } else {
            bubble.appendChild(textContainer);
        }
    }

    textContainer.innerHTML = parsedText + '<span class="streaming-cursor">█</span>';
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Update the thought trace accordion and build the dynamic activity checklist
function updateAssistantThoughtTrace(id, steps, forceOpen = false) {
    const bubble = document.getElementById(`${id}-bubble`);
    if (!bubble) return;

    // 1. Re-render or create the activity checklist dynamically!
    let checklistEl = document.getElementById(`${id}-checklist`);
    if (!checklistEl) {
        checklistEl = document.createElement('div');
        checklistEl.className = 'agent-activity-checklist';
        checklistEl.id = `${id}-checklist`;
        bubble.insertBefore(checklistEl, bubble.firstChild);
    }

    let checklistHtml = "";
    
    steps.forEach((step) => {
        let taskDescription = `Step ${step.loop_index + 1}: Thought process completed`;
        if (step.tool_calls && step.tool_calls.length > 0) {
            const toolName = step.tool_calls[0].name;
            if (toolName === "search_document") {
                taskDescription = `Step ${step.loop_index + 1}: Searched database for relevant chunks`;
            } else if (toolName === "list_documents") {
                taskDescription = `Step ${step.loop_index + 1}: Scanned knowledge base document list`;
            } else if (toolName === "get_full_document") {
                taskDescription = `Step ${step.loop_index + 1}: Read full content of the document`;
            }
        }
        
        checklistHtml += `
            <div class="agent-activity-item completed">
                <span class="agent-activity-icon check">✓</span>
                <span class="agent-activity-text">${taskDescription}</span>
            </div>
        `;
    });

    // Append the next active step prediction
    const nextStepNum = steps.length + 1;
    checklistHtml += `
        <div class="agent-activity-item active" id="${id}-active-item">
            <span class="agent-activity-icon"><div class="agent-spinner"></div></span>
            <span class="agent-activity-text">Step ${nextStepNum}: Processing next loop step...</span>
        </div>
    `;
    
    checklistEl.innerHTML = checklistHtml;

    // Remove existing accordion if any
    const oldAccordion = bubble.querySelector('.thought-accordion');
    let wasOpen = forceOpen;
    if (oldAccordion) {
        // If not forced, retain the user's manual toggle state
        if (!forceOpen) {
            wasOpen = oldAccordion.classList.contains('open');
        }
        oldAccordion.remove();
    }

    // Create new accordion node
    const accordionHtml = renderThoughtSteps(steps);
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = accordionHtml.trim();
    const newAccordion = tempDiv.firstChild;

    // Apply the open class if needed
    if (wasOpen) {
        newAccordion.classList.add('open');
    }

    // Append accordion inside the bubble
    bubble.appendChild(newAccordion);
    
    initIcons();
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Automatically collapse the thought accordion
function collapseAssistantThoughtTrace(id) {
    const bubble = document.getElementById(`${id}-bubble`);
    if (!bubble) return;
    const accordion = bubble.querySelector('.thought-accordion');
    if (accordion && accordion.classList.contains('open')) {
        accordion.classList.remove('open');
        initIcons();
    }
}

// Remove streaming cursor and active progress item at the end of stream
function removeStreamingCursor(id) {
    const bubble = document.getElementById(`${id}-bubble`);
    if (!bubble) return;
    
    const cursor = bubble.querySelector('.streaming-cursor');
    if (cursor) cursor.remove();

    const activeItem = document.getElementById(`${id}-active-item`);
    if (activeItem) activeItem.remove();

    const checklist = document.getElementById(`${id}-checklist`);
    if (checklist && checklist.children.length === 0) {
        checklist.remove();
    }
}

// Append bubble to chat console
function appendMessage(text, role, thoughtSteps = []) {
    const messageContainer = document.createElement('div');
    messageContainer.className = `message ${role}`;

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const metaText = role === 'user' ? 'You' : 'Agent';

    let bubbleContent = '';
    if (role === 'assistant') {
        // Parse markdown text using Marked JS
        const parsedText = window.marked ? window.marked.parse(text) : escapeHtml(text);
        bubbleContent = `
            <div class="message-bubble markdown-body">
                ${parsedText}
                ${thoughtSteps && thoughtSteps.length > 0 ? renderThoughtSteps(thoughtSteps) : ''}
            </div>
        `;
    } else {
        bubbleContent = `
            <div class="message-bubble">
                ${escapeHtml(text)}
            </div>
        `;
    }

    messageContainer.innerHTML = `
        ${bubbleContent}
        <div class="message-meta">${metaText} • ${timestamp}</div>
    `;

    chatBox.appendChild(messageContainer);
    chatBox.scrollTop = chatBox.scrollHeight;
    initIcons();
}

// Render Collapsible Accordion for Thought Traces
function renderThoughtSteps(steps) {
    const accordionId = `accordion-${Math.random().toString(36).substr(2, 9)}`;
    const totalTokens = steps.reduce((sum, step) => sum + (step.token || 0), 0);
    let stepsHtml = '';

    steps.forEach(step => {
        let toolCallsHtml = '';
        if (step.tool_calls && step.tool_calls.length > 0) {
            step.tool_calls.forEach(tc => {
                const argsStr = typeof tc.arguments === 'object' ? JSON.stringify(tc.arguments) : tc.arguments;
                toolCallsHtml += `
                    <div class="step-tool">
                        <div class="tool-name"><i data-lucide="wrench" style="width: 12px; height: 12px; display: inline; vertical-align: middle;"></i> Calling Skill: ${tc.name}</div>
                        <div class="tool-args"><b>Args:</b> ${escapeHtml(argsStr)}</div>
                        <div class="tool-result"><b>Result:</b>\n${escapeHtml(tc.result || '')}</div>
                    </div>
                `;
            });
        }

        stepsHtml += `
            <div class="thought-step">
                <div class="step-label">▶ Loop Step ${step.loop_index + 1} (${step.token || 0} tokens)</div>
                ${step.thought ? `<div class="step-thought">${escapeHtml(step.thought)}</div>` : ''}
                ${toolCallsHtml}
            </div>
        `;
    });

    return `
        <div class="thought-accordion" id="${accordionId}">
            <div class="thought-header" onclick="toggleAccordion('${accordionId}')">
                <span><i data-lucide="eye" style="width: 14px; height: 14px; display: inline; vertical-align: middle; margin-right: 4px;"></i> View Agent Reasoning (${steps.length} steps | ${totalTokens} tokens)</span>
                <i data-lucide="chevron-down" class="thought-header-icon" style="width: 14px; height: 14px;"></i>
            </div>
            <div class="thought-content">
                ${stepsHtml}
            </div>
        </div>
    `;
}

// Toggle accordion open/close
window.toggleAccordion = function(accordionId) {
    const el = document.getElementById(accordionId);
    el.classList.toggle('open');
};

// HTML Escaping to prevent XSS and formatting issues
function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return String(unsafe)
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

// Append Loading indicator bubble
function appendLoadingIndicator() {
    const id = `loading-${Math.random().toString(36).substr(2, 9)}`;
    const messageContainer = document.createElement('div');
    messageContainer.className = 'message assistant';
    messageContainer.id = id;

    messageContainer.innerHTML = `
        <div class="message-bubble" style="display: flex; gap: 8px; align-items: center; padding: 0.75rem 1.25rem;">
            <span>Agent is thinking</span>
            <span style="display: flex; gap: 3px;">
                <span style="width: 6px; height: 6px; background-color: var(--accent-cyan); border-radius: 50%; animation: pulse-glow 1s infinite alternate;"></span>
                <span style="width: 6px; height: 6px; background-color: var(--accent-cyan); border-radius: 50%; animation: pulse-glow 1s infinite alternate; animation-delay: 0.2s;"></span>
                <span style="width: 6px; height: 6px; background-color: var(--accent-cyan); border-radius: 50%; animation: pulse-glow 1s infinite alternate; animation-delay: 0.4s;"></span>
            </span>
        </div>
    `;
    chatBox.appendChild(messageContainer);
    chatBox.scrollTop = chatBox.scrollHeight;
    return id;
}

// Remove Loading indicator
function removeLoadingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// Clear Chat History
clearChatBtn.onclick = () => {
    if (!confirm("Clear this conversation history?")) return;
    if (currentSessionId) {
        deleteSession(currentSessionId);
    } else {
        renderWelcomeMessage();
    }
};

// Initial Setup on load
window.onload = () => {
    fetchDocuments();
    fetchSessions();
};
