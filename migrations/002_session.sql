CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    message_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(20),      -- 'user' | 'assistant' | 'tool'
    type VARCHAR(20),      -- 'message' | 'tool_call' | 'tool_result' | 'thinking'
    content JSONB,
    token_count INT DEFAULT 0 NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_chat_history_session 
ON chat_history(session_id, created_at);