ALTER TABLE sessions
ADD COLUMN is_consolidated BOOLEAN DEFAULT FALSE NOT NULL,
ADD COLUMN is_deleted BOOLEAN DEFAULT FALSE NOT NULL;

CREATE TABLE per_doc_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE per_session_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    content TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_doc_memory ON per_doc_memory(document_id);
CREATE INDEX idx_session_memory ON per_session_memory(session_id);
