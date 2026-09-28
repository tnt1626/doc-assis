ALTER TABLE sessions
DROP COLUMN is_consolidated;

ALTER TABLE chat_history
ADD COLUMN is_consolidated BOOLEAN DEFAULT FALSE NOT NULL;

CREATE INDEX idx_chat_history_consolidated 
ON chat_history(session_id, is_consolidated);