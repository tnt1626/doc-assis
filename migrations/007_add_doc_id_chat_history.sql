ALTER TABLE chat_history
ADD COLUMN document_id UUID REFERENCES documents(id) NULL;

DELETE FROM sessions
WHERE is_deleted = True;

ALTER TABLE sessions
DROP COLUMN is_deleted;