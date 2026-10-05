ALTER TABLE chat_history
DROP CONSTRAINT chat_history_document_id_fkey;

ALTER TABLE chat_history
ADD CONSTRAINT chat_history_document_id_fkey
FOREIGN KEY (document_id)
REFERENCES documents(id)
ON DELETE SET NULL;

ALTER TABLE per_doc_memory
DROP CONSTRAINT per_doc_memory_document_id_fkey;

ALTER TABLE per_doc_memory
ADD CONSTRAINT per_doc_memory_document_id_fkey
FOREIGN KEY (document_id)
REFERENCES documents(id)
ON DELETE CASCADE;