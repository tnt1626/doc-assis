#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
docker compose exec -T db psql -U user -d docqa \
  -c "TRUNCATE chat_history, per_doc_memory, sessions CASCADE;"
rm -f .agent/USER.md
echo "reset done"