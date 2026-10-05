# Doc Assistant

Upload your documents. Ask anything. See exactly how answers are derived.

Most AI chat tools are black boxes — you get an answer but can't verify where it came from. Doc Assistant shows you every reasoning step and source chunk in a clean, reading-focused interface.

---

## Demo

### ☀️ Transparent Agent Reasoning & Thought Traces (Default Light Mode)
![Light Mode with Reasoning](attach/2/demo_2.png)

<br>

| ☀️ Minimalist Chat View | 🌙 Dark Charcoal Mode |
| :---: | :---: |
| <img src="attach/2/demo_1.png" width="100%" alt="Light Mode Chat" /> | <img src="attach/2/demo_3.png" width="100%" alt="Dark Mode with Reasoning" /> |

---

## Usage Guide

### 1. Setup and Installation

#### Step 1: Start Database
Start PostgreSQL with vector extension enabled:
```bash
docker compose up -d db
```

#### Step 2: Environment Configuration
Create a `.env` file in the root directory (refer to `.env.example`):
```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/docqa

# Choose LLM Provider: 'ollama' (default, local & private) or 'groq' (cloud, high speed)
LLM_PROVIDER=ollama

# Ollama settings (required when LLM_PROVIDER=ollama)
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL_NAME=nomic-embed-text

# Groq settings (only required when LLM_PROVIDER=groq)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_SMALL_MODEL=llama-3.1-8b-instant
```

> **Ollama Setup & System Notes:**
> - **Hardware Requirements:** Running `qwen2.5:7b` locally requires at least ~5GB VRAM.
> - **Context Window:** Configure your Ollama model context length (recommended: `num_ctx = 8192`) on the Ollama server side via `Modelfile`.

#### Step 3: Run the Application
Install dependencies and start the backend server:
```bash
uv sync
uv run uvicorn app.main:app --reload
```

#### Step 4: Access Web Interface
Open your browser and navigate to:
```text
http://localhost:8000/ui
```

---

### 2. Dashboard Operations

#### Chat Sessions
- Create new session: Click the "+ New" button on the left sidebar.
- Auto-titling: Sessions are automatically named based on your first query.
- Rename session: Click the edit icon next to a session in the sidebar to rename it.
- Delete session: Click the trash icon to remove a conversation session and its history.

#### Document Upload
- Drag and drop files or click the upload area in the sidebar.
- Supported formats: `.pdf`, `.txt`, `.md`, `.html`, `.css`, `.csv`, `.xml`, `.json`.
- Status indicators show processing state: `Ready`, `Processing`, or `Failed`.

#### Querying & Agent Reasoning
- Scope Selection: Select "Search All Documents" or restrict queries to a specific document.
- Thought Traces: Click "View Agent Reasoning" under responses to inspect step-by-step reasoning logs, skill execution, and token usage.
- Dual Theme: Toggle between Light mode and Dark mode using the header theme button.

---

## Key Features

- **Reading-Focused Design:** Built with clear typography and constrained reading columns for effortless document navigation.
- **Instant Knowledge Retrieval:** Search and extract verified answers across internal documentation (SOPs, guides, reports, and contracts).
- **Transparent Reasoning Traces:** Provide step-by-step auditability into how answers are derived, ensuring verified and trustworthy responses.
- **Interactive Conversation Management:** Support multi-turn chat sessions with automatic topic titling and persistent conversation history.
- **Real-Time Streaming & Auto-scroll:** Receive immediate answer streaming with seamless auto-scrolling as insights are synthesized.
- **System Health Monitoring:** Live `/health` endpoint checking DB connection and Ollama service status.

---

## Roadmap

- [ ] Multi-document comparison & diff visualization
- [ ] Structured data extraction to exportable reports
- [ ] PDF scan support via OCR

Have a use case in mind? Open an issue — I'm actively looking for real-world problems to solve with this tool.
