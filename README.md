# Doc Assistant

Upload your documents. Ask anything. See exactly how answers are derived.

Most AI chat tools are black boxes: you get an answer but can't verify where it came from. Doc Assistant shows every reasoning step and source chunk in a clean, reading-focused interface. The agent loop is written from scratch (no LangGraph), uses a three-tier memory system, exposes tools over MCP, and runs on a local Ollama model or on Groq.

This repo also contains a small evaluation of one question: [does the memory system actually help?](#memory-evaluation)

---

## Demo

### ☀️ Transparent Agent Reasoning & Thought Traces (Default Light Mode)
![Light Mode with Reasoning](attach/2/demo_2.png)

<br>

| ☀️ Minimalist Chat View | 🌙 Dark Charcoal Mode |
| :---: | :---: |
| <img src="attach/2/demo_1.png" width="100%" alt="Light Mode Chat" /> | <img src="attach/2/demo_3.png" width="100%" alt="Dark Mode with Reasoning" /> |

---

## Setup

### 1. Start the database
PostgreSQL with the pgvector extension:
```bash
docker compose up -d db
```

### 2. Configure the environment
Create a `.env` file in the root directory (see `.env.example`):
```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/docqa

# LLM provider: 'ollama' (default, local & private) or 'groq' (cloud, fast)
LLM_PROVIDER=ollama

# Long-term user memory on/off (read once at server startup)
MEMORY_ENABLED=true

# Ollama (required when LLM_PROVIDER=ollama)
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL_NAME=nomic-embed-text

# Groq (required when LLM_PROVIDER=groq)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=<model id>   # the benchmark below used gpt-oss-120b
GROQ_SMALL_MODEL=llama-3.1-8b-instant
```

> **Ollama notes:** running `qwen2.5:7b` locally needs about 5 GB of VRAM. Set the context length to 8192 on the Ollama server side (the benchmark used `OLLAMA_CONTEXT_LENGTH=8192` in the systemd unit, or `num_ctx = 8192` in a `Modelfile`).

### 3. Run
```bash
uv sync
uv run uvicorn app.main:app --reload
```
Open `http://localhost:8000/ui`.

---

## Memory evaluation

### Research question

> Does adding long-term user memory (a profile consolidated from past sessions) improve an agent's ability to recall, update, and admit not knowing facts about the user, and what does it cost in tokens and latency?

The same benchmark was run on two models, because the answer depends heavily on the model:

| Provider | Model |
|---|---|
| Ollama (local, RTX 4060 8 GB) | `qwen2.5:7b`, context 8192 |
| Groq | `gpt-oss-120b` |

### Method

Six scenarios, each repeated 3 times per configuration (4 configurations, 72 runs total). Before every run the memory state is reset (`evals/reset.sh`: truncates chat history, per-document memory and sessions, deletes `.agent/USER.md`; uploaded documents are kept).

Each scenario:

1. **Setup:** open a session, send context messages, then **delete the session** (this triggers memory consolidation).
2. **Test:** open a fresh session and ask one question.
3. **Grade** the answer with keyword rules, then read ambiguous cases by hand.

| Scenario | Kind | Setup | Question |
|---|---|---|---|
| `recall_exam` | recall | "Preparing for a Databases exam on Friday" | What am I studying for? |
| `recall_name` | recall | "Call me Tâm, I like short answers" | What should you call me? |
| `recall_year` | recall | "Third-year Computer Science student" | What year am I in? |
| `recall_project` | recall | "My project is a personal expense tracker app" | What is my project about? |
| `update_exam` | update | Databases exam, then "I changed it to Computer Networks" | What am I studying for? |
| `unknown_exam` | unknown | none | What am I studying for? |

`unknown_exam` is a control: with no information, the correct answer is "I don't know", with or without memory. With memory off, the recall and update scenarios are **expected** to fail; that is a correct result, not a bug.

The benchmark is a mock browser client that calls the same REST endpoints as the web UI. `MEMORY_ENABLED` and `LLM_PROVIDER` are read at server startup, so each configuration needs its own server launch.

### Results

Correct answers out of 18 runs per configuration (6 scenarios x 3 repetitions):

| Configuration | Correct | Language slips (CJK) | Final-question latency | Session delete (incl. consolidation) |
|---|---|---|---|---|
| Groq, memory off | 2 / 18 | 0 | 3.2 s | 0.01 s |
| Groq, memory on | **10 / 18** | 0 | 9.6 s | 0.75 s |
| Ollama 7b, memory off | 3 / 18 | 2 | 1.0 s | 0.01 s |
| Ollama 7b, memory on | 4 / 18 | 6 | 2.2 s | 3.2 s |

By scenario kind (correct runs):

| Kind | Groq off | Groq on | Ollama off | Ollama on |
|---|---|---|---|---|
| recall (12 runs) | 0 | 6 | 0 | 2 |
| update (3 runs) | 0 | 2 | 0 | 0 |
| unknown (3 runs) | 2 | 2 | 3 | 2 |

Token cost, measured on `recall_project` only (3 runs per configuration, prompt + completion):

| Configuration | Agent | Memory retrieve + summarize | Total | vs memory off |
|---|---|---|---|---|
| Groq off | ~5,030 | n/a | ~5,030 | baseline |
| Groq on | ~6,800 | ~1,540 | ~8,340 | +66% |
| Ollama off | ~1,550 | n/a | ~1,550 | baseline |
| Ollama on | ~1,760 | ~1,010 | ~2,770 | +79% |

**Reading the results**

- **Groq 120B: memory helps clearly.** Correct answers went from 2/18 to 10/18. The two memory-off successes are both the `unknown_exam` control. With memory on, the agent recalled the exam subject, name and year in 6 of 9 runs and handled the exam update in 2 of 3 (the miss is failure case 1). The price is roughly 3x slower final answers and about +66% tokens.
- **Ollama 7b: no measurable difference (4/18 vs 3/18).** With 3 runs per cell, a one-run gap is noise. Memory is written correctly (see failure case 1), but the small model rarely uses it, and it drifts into Chinese in 6 of 18 runs with memory on (2 of 18 with it off).
- **The `unknown_exam` control showed no memory effect** in either model. Its failures have different causes, described below.

### Failure cases

#### 1. The profile is correct, but the agent does not use it

In `recall_project`, consolidation worked in every memory-on run: the stored profile named the right topic (for example "developing a personal expense management application"). The question "What is my project about?" still failed:

- **Groq, 0/3:** the agent searched the uploaded paper and described the paper's topic (an agentic travel-planning system) instead of the user's project.
- **Ollama 7b, 1/3:** in two runs the agent treated "project" as a document and asked for a `document_id`; one of those replies switched to Chinese.

The same pattern appeared once in `update_exam` (Groq, memory on, 1 of 3 runs). The profile of that run correctly said the user's focus had shifted from Database Fundamentals to Computer Networks, yet the agent described the paper and guessed the user was probably studying AI or machine learning. The other two runs answered Computer Networks and noted the switch from Databases.

Why: words like "project" or "studying" pull the agent toward document search, and nothing in the prompt makes the user profile take priority over the document tools. This is a retrieval-routing problem, not a memory-writing problem.

My first version of `recall_project` used "a document Q&A system" as the topic, which overlapped with the uploaded paper and produced false passes. Switching to an unrelated topic exposed the real behavior above.

#### 2. Inventing user facts from document context (`unknown_exam`)

With an empty profile, "What am I studying for?" should get "I don't know". In 2 of 6 Groq runs (one with memory off, one on), the agent instead stated that the user is studying Artificial Intelligence, inferred from the paper stored in the database. The other four Groq runs correctly said they had no information.

Separately, in one Ollama 7b run with memory on, the consolidation step added details the user never gave: the profile mentioned "predictive models for expenditure forecasting" and the answer repeated them. This suggests summary-style memory can introduce unsupported details. I saw it once and did not measure its frequency.

### Known limitations

- **Small sample:** 3 runs per scenario per configuration. Differences of one run are not meaningful; only the large Groq effect is clearly outside noise.
- **Keyword grading:** answers were graded by keyword rules plus hand reading of ambiguous cases, by one person. The first keyword grader gave both false passes and false failures, so all results were regraded.
- **Token measurement:** token data comes from `recall_project` only (3 runs per configuration), not the full suite. The log parser had to be fixed after the first runs because the log formatter wrapped lines. Treat the figures as estimates.
- **Retried runs:** Groq occasionally closed streams mid-response (`RemoteProtocolError`). 3 affected runs were re-executed and replaced; original files are kept unchanged.
- **Confounder:** the uploaded paper stays in the database during all runs. This is deliberate (it reflects real usage), but it causes the document-over-profile failures above.
- **Not covered:** memory beyond short facts, long conversations, history compression near the context limit, and English search queries over an English paper (Vietnamese queries missed in earlier manual tests).
- **One model per provider, one prompt per scenario.** No conclusion about other models.

### Reproduce

```bash
# Terminal 1: start the server with the configuration under test
LLM_PROVIDER=groq MEMORY_ENABLED=true \
  uv run uvicorn app.main:app 2>&1 | tee evals/logs/groq-on.txt

# Terminal 2: run the benchmark (the label is only a tag; it does not switch config)
uv run python evals/bench_memory.py --label groq-on \
  --log evals/logs/groq-on.txt --reps 3 --pause 5
```

Raw per-run results go to `evals/results/<label>.jsonl`. Re-grade saved answers without calling any model:

```bash
uv run python evals/regrade.py evals/results/groq-off.jsonl evals/results/groq-on.jsonl \
  evals/results/ollama-off.jsonl evals/results/ollama-on.jsonl
```

Pass the four original files explicitly; the default glob would also pick up rerun and regraded files. Use `--rerun` to replace failed runs and `--replace-scenario` to replace a rewritten scenario. Before letting a run finish, check that the server log's `Memory enabled: True/False` line matches the label.