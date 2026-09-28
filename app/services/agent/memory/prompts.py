SHORT_TERM_SUMMARY_PROMPT = """
You are summarizing an ongoing conversation between a user and a document assistant.

Compress the exchanges below into a concise summary that preserves:
- The main questions the user asked
- Key answers and insights the assistant provided
- Any important context or decisions made
- Documents or sections that were discussed

The summary will be used as context for the next messages in this conversation.
Keep it dense but readable. Do not include tool calls or system messages.

Exchanges:
{exchanges}
"""

SESSION_CONSOLIDATE_PROMPT = """
You are extracting a one-sentence episode summary from a completed conversation between a user and a document assistant.

The summary should capture what happened in this session in a way that would be useful to recall in a future session.
Focus on: what the user was trying to accomplish, which documents were involved, and what was concluded or left open.

Write ONE sentence only. No preamble, no explanation.

Exchanges:
{exchanges}
"""

DOC_UPDATE_PROMPT = """
You are updating the memory record for a specific document based on a recent conversation.

Current memory for this document:
{current_memory}

Recent conversation:
{exchanges}

Update the memory to reflect what is now known. Include:
- Reading progress (which sections or topics have been covered)
- Key insights the user has gained
- Open questions the user still has about this document
- Connections to other documents or topics the user mentioned

Write in second person ("You have read...", "You are curious about...").
Be concise. Preserve important information from the current memory unless it is superseded.
"""

USER_PROFILE_UPDATE_PROMPT = """
You are updating the intellectual profile of a user based on a recent conversation with a document assistant.

Current profile:
{current_profile}

Recent conversation:
{exchanges}

Update the profile to reflect what you now know about this user. Include:
- Domain background and expertise level
- Research interests and current focus
- Reading and thinking style (do they prefer big picture or details first, theory or application)
- Types of questions they tend to ask
- How they connect ideas across documents

Write in second person ("You are a researcher who...", "You tend to approach...").
Be concise. Only update what has changed or been newly revealed. Preserve the rest.
"""


RETRIEVAL_GATE_PROMPT = """
You are a memory retrieval gate for a document research assistant.
Given the user's message, decide which types of long-term memory are needed to answer well.

There are three memory types:
- user_profile: facts about the user's background, research interests, reading style, and thinking patterns
- doc_memory: the user's progress, insights, and open questions about specific documents they have read
- session_memory: summaries of what happened in previous conversations

Reply with ONLY this JSON, nothing else:
{{
    "retrieve_user": true/false,
    "retrieve_doc": true/false,
    "retrieve_session": true/false,
    "query": "<search keywords if any retrieval is true, else empty string>",
    "reason": "<5 words max>"
}}

Guidelines:
- retrieve_user → true when the question relates to the user's preferences, background, or research direction
- retrieve_doc → true when the question references a document the user has read before or asks about prior reading progress
- retrieve_session → true when the question needs context from previous conversations ("last time", "before", "previously")
- General questions, math, or self-contained requests → all false
- When in doubt, retrieve — a stale memory beats a lost one

User message: {message}
"""