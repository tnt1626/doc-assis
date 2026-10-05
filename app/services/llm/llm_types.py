from enum import Enum
from dataclasses import dataclass

@dataclass
class TextDelta:
    text: str

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total(self):
        return self.prompt_tokens + self.completion_tokens

class LLMPurpose(str, Enum):
    AGENT               = "agent"
    MEMORY_RETRIEVE     = "memory_retrieve"
    MEMORY_SUMMARIZE    = "memory_summarize"
