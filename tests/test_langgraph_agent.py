import uuid
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.agent.graph.core import AgentGraph, create_agent_state_graph
from app.services.agent.graph.nodes import think_node, _get_val, _update_state
from _legacy.agent import RawAgentGraph


def test_langgraph_compilation():
    graph = create_agent_state_graph()
    assert graph is not None


def test_langgraph_agent_init():
    agent = AgentGraph(max_loops=5)
    assert agent.max_loops == 5
    assert agent.graph is not None


def test_legacy_raw_agent_import():
    raw_agent = RawAgentGraph(max_loops=5)
    assert raw_agent.max_loops == 5


def test_state_helper_functions():
    dict_state = {"messages": [{"role": "user", "content": "hi"}], "loop_count": 1}
    assert _get_val(dict_state, "messages") == [{"role": "user", "content": "hi"}]
    assert _get_val(dict_state, "loop_count") == 1
    
    updated = _update_state(dict_state, loop_count=2)
    assert updated["loop_count"] == 2
    assert dict_state["loop_count"] == 1  # immutability check


def test_build_init_message_with_mem_context():
    agent = AgentGraph()
    doc_id = uuid.uuid4()
    mem_context = {
        "user_profile": "User profile: prefers concise summaries.",
        "doc_memory": "Doc memory: key findings include transformer efficiency."
    }

    messages = agent._build_init_message(
        question="What are the key findings?",
        chat_history=None,
        mem_context=mem_context,
        document_id=doc_id
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    system_content = messages[0]["content"]

    assert "User profile: prefers concise summaries." in system_content
    assert "Doc memory: key findings include transformer efficiency." in system_content
    assert f"Document ID: {doc_id}" in system_content
    assert messages[1] == {"role": "user", "content": "What are the key findings?"}
