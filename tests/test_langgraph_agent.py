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
