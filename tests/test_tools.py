import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.agent.tools import execute_tool, TOOLS


def test_tools_schema_definitions():
    """Verify that TOOLS list contains valid OpenAI tool definitions."""
    tool_names = [t["function"]["name"] for t in TOOLS]
    assert "search_document" in tool_names
    assert "list_documents" in tool_names
    assert "get_full_document" in tool_names
    assert "summarize_document" in tool_names


@pytest.mark.asyncio
@patch("app.services.agent.tools.mcp_server.call_tool")
async def test_execute_tool_success(mock_call_tool):
    """Test successful tool execution dispatching to mcp_server."""
    mock_res = MagicMock()
    mock_item = MagicMock()
    mock_item.text = "Sample tool result content"
    mock_res.content = [mock_item]
    mock_call_tool.return_value = mock_res

    res = await execute_tool("search_document", {"query": "test"})
    assert res == "Sample tool result content"
    mock_call_tool.assert_called_once_with("search_document", {"query": "test"})


@pytest.mark.asyncio
@patch("app.services.agent.tools.mcp_server.call_tool")
async def test_execute_tool_error_handling(mock_call_tool):
    """Test error handling when mcp_server raises an exception."""
    mock_call_tool.side_effect = Exception("MCP Connection Failed")

    res = await execute_tool("list_documents", {})
    assert "Error executing tool 'list_documents': MCP Connection Failed" in res
