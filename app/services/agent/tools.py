from sqlalchemy.ext.asyncio import AsyncSession
from app.mcp.server import mcp_server

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": (
                "Search for relevant text chunks in a specific document based on the user's query. "
                "Use this tool when the user asks about the content of an uploaded document."
                "Use for targeted lookups; prefer get_full_document only when user explicitly asks to read the entire file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The UUID of the document to search within.",
                    },
                    "query": {
                        "type": "string",
                        "description": "The query or search term to look for in the document.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "The number of relevant text chunks to return. Defaults to 5.",
                        "default": 5,
                    },
                },
                "required": ["document_id", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "List all documents that have been uploaded to the system. "
                "Use this tool when the user wants to know what documents are available "
                "or needs to find a document_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "The maximum number of documents to return. Defaults to 20.",
                        "default": 20,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_document",
            "description": (
                "Retrieve the full content of a document page by page. Use page parameter to navigate."
                "Use this tool when you need to read the entire document text instead of "
                "just searching for relevant chunks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The UUID of the document to retrieve the full content for.",
                    },
                    "page": {
                        "type": "integer", 
                        "default": 1, 
                        "description": "Page number to read (each page is ~4000 chars)"
                    }
                },
                "required": ["document_id", "page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": (
                "Summarization of a document including metadata: status, filename, created_at, updated_at and first chunk of content."
                "Use this tool when you need to read the summarization of a document"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The UUID of the document to get the summarization.",
                    },
                },
                "required": ["document_id"],
            },
        },
    },
]


async def execute_tool(
    tool_name: str,
    tool_input: dict,
    db: AsyncSession = None,
) -> str:
    """Execute a tool via MCP Server and return the result as a string."""
    try:
        call_res = await mcp_server.call_tool(tool_name, tool_input)
        if call_res.content and hasattr(call_res.content[0], "text"):
            return call_res.content[0].text
        elif call_res.structured_content and "result" in call_res.structured_content:
            return str(call_res.structured_content["result"])
        return str(call_res)
    except Exception as e:
        return f"Error executing tool '{tool_name}': {e}"