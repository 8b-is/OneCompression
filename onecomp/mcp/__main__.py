"""Entry point for the OneComp MCP server: ``python -m onecomp.mcp``.

Starts an MCP server on stdio (newline-delimited JSON-RPC 2.0) exposing
the OneComp tools: estimate_memory, recommend_recipe, quantize, and
evaluate.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""


def main():
    """Run the MCP server on stdio until stdin is closed."""
    # Lazy imports keep server startup responsive and avoid touching
    # CUDA before the client has connected.
    from .protocol import MCPServer  # pylint: disable=import-outside-toplevel
    from .compaction_tools import close_all_sessions  # pylint: disable=import-outside-toplevel
    from .tools import build_tools  # pylint: disable=import-outside-toplevel

    server = MCPServer(tools=build_tools())
    try:
        server.serve_forever()
    finally:
        close_all_sessions()


if __name__ == "__main__":
    main()
