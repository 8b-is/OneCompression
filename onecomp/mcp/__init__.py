"""MCP (Model Context Protocol) server and autonomous compression agent.

Exposes OneComp's memory estimation, recipe recommendation,
quantisation, and evaluation as MCP tools over the stdio transport,
plus a goal-driven agent that searches for the best compression recipe.

Classes:
    MCPServer: Newline-delimited JSON-RPC 2.0 server (stdio transport).
    ToolDefinition: Tool metadata and handler.
    AutoCompressConfig: Goal definition for the compression search.
    AutoCompressionAgent: Goal-driven recipe search loop.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

from .protocol import MCPServer, ToolDefinition
from .agent import AutoCompressConfig, AutoCompressionAgent, AutoCompressResult, TrialResult
from .tools import build_tools
