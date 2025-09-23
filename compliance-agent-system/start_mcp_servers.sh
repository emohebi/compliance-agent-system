#!/bin/bash
# Script to start MCP servers

echo "Starting MCP servers..."

# Start compliance server in background
python mcp_servers/compliance_mcp_server.py &
COMPLIANCE_PID=$!
echo "Started Compliance MCP Server (PID: $COMPLIANCE_PID)"

# Start KB server in background
python mcp_servers/kb_mcp_server.py &
KB_PID=$!
echo "Started Knowledge Base MCP Server (PID: $KB_PID)"

echo "MCP Servers are running. Press Ctrl+C to stop."

# Wait for interrupt
trap "kill $COMPLIANCE_PID $KB_PID; exit" INT
wait
