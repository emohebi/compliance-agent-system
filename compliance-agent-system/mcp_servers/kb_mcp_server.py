#!/usr/bin/env python3
"""MCP Server for AWS Bedrock Knowledge Base operations."""

import json
import boto3
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("knowledge-base-mcp-server")

# AWS Bedrock client
bedrock_client = boto3.client(
    'bedrock-agent-runtime',
    region_name=os.getenv("AWS_REGION", "us-west-2")
)

KB_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available knowledge base tools."""
    return [
        types.Tool(
            name="retrieve_documents",
            description="Retrieve documents from AWS Bedrock Knowledge Base using semantic search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for retrieving documents"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum relevance score (0-1)",
                        "default": 0.4,
                        "minimum": 0,
                        "maximum": 1
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="search_by_metadata",
            description="Search documents by metadata filters",
            inputSchema={
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "description": "Metadata filters to apply"
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional text query"
                    }
                },
                "required": ["filters"]
            }
        ),
        types.Tool(
            name="get_document_summary",
            description="Get a summary of documents in the knowledge base",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_stats": {
                        "type": "boolean",
                        "description": "Include statistics in summary",
                        "default": True
                    }
                }
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, 
    arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution."""
    
    if name == "retrieve_documents":
        return await retrieve_documents(arguments)
    elif name == "search_by_metadata":
        return await search_by_metadata(arguments)
    elif name == "get_document_summary":
        return await get_document_summary(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def retrieve_documents(args: dict) -> list[types.TextContent]:
    """Retrieve documents from knowledge base."""
    query = args.get("query", "")
    max_results = args.get("max_results", 10)
    min_score = args.get("min_score", 0.4)
    
    try:
        response = bedrock_client.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': max_results,
                    'overrideSearchType': 'SEMANTIC'
                }
            }
        )
        
        # Filter and format results
        documents = []
        for result in response.get('retrievalResults', []):
            score = result.get('score', 0)
            if score >= min_score:
                documents.append({
                    'content': result.get('content', {}).get('text', ''),
                    'score': score,
                    'metadata': result.get('metadata', {}),
                    'location': result.get('location', {})
                })
        
        result = {
            'success': True,
            'query': query,
            'documents_found': len(documents),
            'documents': documents,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        result = {
            'success': False,
            'error': str(e),
            'query': query,
            'documents': []
        }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]

async def search_by_metadata(args: dict) -> list[types.TextContent]:
    """Search documents by metadata."""
    filters = args.get("filters", {})
    query = args.get("query", "")
    
    # This would require specific implementation based on your KB setup
    # For now, returning a placeholder
    result = {
        'success': True,
        'filters_applied': filters,
        'query': query,
        'message': 'Metadata search functionality requires specific KB configuration',
        'documents': []
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]

async def get_document_summary(args: dict) -> list[types.TextContent]:
    """Get summary of documents in knowledge base."""
    include_stats = args.get("include_stats", True)
    
    # This would query the KB for statistics
    # Placeholder implementation
    summary = {
        'knowledge_base_id': KB_ID,
        'status': 'active',
        'last_updated': datetime.now().isoformat()
    }
    
    if include_stats:
        summary['statistics'] = {
            'total_documents': 'N/A',
            'total_chunks': 'N/A',
            'last_sync': 'N/A'
        }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(summary, indent=2)
    )]

async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="knowledge-base-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())