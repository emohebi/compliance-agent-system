# AWS Bedrock Compliance Agent System

A comprehensive multi-agent system built with Strands Agents SDK and AWS Bedrock for document management and compliance checking.

## Features

- **Knowledge Base Agent**: Retrieves and manages documents from AWS Bedrock Knowledge Base
- **Compliance Agent**: Checks documents for compliance violations (PII, security, regulatory)
- **Orchestrator Agent**: Coordinates multi-agent workflows
- **MCP Server Integration**: Modular tool servers using Model Context Protocol
- **Batch Processing**: Handle multiple documents simultaneously
- **Compliance Reporting**: Generate detailed compliance reports
- **Remediation Suggestions**: Provide actionable recommendations

## Prerequisites

- Python 3.10+
- AWS Account with Bedrock access
- AWS Bedrock Knowledge Base configured
- Claude 4 Sonnet model access in Bedrock

## Quick Start

1. Clone the repository and navigate to the project:
```bash
cd compliance-agent-system
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your AWS credentials and configuration
```

5. Run examples:
```bash
python examples/basic_usage.py
python examples/mcp_usage.py
```

## Project Structure

```
compliance-agent-system/
├── config/           # Configuration files
├── agents/           # Agent implementations
├── tools/            # Reusable tools
├── models/           # Data models
├── utils/            # Utilities
├── mcp_servers/      # MCP server implementations
├── examples/         # Usage examples
└── tests/            # Test files
```

## Documentation

For detailed documentation, see the individual module files and examples.

## License

MIT License
