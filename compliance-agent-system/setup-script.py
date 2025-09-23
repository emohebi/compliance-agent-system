#!/usr/bin/env python3
"""
Setup script to create the complete project structure for the Compliance Agent System.
Run this script to automatically create all directories and files with initial content.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

class ProjectSetup:
    """Setup the Compliance Agent System project structure."""
    
    def __init__(self, base_dir: str = "compliance-agent-system"):
        """Initialize project setup with base directory."""
        self.base_dir = Path(base_dir)
        
    def create_directory_structure(self):
        """Create all project directories."""
        directories = [
            "",  # Root directory
            "config",
            "agents",
            "tools",
            "models",
            "utils",
            "examples",
            "tests",
            "mcp_servers",
            "logs",
        ]
        
        for directory in directories:
            dir_path = self.base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
            
            # Create __init__.py for Python packages
            if directory and directory != "logs":
                init_file = dir_path / "__init__.py"
                init_file.touch()
                print(f"Created file: {init_file}")
    
    def create_requirements_txt(self):
        """Create requirements.txt file."""
        content = """strands-agents>=1.0.0
strands-agents-tools>=0.2.0
boto3>=1.34.0
pydantic>=2.0.0
python-dotenv>=1.0.0
typing-extensions>=4.8.0
colorama>=0.4.6
tqdm>=4.66.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
mcp>=0.1.0
fastmcp>=0.1.0
httpx>=0.24.0
uvicorn>=0.23.0
sse-starlette>=1.6.0
"""
        file_path = self.base_dir / "requirements.txt"
        file_path.write_text(content)
        print(f"Created file: {file_path}")
    
    def create_env_example(self):
        """Create .env.example file."""
        content = """# AWS Configuration
AWS_REGION=us-west-2
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_SESSION_TOKEN=your_session_token  # Optional for temporary credentials

# Bedrock Configuration
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0
BEDROCK_KNOWLEDGE_BASE_ID=your_kb_id
BEDROCK_CACHE_PROMPT=default

# Agent Configuration
AGENT_MAX_ITERATIONS=10
AGENT_TEMPERATURE=0.3
AGENT_MAX_TOKENS=4096

# Compliance Configuration
COMPLIANCE_THRESHOLD=0.8
COMPLIANCE_RULES_FILE=./config/compliance_rules.json

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/compliance_agent.log
"""
        file_path = self.base_dir / ".env.example"
        file_path.write_text(content)
        print(f"Created file: {file_path}")
    
    def create_readme(self):
        """Create README.md file."""
        content = """# AWS Bedrock Compliance Agent System

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
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
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
"""
        file_path = self.base_dir / "README.md"
        file_path.write_text(content)
        print(f"Created file: {file_path}")
    
    def create_setup_py(self):
        """Create setup.py file."""
        content = '''"""Setup configuration for the compliance agent system."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="compliance-agent-system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AWS Bedrock Compliance Agent System using Strands Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/compliance-agent-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "compliance-agent=examples.basic_usage:main",
        ],
    },
)
'''
        file_path = self.base_dir / "setup.py"
        file_path.write_text(content)
        print(f"Created file: {file_path}")
    
    def create_config_files(self):
        """Create configuration files."""
        # settings.py
        settings_content = '''"""Configuration settings for the compliance agent system."""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings."""
    
    # AWS Configuration
    AWS_REGION: str = os.getenv("AWS_REGION", "us-west-2")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN: Optional[str] = os.getenv("AWS_SESSION_TOKEN")
    
    # Bedrock Configuration
    BEDROCK_MODEL_ID: str = os.getenv(
        "BEDROCK_MODEL_ID", 
        "us.anthropic.claude-sonnet-4-20250514-v1:0"
    )
    BEDROCK_KNOWLEDGE_BASE_ID: str = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "")
    BEDROCK_CACHE_PROMPT: str = os.getenv("BEDROCK_CACHE_PROMPT", "default")
    
    # Agent Configuration
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
    AGENT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.3"))
    AGENT_MAX_TOKENS: int = int(os.getenv("AGENT_MAX_TOKENS", "4096"))
    
    # Compliance Configuration
    COMPLIANCE_THRESHOLD: float = float(os.getenv("COMPLIANCE_THRESHOLD", "0.8"))
    COMPLIANCE_RULES_FILE: str = os.getenv(
        "COMPLIANCE_RULES_FILE", 
        "./config/compliance_rules.json"
    )
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/compliance_agent.log")
    
    @classmethod
    def get_bedrock_config(cls) -> Dict[str, Any]:
        """Get Bedrock model configuration."""
        return {
            "model_id": cls.BEDROCK_MODEL_ID,
            "temperature": cls.AGENT_TEMPERATURE,
            "max_tokens": cls.AGENT_MAX_TOKENS,
            "cache_prompt": cls.BEDROCK_CACHE_PROMPT,
            "region_name": cls.AWS_REGION,
            "streaming": True
        }

settings = Settings()
'''
        file_path = self.base_dir / "config" / "settings.py"
        file_path.write_text(settings_content)
        print(f"Created file: {file_path}")
        
        # logging_config.py
        logging_content = '''"""Logging configuration for the compliance agent system."""

import logging
import sys
from pathlib import Path

def setup_logging():
    """Setup logging configuration."""
    # Import settings here to avoid circular import
    from config.settings import settings
    
    # Create logs directory if it doesn't exist
    log_file = Path(settings.LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Setup handlers
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    # Set specific loggers
    logging.getLogger("strands").setLevel(logging.INFO)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
'''
        file_path = self.base_dir / "config" / "logging_config.py"
        file_path.write_text(logging_content)
        print(f"Created file: {file_path}")
        
        # compliance_rules.json
        compliance_rules = {
            "rules": [
                {
                    "rule_id": "PII_001",
                    "name": "PII Detection",
                    "description": "Check for personally identifiable information",
                    "category": "data_protection",
                    "severity": "high",
                    "patterns": [
                        "\\b\\d{3}-\\d{2}-\\d{4}\\b",
                        "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",
                        "\\b(?:\\d{4}[-\\s]?){3}\\d{4}\\b"
                    ],
                    "enabled": True
                },
                {
                    "rule_id": "SEC_001",
                    "name": "Security Keywords",
                    "description": "Check for security-sensitive keywords",
                    "category": "security",
                    "severity": "medium",
                    "keywords": ["password", "api_key", "secret", "token", "credential"],
                    "enabled": True
                },
                {
                    "rule_id": "REG_001",
                    "name": "Regulatory Compliance",
                    "description": "Check for regulatory compliance keywords",
                    "category": "regulatory",
                    "severity": "high",
                    "keywords": ["gdpr", "hipaa", "sox", "pci-dss"],
                    "enabled": True
                }
            ]
        }
        file_path = self.base_dir / "config" / "compliance_rules.json"
        file_path.write_text(json.dumps(compliance_rules, indent=2))
        print(f"Created file: {file_path}")
        
        # mcp_config.json
        mcp_config = {
            "mcp_servers": {
                "compliance": {
                    "enabled": True,
                    "type": "stdio",
                    "command": "python",
                    "args": ["mcp_servers/compliance_mcp_server.py"],
                    "env": {},
                    "description": "Compliance checking and validation tools"
                },
                "knowledge_base": {
                    "enabled": True,
                    "type": "stdio",
                    "command": "python",
                    "args": ["mcp_servers/kb_mcp_server.py"],
                    "env": {
                        "AWS_REGION": "${AWS_REGION}",
                        "BEDROCK_KNOWLEDGE_BASE_ID": "${BEDROCK_KNOWLEDGE_BASE_ID}"
                    },
                    "description": "AWS Bedrock Knowledge Base operations"
                }
            },
            "workflows": {
                "full_compliance_audit": {
                    "description": "Complete compliance audit workflow",
                    "required_servers": ["compliance", "knowledge_base"],
                    "optional_servers": ["analytics", "notifications"],
                    "timeout": 300,
                    "retry_count": 3
                }
            }
        }
        file_path = self.base_dir / "config" / "mcp_config.json"
        file_path.write_text(json.dumps(mcp_config, indent=2))
        print(f"Created file: {file_path}")
    
    def create_placeholder_files(self):
        """Create placeholder files for main modules."""
        # Create placeholder content for each module
        placeholders = {
            # Models
            "models/compliance_rules.py": '''"""Compliance rule models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class ComplianceRule(BaseModel):
    """Model for a compliance rule."""
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    category: str = Field(..., description="Rule category")
    severity: str = Field("medium", description="Rule severity: low, medium, high")
    patterns: Optional[List[str]] = Field(None, description="Regex patterns to match")
    keywords: Optional[List[str]] = Field(None, description="Keywords to check")
    enabled: bool = Field(True, description="Whether rule is enabled")
    
class ComplianceResult(BaseModel):
    """Model for compliance check results."""
    is_compliant: bool = Field(..., description="Overall compliance status")
    score: float = Field(..., description="Compliance score (0-1)")
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    checked_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    document_metadata: Optional[Dict[str, Any]] = None
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_compliant': self.is_compliant,
            'score': self.score,
            'violations': self.violations,
            'warnings': self.warnings,
            'checked_at': self.checked_at,
            'document_metadata': self.document_metadata
        }
''',
            
            "models/document_models.py": '''"""Document data models."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

class Document(BaseModel):
    """Model for a document."""
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    source: Optional[str] = Field(None, description="Document source")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
class DocumentBatch(BaseModel):
    """Model for a batch of documents."""
    documents: List[Document]
    batch_id: str = Field(..., description="Batch ID")
    processing_status: str = Field("pending", description="Processing status")
''',
            
            # Tools placeholders
            "tools/knowledge_base_tools.py": '''"""Knowledge base tools for interacting with AWS Bedrock Knowledge Base."""

# Placeholder for knowledge base tools
# Full implementation available in the main documentation

from strands import tool

class KnowledgeBaseTools:
    """Tools for interacting with AWS Bedrock Knowledge Base."""
    
    def __init__(self):
        """Initialize the knowledge base tools."""
        pass
    
    @tool
    def retrieve_from_knowledge_base(self, query: str):
        """Retrieve documents from knowledge base."""
        return {"message": "Placeholder - implement KB retrieval"}

kb_tools = KnowledgeBaseTools()
''',
            
            "tools/compliance_tools.py": '''"""Compliance checking tools."""

# Placeholder for compliance tools
# Full implementation available in the main documentation

from strands import tool

class ComplianceTools:
    """Tools for compliance checking."""
    
    def __init__(self):
        """Initialize compliance tools."""
        pass
    
    @tool
    def check_document_compliance(self, document_content: str):
        """Check document for compliance."""
        return {"message": "Placeholder - implement compliance checking"}

compliance_tools = ComplianceTools()
''',
            
            "tools/notification_tools.py": '''"""Notification and alerting tools."""

from strands import tool
import logging

logger = logging.getLogger(__name__)

class NotificationTools:
    """Tools for notifications and alerts."""
    
    @tool
    def send_alert(self, message: str, severity: str = "info"):
        """Send an alert notification."""
        logger.info(f"Alert [{severity}]: {message}")
        return {"status": "sent", "message": message}

notification_tools = NotificationTools()
''',
            
            # Utils
            "utils/aws_helper.py": '''"""AWS service helper utilities."""

import boto3
from typing import Optional

def get_bedrock_client(region: str = "us-west-2"):
    """Get AWS Bedrock client."""
    return boto3.client('bedrock-runtime', region_name=region)

def get_s3_client(region: str = "us-west-2"):
    """Get AWS S3 client."""
    return boto3.client('s3', region_name=region)
''',
            
            "utils/validators.py": '''"""Data validation utilities."""

from typing import Any, Dict
import re

def validate_email(email: str) -> bool:
    """Validate email address."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_document(document: Dict[str, Any]) -> bool:
    """Validate document structure."""
    required_fields = ['content']
    return all(field in document for field in required_fields)
''',
            
            # Agent placeholders
            "agents/knowledge_base_agent.py": '''"""Knowledge Base Agent for reading and managing KB documents."""

# Placeholder for knowledge base agent
# Full implementation available in the main documentation

from strands import Agent
from strands.models import BedrockModel

class KnowledgeBaseAgent:
    """Agent for interacting with AWS Bedrock Knowledge Base."""
    
    def __init__(self):
        """Initialize the Knowledge Base agent."""
        self.model = BedrockModel()
        self.agent = Agent(model=self.model)
    
    def retrieve_documents(self, query: str):
        """Retrieve documents from the knowledge base."""
        return {"message": "Placeholder - implement document retrieval"}
''',
            
            "agents/compliance_agent.py": '''"""Compliance Agent for checking document compliance."""

# Placeholder for compliance agent
# Full implementation available in the main documentation

from strands import Agent
from strands.models import BedrockModel

class ComplianceAgent:
    """Agent for compliance checking and reporting."""
    
    def __init__(self):
        """Initialize the Compliance agent."""
        self.model = BedrockModel()
        self.agent = Agent(model=self.model)
    
    def check_document(self, document_content: str):
        """Check a document for compliance."""
        return {"message": "Placeholder - implement compliance checking"}
''',
            
            "agents/orchestrator_agent.py": '''"""Main orchestrator agent that coordinates KB and compliance agents."""

# Placeholder for orchestrator agent
# Full implementation available in the main documentation

from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.compliance_agent import ComplianceAgent

class OrchestratorAgent:
    """Main orchestrator agent that coordinates sub-agents."""
    
    def __init__(self):
        """Initialize the orchestrator with sub-agents."""
        self.kb_agent = KnowledgeBaseAgent()
        self.compliance_agent = ComplianceAgent()
    
    def process_compliance_workflow(self, query: str):
        """Process a complete workflow."""
        return {"message": "Placeholder - implement workflow processing"}
''',
            
            "agents/mcp_enhanced_agent.py": '''"""Enhanced agent using MCP servers for extended capabilities."""

# Placeholder for MCP enhanced agent
# Full implementation available in the main documentation

from strands import Agent
from strands.models import BedrockModel

class MCPEnhancedAgent:
    """Agent enhanced with MCP server capabilities."""
    
    def __init__(self):
        """Initialize the MCP-enhanced agent."""
        self.model = BedrockModel()
        self.agent = Agent(model=self.model)
    
    def process_document_compliance(self, query: str):
        """Process document retrieval and compliance checking."""
        return {"message": "Placeholder - implement MCP processing"}
''',
            
            "agents/multi_mcp_orchestrator.py": '''"""Orchestrator agent managing multiple MCP servers."""

# Placeholder for multi-MCP orchestrator
# Full implementation available in the main documentation

from strands import Agent
from strands.models import BedrockModel

class MultiMCPOrchestrator:
    """Orchestrator managing multiple MCP servers for complex workflows."""
    
    def __init__(self):
        """Initialize orchestrator with multiple MCP servers."""
        self.model = BedrockModel()
    
    def execute_workflow(self, workflow_name: str, parameters: dict):
        """Execute a predefined workflow."""
        return {"message": "Placeholder - implement workflow execution"}
''',
            
            # MCP Server placeholders
            "mcp_servers/compliance_mcp_server.py": '''#!/usr/bin/env python3
"""MCP Server for compliance checking operations."""

# Placeholder for compliance MCP server
# Full implementation available in the main documentation

print("Compliance MCP Server - Placeholder")
print("See documentation for full implementation")
''',
            
            "mcp_servers/kb_mcp_server.py": '''#!/usr/bin/env python3
"""MCP Server for AWS Bedrock Knowledge Base operations."""

# Placeholder for knowledge base MCP server
# Full implementation available in the main documentation

print("Knowledge Base MCP Server - Placeholder")
print("See documentation for full implementation")
''',
            
            "mcp_servers/analytics_mcp_server.py": '''#!/usr/bin/env python3
"""MCP Server for analytics operations."""

# Placeholder for analytics MCP server

print("Analytics MCP Server - Placeholder")
''',
            
            "mcp_servers/notification_mcp_server.py": '''#!/usr/bin/env python3
"""MCP Server for notification operations."""

# Placeholder for notification MCP server

print("Notification MCP Server - Placeholder")
''',
            
            # Examples
            "examples/basic_usage.py": '''"""Basic usage examples for the compliance agent system."""

def main():
    """Run basic examples."""
    print("=" * 60)
    print("Compliance Agent System - Basic Usage Examples")
    print("=" * 60)
    
    print("\\nExample 1: Simple Document Retrieval")
    print("Example 2: Compliance Check")
    print("Example 3: Orchestrated Workflow")
    print("\\nSee documentation for full implementation")

if __name__ == "__main__":
    main()
''',
            
            "examples/mcp_usage.py": '''"""Examples of using MCP-enhanced agents."""

def main():
    """Run MCP examples."""
    print("=" * 60)
    print("MCP Server Integration Examples")
    print("=" * 60)
    
    print("\\nExample 1: MCP Enhanced Agent")
    print("Example 2: Batch PII Scanning")
    print("Example 3: Multi-MCP Orchestration")
    print("\\nSee documentation for full implementation")

if __name__ == "__main__":
    main()
''',
            
            "examples/advanced_workflow.py": '''"""Advanced workflow examples."""

def main():
    """Run advanced workflow examples."""
    print("=" * 60)
    print("Advanced Workflow Examples")
    print("=" * 60)
    
    print("\\nAdvanced workflows coming soon...")
    print("See documentation for examples")

if __name__ == "__main__":
    main()
''',
            
            # Tests
            "tests/test_agents.py": '''"""Tests for agent modules."""

import pytest

def test_placeholder():
    """Placeholder test."""
    assert True

def test_imports():
    """Test that modules can be imported."""
    try:
        from agents.knowledge_base_agent import KnowledgeBaseAgent
        from agents.compliance_agent import ComplianceAgent
        from agents.orchestrator_agent import OrchestratorAgent
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
''',
            
            "tests/test_tools.py": '''"""Tests for tools modules."""

import pytest

def test_placeholder():
    """Placeholder test."""
    assert True

def test_tool_imports():
    """Test that tool modules can be imported."""
    try:
        from tools.knowledge_base_tools import kb_tools
        from tools.compliance_tools import compliance_tools
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
''',
        }
        
        # Create all placeholder files
        for file_path, content in placeholders.items():
            full_path = self.base_dir / file_path
            full_path.write_text(content)
            print(f"Created file: {full_path}")
    
    def create_docker_files(self):
        """Create Docker-related files."""
        # Dockerfile
        dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "examples/basic_usage.py"]
"""
        file_path = self.base_dir / "Dockerfile"
        file_path.write_text(dockerfile_content)
        print(f"Created file: {file_path}")
        
        # docker-compose.yml
        docker_compose_content = """version: '3.8'

services:
  compliance-agent:
    build: .
    container_name: compliance-agent
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    restart: unless-stopped
    command: python examples/basic_usage.py

  mcp-compliance:
    build: .
    container_name: mcp-compliance-server
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    restart: unless-stopped
    command: python mcp_servers/compliance_mcp_server.py

  mcp-knowledge-base:
    build: .
    container_name: mcp-kb-server
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    restart: unless-stopped
    command: python mcp_servers/kb_mcp_server.py
"""
        file_path = self.base_dir / "docker-compose.yml"
        file_path.write_text(docker_compose_content)
        print(f"Created file: {file_path}")
        
        # .dockerignore
        dockerignore_content = """*.pyc
__pycache__
.env
.venv
venv/
.git
.gitignore
.pytest_cache
*.egg-info
dist/
build/
.coverage
htmlcov/
.mypy_cache/
logs/*.log
"""
        file_path = self.base_dir / ".dockerignore"
        file_path.write_text(dockerignore_content)
        print(f"Created file: {file_path}")
    
    def create_git_files(self):
        """Create Git-related files."""
        gitignore_content = """# Python
*.pyc
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg
*.egg-info/
dist/
build/
develop-eggs/
.eggs/
*.manifest
*.spec

# Virtual Environment
venv/
env/
ENV/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.hypothesis/

# Environment
.env
.env.local
.env.*.local

# Logs
logs/
*.log

# Documentation
docs/_build/
site/

# Temporary files
tmp/
temp/
*.tmp
*.bak

# AWS
.aws/
"""
        file_path = self.base_dir / ".gitignore"
        file_path.write_text(gitignore_content)
        print(f"Created file: {file_path}")
    
    def create_scripts(self):
        """Create utility scripts."""
        # run_tests.sh
        run_tests_content = """#!/bin/bash
# Script to run tests

echo "Running tests..."
python -m pytest tests/ -v --cov=. --cov-report=html
echo "Tests completed. Coverage report available in htmlcov/index.html"
"""
        file_path = self.base_dir / "run_tests.sh"
        file_path.write_text(run_tests_content)
        file_path.chmod(0o755)
        print(f"Created file: {file_path}")
        
        # start_mcp_servers.sh
        start_servers_content = """#!/bin/bash
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
"""
        file_path = self.base_dir / "start_mcp_servers.sh"
        file_path.write_text(start_servers_content)
        file_path.chmod(0o755)
        print(f"Created file: {file_path}")
    
    def setup_project(self):
        """Run the complete project setup."""
        print(f"\n{'='*60}")
        print(f"Setting up Compliance Agent System in: {self.base_dir}")
        print(f"{'='*60}\n")
        
        # Create all components
        self.create_directory_structure()
        self.create_requirements_txt()
        self.create_env_example()
        self.create_readme()
        self.create_setup_py()
        self.create_config_files()
        self.create_placeholder_files()
        self.create_docker_files()
        self.create_git_files()
        self.create_scripts()
        
        print(f"\n{'='*60}")
        print("✅ Project setup complete!")
        print(f"{'='*60}\n")
        
        print("Next steps:")
        print(f"1. cd {self.base_dir}")
        print("2. python -m venv venv")
        print("3. source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print("4. pip install -r requirements.txt")
        print("5. cp .env.example .env")
        print("6. Edit .env with your AWS credentials")
        print("7. python examples/basic_usage.py")
        print("\nFor full implementation, replace placeholder files with the complete code from documentation.")
        print(f"{'='*60}\n")


def main():
    """Main entry point for the setup script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup the Compliance Agent System project structure"
    )
    parser.add_argument(
        "--dir",
        default="compliance-agent-system",
        help="Base directory for the project (default: compliance-agent-system)"
    )
    
    args = parser.parse_args()
    
    # Create and run setup
    setup = ProjectSetup(base_dir=args.dir)
    setup.setup_project()


if __name__ == "__main__":
    main()
