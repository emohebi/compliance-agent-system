"""Configuration settings for the compliance agent system."""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
from config.aws_session_token import AWSSessionTokenManager

# Load environment variables
if (Path(__file__).parent.parent / '.env').is_file():
    load_dotenv(Path(__file__).parent.parent / '.env', override=True)
    print(f"Loaded .env from {Path(__file__).parent.parent / '.env'}")
else:
    print("No .env file found, using system environment variables")
    raise FileNotFoundError(".env file not found in the project root")

# Step 1: Initialize with your profile
profile_name = os.getenv("AWS_PROFILE", "myprofile")  # Get from env or use default
token_manager = AWSSessionTokenManager(profile_name)

# Step 2: Verify profile works
if not token_manager.verify_profile():
    raise ValueError(f"Profile '{profile_name}' verification failed")

# Step 3: Get session token duration from environment or use default
session_duration = int(os.getenv("AWS_SESSION_DURATION", "3600"))

# Method A: Basic session token (no MFA)
credentials = token_manager.get_session_token(duration_seconds=session_duration)

# Method B: Session token with MFA (uncomment if you have MFA)
# mfa_serial = os.getenv("AWS_MFA_SERIAL")  # Get from environment
# if mfa_serial:
#     mfa_code = input("Enter MFA code: ")
#     credentials = token_manager.get_session_token(
#         duration_seconds=session_duration,
#         mfa_serial=mfa_serial,
#         mfa_token=mfa_code
#     )
# else:
#     credentials = token_manager.get_session_token(duration_seconds=session_duration)

if not credentials:
    raise ValueError("Failed to obtain temporary AWS session credentials")


class Settings:
    """Application settings."""
    
    # Knowledge Base Mode
    USE_LOCAL_KNOWLEDGE_BASE: bool = os.getenv("USE_LOCAL_KNOWLEDGE_BASE", "true").lower() == "true"
    LOCAL_KB_FILE: str = os.getenv("LOCAL_KB_FILE", "./data/local_knowledge_base.json")
    
    # AWS Configuration (optional when using local KB)
    AWS_REGION: str = os.getenv("AWS_REGION", "us-west-2")
    AWS_ACCESS_KEY_ID: Optional[str] = credentials['AccessKeyId']
    AWS_SECRET_ACCESS_KEY: Optional[str] = credentials['SecretAccessKey']
    AWS_SESSION_TOKEN: Optional[str] = credentials['SessionToken']
    
    # Bedrock Configuration (optional when using local KB)
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
        """Get Bedrock model configuration with proper AWS credentials."""
        import boto3
        
        config = {
            "temperature": cls.AGENT_TEMPERATURE,
            "max_tokens": cls.AGENT_MAX_TOKENS,
            "streaming": True
        }
        
        # Create a boto3 session with temporary credentials if available
        if cls.AWS_ACCESS_KEY_ID and cls.AWS_SECRET_ACCESS_KEY:
            # Create boto3 session with temporary credentials
            boto_session = boto3.Session(
                aws_access_key_id=cls.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=cls.AWS_SECRET_ACCESS_KEY,
                aws_session_token=cls.AWS_SESSION_TOKEN,  # Important for temporary credentials!
                region_name=cls.AWS_REGION
            )
            
            config.update({
                "model_id": cls.BEDROCK_MODEL_ID,
                "cache_prompt": cls.BEDROCK_CACHE_PROMPT,
                "boto_session": boto_session  # Pass the session, not individual credentials!
            })
        elif not cls.USE_LOCAL_KNOWLEDGE_BASE:
            # If AWS mode but no credentials, this is a problem
            print("WARNING: AWS mode selected but no AWS credentials available!")
            raise ValueError("AWS credentials required for Bedrock model when not using local KB mode")
        
        return config
    
    @classmethod
    def get_kb_mode(cls) -> str:
        """Get current knowledge base mode."""
        return "local" if cls.USE_LOCAL_KNOWLEDGE_BASE else "aws"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration based on mode."""
        if cls.USE_LOCAL_KNOWLEDGE_BASE:
            # Local mode - minimal requirements
            return True
        else:
            # AWS mode - need credentials and KB ID
            if not cls.BEDROCK_KNOWLEDGE_BASE_ID:
                print("Warning: AWS mode selected but BEDROCK_KNOWLEDGE_BASE_ID not set")
                return False
            if not cls.AWS_ACCESS_KEY_ID or not cls.AWS_SECRET_ACCESS_KEY:
                print("Warning: AWS mode selected but AWS credentials not available")
                return False
            return True

settings = Settings()

# Validate configuration on import
if not settings.validate_config():
    print(f"Configuration validation failed for {settings.get_kb_mode()} mode")