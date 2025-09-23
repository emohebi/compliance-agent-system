"""AWS service helper utilities."""

import boto3
from typing import Optional

def get_bedrock_client(region: str = "us-west-2"):
    """Get AWS Bedrock client."""
    return boto3.client('bedrock-runtime', region_name=region)

def get_s3_client(region: str = "us-west-2"):
    """Get AWS S3 client."""
    return boto3.client('s3', region_name=region)
