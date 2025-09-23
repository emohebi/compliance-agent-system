import boto3
import json
from datetime import datetime, timezone

class BedrockSessionManager:
    def __init__(self, profile_name, region='us-east-1'):
        self.profile_name = profile_name
        self.region = region
        self.session = boto3.Session(profile_name=profile_name)
        self.sts_client = self.session.client('sts')
        self.credentials = None
        self.bedrock_client = None
        
    def get_fresh_session_token(self, duration_seconds=3600):
        """Get a fresh session token"""
        try:
            print("🔄 Getting fresh session token...")
            response = self.sts_client.get_session_token(
                DurationSeconds=duration_seconds
            )
            
            self.credentials = response['Credentials']
            print("✅ Fresh session token obtained")
            print(f"Expires at: {self.credentials['Expiration']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error getting session token: {e}")
            return False
    
    def is_token_valid(self):
        """Check if current token is still valid"""
        if not self.credentials:
            return False
            
        expiration = self.credentials['Expiration']
        current_time = datetime.now(timezone.utc)
        
        # Add 5 minute buffer
        buffer_time = 300  # 5 minutes
        return (expiration - current_time).total_seconds() > buffer_time
    
    def create_bedrock_client(self):
        """Create or refresh Bedrock client"""
        # Check if we need a fresh token
        if not self.is_token_valid():
            print("Token expired or missing, getting fresh token...")
            if not self.get_fresh_session_token():
                return None
        
        try:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=self.region,
                aws_access_key_id=self.credentials['AccessKeyId'],
                aws_secret_access_key=self.credentials['SecretAccessKey'],
                aws_session_token=self.credentials['SessionToken']
            )
            
            print(f"✅ Bedrock client created for region: {self.region}")
            return self.bedrock_client
            
        except Exception as e:
            print(f"❌ Error creating Bedrock client: {e}")
            return None
    
    def test_bedrock_access(self):
        """Test Bedrock access by listing models"""
        if not self.bedrock_client:
            self.create_bedrock_client()
        
        try:
            # Create bedrock client for listing models (not bedrock-runtime)
            bedrock_client = boto3.client(
                'bedrock',
                region_name=self.region,
                aws_access_key_id=self.credentials['AccessKeyId'],
                aws_secret_access_key=self.credentials['SecretAccessKey'],
                aws_session_token=self.credentials['SessionToken']
            )
            
            response = bedrock_client.list_foundation_models()
            models = response.get('modelSummaries', [])
            
            print(f"✅ Bedrock access successful!")
            print(f"Found {len(models)} available models")
            
            # Show first few models
            for model in models[:3]:
                print(f"  - {model['modelId']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Bedrock access test failed: {e}")
            return False
    
    def converse_with_model(self, model_id, message):
        """Use ConverseStream with proper error handling"""
        if not self.bedrock_client:
            if not self.create_bedrock_client():
                return None
        
        try:
            response = self.bedrock_client.converse_stream(
                modelId=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": message}]
                    }
                ]
            )
            
            print("✅ ConverseStream successful!")
            return response
            
        except Exception as e:
            print(f"❌ ConverseStream failed: {e}")
            
            # If token error, try refreshing
            if "security token" in str(e).lower() or "unrecognized" in str(e).lower():
                print("🔄 Token issue detected, refreshing...")
                if self.get_fresh_session_token():
                    self.create_bedrock_client()
                    # Retry once
                    try:
                        response = self.bedrock_client.converse_stream(
                            modelId=model_id,
                            messages=[
                                {
                                    "role": "user", 
                                    "content": [{"text": message}]
                                }
                            ]
                        )
                        print("✅ ConverseStream successful after token refresh!")
                        return response
                    except Exception as retry_error:
                        print(f"❌ Retry failed: {retry_error}")
            
            return None

# Usage Example
def main():
    # Initialize with your profile and correct region
    bedrock_manager = BedrockSessionManager(
        profile_name="myprofile",  # Replace with your profile
        region="ap-southeast-2"         # Make sure Bedrock is available in this region
    )
    
    # Test Bedrock access
    if bedrock_manager.test_bedrock_access():
        # Try using ConverseStream
        model_id = "amazon.nova-pro-v1:0"  # Example model
        message = "Hello, how are you?"
        
        response = bedrock_manager.converse_with_model(model_id, message)
        
        if response:
            print("Processing response...")
            # Handle the streaming response
            for event in response['stream']:
                if 'contentBlockDelta' in event:
                    delta = event['contentBlockDelta']
                    if 'text' in delta:
                        print(delta['text'], end='', flush=True)

if __name__ == "__main__":
    main()
