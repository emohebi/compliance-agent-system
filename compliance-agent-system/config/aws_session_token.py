import boto3
from datetime import datetime
import json

class AWSSessionTokenManager:
    def __init__(self, profile_name):
        self.profile_name = profile_name
        self.session = boto3.Session(profile_name=profile_name)
        self.sts_client = self.session.client('sts')
        
    def verify_profile(self):
        """Verify the profile is working"""
        try:
            response = self.sts_client.get_caller_identity()
            print(f"✅ Profile '{self.profile_name}' is valid")
            print(f"Account ID: {response['Account']}")
            print(f"User ARN: {response['Arn']}")
            return True
        except Exception as e:
            print(f"❌ Profile verification failed: {e}")
            return False
    
    def get_session_token(self, duration_seconds=3600, mfa_serial=None, mfa_token=None):
        """
        Get session token with optional MFA
        
        :param duration_seconds: Token duration (15 minutes to 36 hours)
        :param mfa_serial: MFA device serial number/ARN (optional)
        :param mfa_token: MFA token code (required if mfa_serial provided)
        """
        try:
            # Prepare parameters
            params = {'DurationSeconds': duration_seconds}
            
            # Add MFA if provided
            if mfa_serial and mfa_token:
                params['SerialNumber'] = mfa_serial
                params['TokenCode'] = mfa_token
                print("🔐 Using MFA authentication...")
            
            # Get session token
            print("🔄 Requesting session token...")
            response = self.sts_client.get_session_token(**params)
            
            credentials = response['Credentials']
            
            print("✅ Session token obtained successfully!")
            print(f"Access Key ID: {credentials['AccessKeyId']}")
            print(f"Expiration: {credentials['Expiration']}")
            
            return credentials
            
        except Exception as e:
            print(f"❌ Error getting session token: {e}")
            return None
    
    def test_credentials(self, credentials):
        """Test the temporary credentials"""
        try:
            # Create S3 client with temporary credentials
            s3_client = boto3.client(
                's3',
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken']
            )
            
            # Test by listing buckets
            response = s3_client.list_buckets()
            
            print("✅ Credentials test successful!")
            print(f"Found {len(response['Buckets'])} S3 buckets")
            
            return True
            
        except Exception as e:
            print(f"❌ Credentials test failed: {e}")
            return False
    
    def save_credentials_to_file(self, credentials, filename="temp_credentials.json"):
        """Save credentials to a file for later use"""
        try:
            cred_data = {
                'AccessKeyId': credentials['AccessKeyId'],
                'SecretAccessKey': credentials['SecretAccessKey'],
                'SessionToken': credentials['SessionToken'],
                'Expiration': credentials['Expiration'].isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(cred_data, f, indent=2)
            
            print(f"✅ Credentials saved to {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving credentials: {e}")
            return False

# Usage Example
def main():
    # Step 1: Initialize with your profile
    profile_name = "myprofile"  # Replace with your profile name
    token_manager = AWSSessionTokenManager(profile_name)
    
    # Step 2: Verify profile works
    if not token_manager.verify_profile():
        return
    
    # Step 3: Get session token (choose one method)
    
    # Method A: Basic session token (no MFA)
    credentials = token_manager.get_session_token(duration_seconds=3600)
    
    # Method B: Session token with MFA (uncomment if you have MFA)
    # mfa_serial = "arn:aws:iam::123456789012:mfa/yourusername"
    # mfa_code = input("Enter MFA code: ")
    # credentials = token_manager.get_session_token(
    #     duration_seconds=3600,
    #     mfa_serial=mfa_serial,
    #     mfa_token=mfa_code
    # )
    
    if credentials:
        # Step 4: Test the credentials
        # token_manager.test_credentials(credentials)
        
        # Step 5: Save credentials for later use
        token_manager.save_credentials_to_file(credentials)
        
        print("\n📋 You can now use these credentials in your applications:")
        print(f"AWS_ACCESS_KEY_ID={credentials['AccessKeyId']}")
        print(f"AWS_SECRET_ACCESS_KEY={credentials['SecretAccessKey']}")
        print(f"AWS_SESSION_TOKEN={credentials['SessionToken']}")

if __name__ == "__main__":
    main()
